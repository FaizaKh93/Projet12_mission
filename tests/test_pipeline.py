"""Le pipeline MLOps : préparation, découpage, orchestration.

**Pourquoi ces tests tournent sans `data/`.** Les fonctions de préparation sont
pures : elles prennent des tables, en rendent d'autres. On peut donc leur
donner six tables minuscules, construites ici, et vérifier *les règles* —
qu'un rendement négatif disparaît, qu'une pluie illisible devient une absence,
qu'une culture sans référence FAO garde ses `NaN`.

C'est une tout autre chose que les tests de parité de `test_preprocessing.py`,
qui comparent une reconstruction aux colonnes que le dataset porte lui-même :
ceux-là exigent le vrai dataset, et sont réservés à l'environnement MLOps.
Ici, la vérité de référence est la règle énoncée, pas une donnée observée.

Un test manque volontairement à l'appel : « le pipeline reproduit le dataset
livré ». Il demande le million de lignes, et c'est
`scripts/prepare_data.py --verifier` qui le porte.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from agritech import decoupage, pipeline, preparation
from agritech.config import FEATURES_MODELE, SPLIT_PROPORTIONS, TARGET

# --------------------------------------------------------------------------- #
# Sources synthétiques, fidèles au schéma des six fichiers
# --------------------------------------------------------------------------- #

PAYS = ["France", "Spain", "Italy"]
ANNEES = list(range(2000, 2011))
#: Les quatre cultures FAO, dans leur vocabulaire d'origine.
ITEMS = ["Maize", "Wheat", "Rice, paddy", "Soybeans"]
#: Les six cultures du jeu de parcelles. Orge et coton n'ont pas de référence.
CULTURES = ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"]


def _parcelles(n_par_culture: int = 40) -> pd.DataFrame:
    """Un jeu de parcelles couvrant les six cultures et toutes les modalités."""
    lignes = []
    for i, culture in enumerate(CULTURES):
        for j in range(n_par_culture):
            k = i * n_par_culture + j
            lignes.append(
                {
                    "Region": ["East", "North", "South", "West"][k % 4],
                    "Soil_Type": ["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"][k % 6],
                    "Crop": culture,
                    "Rainfall_mm": 150.0 + 100.0 * (k % 9),
                    "Temperature_Celsius": 16.0 + 2.0 * (k % 12),
                    "Fertilizer_Used": bool(k % 2),
                    "Irrigation_Used": bool((k // 2) % 2),
                    "Weather_Condition": ["Cloudy", "Rainy", "Sunny"][k % 3],
                    "Days_to_Harvest": 90 + (k % 40),
                    TARGET: 1.0 + (k % 70) / 10,
                }
            )
    return pd.DataFrame(lignes)


def _fao() -> pd.DataFrame:
    """`yield.csv` : rendements FAO en hectogrammes par hectare."""
    lignes = []
    for pays in PAYS:
        for item in ITEMS:
            for annee in ANNEES:
                lignes.append(
                    {
                        "Domain Code": "QCL",
                        "Domain": "Crops",
                        "Area Code": PAYS.index(pays),
                        "Area": pays,
                        "Element Code": 5419,
                        "Element": "Yield",
                        "Item Code": ITEMS.index(item),
                        "Item": item,
                        "Year Code": annee,
                        "Year": annee,
                        "Unit": "hg/ha",
                        # Niveaux distincts par culture, et une tendance par année
                        "Value": 20_000 * (ITEMS.index(item) + 1) + 300 * (annee - 2000),
                    }
                )
    return pd.DataFrame(lignes)


def _pesticides() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Domain": "Pesticides", "Area": pays, "Element": "Use",
             "Item": "Pesticides", "Year": annee, "Unit": "tonnes",
             "Value": 100 + 10 * PAYS.index(pays)}
            for pays in PAYS
            for annee in ANNEES
        ]
    )


def _pluie(avec_valeurs_illisibles: bool = True) -> pd.DataFrame:
    """`rainfall.csv`, dont la colonne porte un espace initial dans son nom."""
    lignes = []
    for pays in PAYS:
        for annee in ANNEES:
            valeur = str(500 + 20 * PAYS.index(pays))
            if avec_valeurs_illisibles and annee == 2003:
                valeur = ".."  # masque une absence sous une chaîne
            lignes.append({" Area": pays, "Year": annee,
                           "average_rain_fall_mm_per_year": valeur})
    return pd.DataFrame(lignes)


def _temperatures() -> pd.DataFrame:
    """`temp.csv` porte plusieurs relevés par (pays, année) : d'où l'agrégation."""
    lignes = []
    for pays in PAYS:
        for annee in ANNEES:
            for releve in range(3):
                lignes.append({"year": annee, "country": pays,
                               "avg_temp": 12.0 + releve + PAYS.index(pays)})
    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def sources() -> preparation.Sources:
    """Les six tables, en mémoire. Aucun fichier n'est lu."""
    return preparation.Sources(
        parcelles=_parcelles(),
        fao=_fao(),
        pesticides=_pesticides(),
        pluie=_pluie(),
        temperatures=_temperatures(),
        fusion_fournie=pd.DataFrame(),
    )


@pytest.fixture(scope="module")
def preparee(sources) -> preparation.Preparation:
    """Le résultat complet de la chaîne, partagé par tout le module."""
    return preparation.preparer(sources)


# --------------------------------------------------------------------------- #
# Nettoyage
# --------------------------------------------------------------------------- #


def test_un_rendement_negatif_est_retire():
    """Un rendement négatif est physiquement impossible : il sort."""
    parcelles = _parcelles(n_par_culture=2)
    parcelles.loc[0, TARGET] = -1.0

    nettoyees = preparation.nettoyer_parcelles(parcelles)

    assert len(nettoyees) == len(parcelles) - 1
    assert (nettoyees[TARGET] >= 0).all()


def test_une_cible_manquante_est_conservee():
    """`NaN` n'est pas négatif : une absence se signale, elle ne se supprime pas.

    Le filtre s'écrit `~(< 0)` et non `>= 0`. Les deux diffèrent exactement ici,
    puisqu'une comparaison avec `NaN` est toujours fausse — et retirer
    silencieusement des lignes changerait le dataset livré.
    """
    parcelles = _parcelles(n_par_culture=2)
    parcelles.loc[0, TARGET] = np.nan

    nettoyees = preparation.nettoyer_parcelles(parcelles)

    assert len(nettoyees) == len(parcelles)
    assert nettoyees[TARGET].isna().sum() == 1


def test_les_temperatures_sont_agregees_par_pays_et_annee():
    """Trois relevés par couple deviennent un seul, sans quoi la jointure duplique."""
    agregees = preparation.agreger_temperatures(_temperatures())

    assert len(agregees) == len(PAYS) * len(ANNEES)
    assert not agregees.duplicated(subset=["Area", "Year"]).any()
    assert list(agregees.columns) == ["Area", "Year", "avg_temp"]


def test_un_couple_sans_aucune_mesure_disparait():
    """Une température entièrement absente ne produit pas une ligne vide."""
    brut = _temperatures()
    brut.loc[brut["country"] == "Italy", "avg_temp"] = np.nan

    agregees = preparation.agreger_temperatures(brut)

    assert "Italy" not in set(agregees["Area"])
    assert agregees["avg_temp"].notna().all()


def test_une_pluie_non_numerique_devient_une_absence():
    """Les valeurs masquées en texte sont révélées, puis écartées.

    Sans la conversion, la colonne resterait du texte et la comparaison
    numérique en aval échouerait — ou pire, passerait silencieusement.
    """
    nettoyee = preparation.nettoyer_pluie(_pluie())

    assert list(nettoyee.columns) == ["Area", "Year", "rain_mm"]
    assert pd.api.types.is_numeric_dtype(nettoyee["rain_mm"])
    # Une année sur onze était illisible, pour chacun des trois pays
    assert len(nettoyee) == len(PAYS) * (len(ANNEES) - 1)


def test_le_nom_de_colonne_avec_espace_initial_est_repris():
    """`rainfall.csv` nomme sa première colonne ` Area`, espace compris."""
    assert " Area" in _pluie().columns
    assert "Area" in preparation.nettoyer_pluie(_pluie()).columns


def test_les_rendements_sont_convertis_en_tonnes():
    """20 000 hg/ha font 2 t/ha : l'unité FAO n'est pas celle des parcelles."""
    rendements = preparation.nettoyer_rendements(_fao())

    assert (rendements["yield_t_ha"] == rendements["yield_hg_ha"] / 10_000).all()
    assert (rendements["yield_t_ha"] > 0).all()


# --------------------------------------------------------------------------- #
# Panel et référentiel
# --------------------------------------------------------------------------- #


def test_le_panel_ne_duplique_aucun_rendement(sources):
    """Autant de lignes qu'il y a de rendements : la jointure n'en crée pas."""
    rendements = preparation.nettoyer_rendements(sources.fao)
    panel = preparation.construire_panel(
        rendements,
        preparation.agreger_temperatures(sources.temperatures),
        preparation.nettoyer_pluie(sources.pluie),
        preparation.nettoyer_pesticides(sources.pesticides),
    )

    assert len(panel) == len(rendements)


def test_un_doublon_dans_une_table_annexe_est_refuse(sources):
    """Mieux vaut échouer que gonfler le panel sans que rien ne le signale."""
    temperatures = preparation.agreger_temperatures(sources.temperatures)
    doublees = pd.concat([temperatures, temperatures.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="doublons"):
        preparation.construire_panel(
            preparation.nettoyer_rendements(sources.fao),
            doublees,
            preparation.nettoyer_pluie(sources.pluie),
            preparation.nettoyer_pesticides(sources.pesticides),
        )


def test_la_reference_porte_une_ligne_par_culture_fao(preparee):
    """Quatre cultures FAO, quatre lignes, et les indicateurs attendus."""
    reference = preparee.reference

    assert sorted(reference.index) == sorted(ITEMS)
    for colonne in preparation.COLONNES_REFERENCE:
        assert colonne in reference.columns


def test_l_indice_de_productivite_est_rapporte_au_pays(preparee):
    """La médiane des ratios intra-pays vaut 1 pour la culture médiane.

    Un classement mondial brut refléterait la richesse agricole des pays où
    chaque culture est répandue ; l'indice intra-pays neutralise cet effet.
    """
    indice = preparee.reference["crop_productivity_index"]

    assert (indice > 0).all()
    # Quatre cultures de niveaux réguliers : l'indice les ordonne pareillement
    assert indice.idxmax() == "Soybeans"
    assert indice.idxmin() == "Maize"


def test_le_vocabulaire_fao_est_traduit(preparee):
    """« Rice, paddy » devient « Rice » ; ce qui n'a pas d'équivalent sort."""
    traduite = preparation.traduire_reference(preparee.reference)

    assert sorted(traduite.index) == ["Maize", "Rice", "Soybean", "Wheat"]


# --------------------------------------------------------------------------- #
# Enrichissement et livrable
# --------------------------------------------------------------------------- #


def test_les_cultures_sans_reference_gardent_leurs_nan(preparee):
    """L'orge et le coton n'ont pas de référence FAO : on ne fabrique rien.

    Substituer zéro ne lèverait aucune erreur et fausserait les prédictions
    d'un tiers des cultures.
    """
    dataset = preparee.dataset
    sans_reference = dataset[~dataset["crop_in_fao_reference"]]

    assert set(sans_reference["Crop"]) == {"Barley", "Cotton"}
    assert sans_reference[preparation.COLONNES_DIFFUSEES].isna().all().all()


def test_les_cultures_referencees_sont_completes(preparee):
    """À l'inverse, aucune absence là où la référence existe."""
    dataset = preparee.dataset
    referencees = dataset[dataset["crop_in_fao_reference"]]

    assert referencees[preparation.COLONNES_DIFFUSEES].notna().all().all()


def test_le_seuil_hydrique_combine_pluie_et_irrigation(preparee):
    """`low_rain_no_irrigation` : pluie sous le seuil **et** absence d'irrigation."""
    dataset = preparee.dataset
    attendu = (dataset["Rainfall_mm"] < 400) & ~dataset["Irrigation_Used"]

    assert dataset["low_rain_no_irrigation"].equals(attendu)


def test_l_enrichissement_ne_duplique_aucune_parcelle(sources, preparee):
    """Une ligne de parcelle doit rester une ligne de parcelle."""
    assert len(preparee.dataset) == len(sources.parcelles)


def test_une_reference_non_unique_est_refusee(sources, preparee):
    """Une clé dupliquée ferait exploser le nombre de lignes : on le détecte."""
    traduite = preparation.traduire_reference(preparee.reference)
    doublee = pd.concat([traduite, traduite.head(1)])

    with pytest.raises(ValueError, match="dupliqué"):
        preparation.enrichir(sources.parcelles, doublee)


def test_le_livrable_porte_les_treize_variables_et_la_cible(preparee):
    """Ni plus, ni moins : c'est le contrat que le modèle attend."""
    colonnes = set(preparee.dataset.columns)

    assert colonnes == set(FEATURES_MODELE) | {TARGET}


def test_l_ordre_des_colonnes_est_celui_du_livrable(sources, preparee):
    """L'ordre fixe les octets du fichier, donc `data_version`.

    Les deux retraits ont lieu dans l'ordre du notebook — redondantes d'abord,
    non retenues ensuite. Inverser les deux donnerait les mêmes colonnes dans
    une autre disposition, et une empreinte différente.
    """
    enrichi = preparation.enrichir(
        preparation.nettoyer_parcelles(sources.parcelles),
        preparation.traduire_reference(preparee.reference),
    )
    attendu = [c for c in enrichi.columns
               if c not in preparation.COLONNES_REDONDANTES
               and c not in preparation.COLONNES_NON_RETENUES]

    assert list(preparee.dataset.columns) == attendu


def test_les_colonnes_ecartees_ont_disparu(preparee):
    """Les quatre colonnes retirées par l'analyse ne sont plus là."""
    ecartees = preparation.COLONNES_REDONDANTES + preparation.COLONNES_NON_RETENUES

    assert not set(ecartees) & set(preparee.dataset.columns)


def test_controler_accepte_un_livrable_sain(preparee):
    """Le bilan compte ce qu'il annonce."""
    bilan = preparation.controler(preparee)

    assert bilan["lignes"] == len(preparee.dataset)
    assert bilan["rendements_negatifs"] == 0
    assert bilan["lignes_hors_referentiel"] == 2 * 40  # orge et coton


def test_controler_refuse_une_absence_inattendue(preparee):
    """Une valeur manquante hors des cultures sans référence doit lever."""
    abime = preparee.dataset.copy()
    abime.loc[abime.index[0], "Rainfall_mm"] = np.nan
    suspect = preparation.Preparation(
        dataset=abime, panel=preparee.panel,
        reference=preparee.reference, dictionnaire=preparee.dictionnaire,
    )

    with pytest.raises(ValueError, match="manquantes inattendues"):
        preparation.controler(suspect)


def test_controler_refuse_une_colonne_non_documentee(preparee):
    """Le dictionnaire des variables doit décrire tout le livrable."""
    abime = preparee.dataset.copy()
    abime["colonne_surprise"] = 1
    suspect = preparation.Preparation(
        dataset=abime, panel=preparee.panel,
        reference=preparee.reference, dictionnaire=preparee.dictionnaire,
    )

    with pytest.raises(ValueError, match="non documentées"):
        preparation.controler(suspect)


def test_le_dictionnaire_couvre_le_livrable(preparee):
    """Chaque colonne livrée porte une provenance et une unité."""
    dictionnaire = preparee.dictionnaire

    assert set(dictionnaire.index) == set(preparee.dataset.columns)
    assert dictionnaire["provenance"].notna().all()


# --------------------------------------------------------------------------- #
# Découpage
# --------------------------------------------------------------------------- #


@pytest.fixture
def partitions(preparee):
    return decoupage.decouper(preparee.dataset)


def test_les_proportions_visees_sont_tenues(partitions, preparee):
    """70 / 15 / 15, à l'arrondi près sur un petit effectif."""
    total = len(preparee.dataset)

    for nom, attendue in SPLIT_PROPORTIONS.items():
        assert abs(len(partitions[nom]) / total - attendue) < 0.01


def test_aucune_ligne_n_appartient_a_deux_partitions(partitions, preparee):
    """Une fuite de `train` vers `test` fausserait l'évaluation finale."""
    repartition = decoupage.controler(partitions, preparee.dataset)

    assert set(repartition) == {"train", "val", "test"}


def test_un_chevauchement_est_detecte(partitions, preparee):
    """Le contrôle sert à quelque chose : on le met en défaut."""
    fautif = dict(partitions)
    fautif["val"] = pd.concat([partitions["val"], partitions["train"].head(1)])

    with pytest.raises(ValueError, match="communes"):
        decoupage.controler(fautif, preparee.dataset)


def test_une_ligne_perdue_est_detectee(partitions, preparee):
    """Une partition amputée ne couvre plus le dataset."""
    fautif = dict(partitions)
    fautif["train"] = partitions["train"].iloc[1:]

    with pytest.raises(ValueError, match="aucune partition"):
        decoupage.controler(fautif, preparee.dataset)


def test_le_decoupage_est_deterministe(preparee):
    """Deux appels donnent les mêmes lignes : la graine est fixée."""
    premier = decoupage.decouper(preparee.dataset)
    second = decoupage.decouper(preparee.dataset)

    for nom in premier:
        assert list(premier[nom].index) == list(second[nom].index)


def test_la_stratification_preserve_les_cultures(partitions, preparee):
    """Chaque partition porte les six cultures, en proportions comparables."""
    attendues = set(preparee.dataset["Crop"])

    for partition in partitions.values():
        assert set(partition["Crop"]) == attendues


# --------------------------------------------------------------------------- #
# Décision d'écriture
# --------------------------------------------------------------------------- #


def _rediriger(monkeypatch, dossier):
    """Fait écrire le module de découpage dans un dossier temporaire."""
    chemins = {nom: dossier / f"{nom}.parquet" for nom in ("train", "val", "test")}
    metadonnees = dossier / "split_metadata.json"

    monkeypatch.setattr(decoupage, "CHEMINS", chemins)
    monkeypatch.setattr(decoupage, "ARTEFACTS", {**chemins, "metadata": metadonnees})
    monkeypatch.setattr(decoupage, "SPLIT_METADATA_PATH", metadonnees)
    monkeypatch.setattr(decoupage, "SPLITS_DIR", dossier)
    return chemins, metadonnees


def test_ecrire_quand_rien_n_existe(monkeypatch, tmp_path):
    """Premier découpage : rien à protéger, on écrit."""
    _rediriger(monkeypatch, tmp_path)

    decision = decoupage.decider("abcdef123456")

    assert decision.ecrire
    assert "aucun artefact" in decision.motif


def test_un_lot_complet_et_a_jour_n_est_pas_reecrit(monkeypatch, tmp_path, partitions, preparee):
    """Le cas courant : les partitions décrivent le dataset, on n'y touche pas."""
    _rediriger(monkeypatch, tmp_path)
    decoupage.ecrire(partitions, preparee.dataset, "abcdef123456")

    decision = decoupage.decider("abcdef123456")

    assert not decision.ecrire
    assert "à jour" in decision.motif


def test_un_lot_incomplet_est_refuse(monkeypatch, tmp_path, partitions, preparee):
    """Vestige d'une exécution interrompue : on refuse de deviner."""
    chemins, _ = _rediriger(monkeypatch, tmp_path)
    decoupage.ecrire(partitions, preparee.dataset, "abcdef123456")
    chemins["val"].unlink()

    with pytest.raises(RuntimeError, match="État partiel"):
        decoupage.decider("abcdef123456")


def test_des_partitions_perimees_sont_refusees(monkeypatch, tmp_path, partitions, preparee):
    """Elles ont été générées pour un autre dataset : le dire, pas l'ignorer."""
    _rediriger(monkeypatch, tmp_path)
    decoupage.ecrire(partitions, preparee.dataset, "abcdef123456")

    with pytest.raises(RuntimeError, match="PÉRIMÉES"):
        decoupage.decider("000000000000")


def test_forcer_passe_outre(monkeypatch, tmp_path, partitions, preparee):
    """L'écrasement reste possible, mais il faut le demander."""
    _rediriger(monkeypatch, tmp_path)
    decoupage.ecrire(partitions, preparee.dataset, "abcdef123456")

    decision = decoupage.decider("000000000000", forcer=True)

    assert decision.ecrire


def test_les_metadonnees_decrivent_les_partitions(monkeypatch, tmp_path, partitions, preparee):
    """Effectifs, colonnes, graine, empreintes : tout ce qu'un run MLflow cite."""
    _, chemin = _rediriger(monkeypatch, tmp_path)
    metadonnees = decoupage.ecrire(partitions, preparee.dataset, "abcdef123456")

    sur_disque = json.loads(chemin.read_text(encoding="utf-8"))
    assert sur_disque == metadonnees
    assert metadonnees["data_version"] == "abcdef123456"
    assert metadonnees["effectifs"] == {n: len(d) for n, d in partitions.items()}
    assert metadonnees["colonnes"] == list(preparee.dataset.columns)
    assert len(metadonnees["split_version"]) == 12


def test_relire_detecte_un_fichier_reecrit(monkeypatch, tmp_path, partitions, preparee):
    """Un parquet remplacé hors de la chaîne change `split_version`."""
    chemins, _ = _rediriger(monkeypatch, tmp_path)
    metadonnees = decoupage.ecrire(partitions, preparee.dataset, "abcdef123456")

    partitions["test"].iloc[:-1].to_parquet(chemins["test"], index=False)

    with pytest.raises(ValueError, match="lignes sur le disque"):
        decoupage.relire_et_verifier(metadonnees)


def test_relire_detecte_des_valeurs_differentes(monkeypatch, tmp_path, partitions, preparee):
    """Mêmes effectifs, autres valeurs : le contrôle porte aussi sur la cible."""
    chemins, _ = _rediriger(monkeypatch, tmp_path)
    metadonnees = decoupage.ecrire(partitions, preparee.dataset, "abcdef123456")

    falsifiee = partitions["test"].copy()
    falsifiee[TARGET] = falsifiee[TARGET] + 1.0
    falsifiee.to_parquet(chemins["test"], index=False)
    metadonnees["split_version"] = decoupage.calculer_split_version()

    with pytest.raises(ValueError, match="diffère de celle calculée"):
        decoupage.relire_et_verifier(metadonnees, partitions)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def test_les_etapes_sont_dans_l_ordre():
    """L'ordre n'est pas décoratif : chaque étape consomme la précédente."""
    assert pipeline.ETAPES == ["preparation", "decoupage", "entrainement", "export"]
    assert set(pipeline.ACTIONS) == set(pipeline.ETAPES)


def test_une_etape_de_depart_posterieure_est_refusee():
    """Demander « du découpage à la préparation » n'a pas de sens."""
    with pytest.raises(ValueError, match="vient après"):
        pipeline.executer(depuis="decoupage", jusqua="preparation")


def test_une_etape_inconnue_est_refusee():
    with pytest.raises(ValueError, match="Étapes connues"):
        pipeline.executer(depuis="nettoyage", jusqua="export")


def test_ecrire_preparation_produit_les_cinq_fichiers(preparee, tmp_path):
    """Deux formats pour le livrable, trois artefacts reconstructibles."""
    processed = tmp_path / "processed"
    interim = tmp_path / "interim"

    chemins = pipeline.ecrire_preparation(preparee, processed=processed, interim=interim)

    assert set(chemins) == {"csv", "parquet", "reference", "panel", "dictionnaire"}
    for chemin in chemins.values():
        assert chemin.exists() and chemin.stat().st_size > 0


def test_le_csv_et_le_parquet_portent_la_meme_chose(preparee, tmp_path):
    """Le CSV est le livrable de la mission ; le Parquet, ce que la chaîne lit."""
    chemins = pipeline.ecrire_preparation(
        preparee, processed=tmp_path / "p", interim=tmp_path / "i"
    )

    relu_parquet = pd.read_parquet(chemins["parquet"])
    relu_csv = pd.read_csv(chemins["csv"])

    assert list(relu_csv.columns) == list(relu_parquet.columns)
    assert len(relu_csv) == len(relu_parquet)
    # Le Parquet préserve les types ; le CSV les redevine
    assert relu_parquet.dtypes.equals(preparee.dataset.dtypes)


# --------------------------------------------------------------------------- #
# Les étapes, rebranchées sur un dossier temporaire
#
# Les constantes de chemin sont lues **à l'appel** et non figées dans les
# signatures : les remplacer sur le module suffit donc, et aucun de ces tests
# ne touche au vrai `data/`.
# --------------------------------------------------------------------------- #


@pytest.fixture
def chaine(monkeypatch, tmp_path, sources):
    """Un pipeline complet, isolé : chemins, sources et scripts."""
    processed, interim = tmp_path / "processed", tmp_path / "interim"
    chemins = {
        "CONSOLIDATED_PATH": processed / "consolide.csv",
        "CONSOLIDATED_PARQUET_PATH": processed / "consolide.parquet",
        "CROP_REFERENCE_PATH": interim / "crop_reference.csv",
        "FAO_PANEL_PATH": interim / "fao_panel.csv",
        "DATA_DICTIONARY_PATH": interim / "data_dictionary.csv",
        "PROCESSED_DIR": processed,
        "INTERIM_DIR": interim,
        "API_METADATA_PATH": tmp_path / "modele_api.json",
        "SPLIT_METADATA_PATH": tmp_path / "split_metadata.json",
        "PROJECT_ROOT": tmp_path,
    }
    for nom, valeur in chemins.items():
        monkeypatch.setattr(pipeline, nom, valeur)

    monkeypatch.setattr(
        decoupage, "CONSOLIDATED_PARQUET_PATH", chemins["CONSOLIDATED_PARQUET_PATH"]
    )
    monkeypatch.setattr(preparation, "charger_sources", lambda: sources)
    _rediriger(monkeypatch, tmp_path)

    (tmp_path / "scripts").mkdir()
    return chemins


def test_sans_livrable_la_preparation_le_construit(chaine):
    """Rien sur le disque : la chaîne produit les cinq fichiers."""
    resultat = pipeline.lancer_preparation()

    assert resultat.agi
    assert resultat.bilan["lignes"] == 240
    assert chaine["CONSOLIDATED_PARQUET_PATH"].exists()
    assert chaine["FAO_PANEL_PATH"].exists()


def test_un_livrable_a_jour_n_est_pas_reecrit(chaine):
    """Deuxième passage : le contenu est vérifié, le fichier n'est pas touché."""
    pipeline.lancer_preparation()
    avant = chaine["CONSOLIDATED_PARQUET_PATH"].read_bytes()

    resultat = pipeline.lancer_preparation()

    assert not resultat.agi
    assert chaine["CONSOLIDATED_PARQUET_PATH"].read_bytes() == avant


def test_un_livrable_different_bloque_l_ecriture(chaine, monkeypatch, sources):
    """Le cas qui justifie tout : écraser périmerait les runs MLflow."""
    pipeline.lancer_preparation()

    autres = dataclasses.replace(sources, parcelles=sources.parcelles.iloc[1:])
    monkeypatch.setattr(preparation, "charger_sources", lambda: autres)

    with pytest.raises(RuntimeError, match="diffère de celui du disque"):
        pipeline.lancer_preparation()


def test_forcer_reecrit_un_livrable_different(chaine, monkeypatch, sources):
    """L'écrasement reste possible, mais il faut l'avoir demandé."""
    pipeline.lancer_preparation()

    autres = dataclasses.replace(sources, parcelles=sources.parcelles.iloc[1:])
    monkeypatch.setattr(preparation, "charger_sources", lambda: autres)
    resultat = pipeline.lancer_preparation(forcer=True)

    assert resultat.agi
    assert resultat.bilan["lignes"] == 239


def test_verifier_ne_signale_aucune_difference(chaine):
    """Contenu et octets concordent quand rien n'a changé entre-temps."""
    pipeline.lancer_preparation()

    rapport = pipeline.verifier_preparation()

    assert rapport["contenu_identique"]
    assert rapport["octets_identiques"]
    assert rapport["data_version_obtenue"] == rapport["data_version_attendue"]


def test_un_livrable_incomplet_est_signale(chaine):
    """Quatre fichiers sur cinq : l'étape n'est pas à jour."""
    pipeline.lancer_preparation()
    chaine["FAO_PANEL_PATH"].unlink()

    a_jour, detail = pipeline.etat_preparation()

    assert not a_jour
    assert "incomplet" in detail


def test_un_livrable_partiel_est_refuse(chaine):
    """Et surtout : il n'est pas reconstruit en silence.

    Les cinq fichiers forment un lot. S'il en manque, les autres peuvent
    décrire un autre dataset — les écraser changerait `data_version` sans que
    rien ne le signale. Même discipline que pour les partitions.
    """
    pipeline.lancer_preparation()
    chaine["FAO_PANEL_PATH"].unlink()
    empreinte_avant = chaine["CONSOLIDATED_PARQUET_PATH"].read_bytes()

    with pytest.raises(RuntimeError, match="LIVRABLE PARTIEL"):
        pipeline.lancer_preparation()

    # Le refus n'a touché à rien
    assert chaine["CONSOLIDATED_PARQUET_PATH"].read_bytes() == empreinte_avant
    assert not chaine["FAO_PANEL_PATH"].exists()


def test_le_refus_nomme_les_fichiers_manquants(chaine):
    """Le diagnostic doit suffire à agir sans relire le code."""
    pipeline.lancer_preparation()
    chaine["FAO_PANEL_PATH"].unlink()
    chaine["CROP_REFERENCE_PATH"].unlink()

    with pytest.raises(RuntimeError) as refus:
        pipeline.lancer_preparation()

    message = str(refus.value)
    assert "3/5" in message
    assert chaine["FAO_PANEL_PATH"].name in message
    assert chaine["CROP_REFERENCE_PATH"].name in message


def test_forcer_reconstruit_un_livrable_partiel(chaine):
    """L'issue existe, mais il faut l'avoir demandée."""
    pipeline.lancer_preparation()
    chaine["FAO_PANEL_PATH"].unlink()

    resultat = pipeline.lancer_preparation(forcer=True)

    assert resultat.agi
    assert chaine["FAO_PANEL_PATH"].exists()


def test_sans_aucun_livrable_l_etape_n_est_pas_a_jour(chaine):
    a_jour, detail = pipeline.etat_preparation()

    assert not a_jour
    assert detail == "aucun livrable"


def test_le_decoupage_s_enchaine_sur_la_preparation(chaine):
    """La deuxième étape lit ce que la première a écrit, puis se stabilise."""
    pipeline.lancer_preparation()

    premier = pipeline.lancer_decoupage()
    second = pipeline.lancer_decoupage()

    assert premier.agi
    assert not second.agi
    assert "split_version" in second.detail


def test_executer_deroule_les_etapes_demandees(chaine):
    """L'enchaînement par défaut s'arrête au découpage."""
    resultats = pipeline.executer(jusqua="decoupage")

    assert [r.etape for r in resultats] == ["preparation", "decoupage"]
    assert all(r.agi for r in resultats)


# --------------------------------------------------------------------------- #
# Export et sous-processus
# --------------------------------------------------------------------------- #


def test_sans_artefact_l_export_n_est_pas_a_jour(chaine):
    a_jour, detail = pipeline.etat_export()

    assert not a_jour
    assert "aucun artefact" in detail


def test_sans_partitions_l_export_n_est_pas_a_jour(chaine):
    chaine["API_METADATA_PATH"].write_text("{}", encoding="utf-8")

    a_jour, detail = pipeline.etat_export()

    assert not a_jour
    assert "aucune partition" in detail


def _poser_provenance(chemins: dict, split_version: str) -> None:
    chemins["API_METADATA_PATH"].write_text(
        json.dumps({"split_version": split_version}), encoding="utf-8"
    )


def test_l_artefact_est_a_jour_s_il_cite_les_bonnes_partitions(chaine):
    """C'est `split_version` qui lie l'artefact servi aux données lues."""
    pipeline.executer(jusqua="decoupage")
    _poser_provenance(chaine, decoupage.metadonnees_du_disque()["split_version"])

    a_jour, _ = pipeline.etat_export()

    assert a_jour


def test_un_artefact_sur_d_autres_partitions_est_perime(chaine):
    """L'écart est nommé : on lit les deux empreintes dans le message."""
    pipeline.executer(jusqua="decoupage")
    _poser_provenance(chaine, "000000000000")

    a_jour, detail = pipeline.etat_export()

    assert not a_jour
    assert "000000000000" in detail


def test_un_export_a_jour_est_ignore(chaine):
    """Ré-exporter un artefact déjà conforme ne sert à rien."""
    pipeline.executer(jusqua="decoupage")
    _poser_provenance(chaine, decoupage.metadonnees_du_disque()["split_version"])

    resultat = pipeline.lancer_export()

    assert not resultat.agi


def test_un_export_perime_relance_le_script(chaine, tmp_path):
    """Sinon, le script d'export est appelé dans un processus neuf."""
    pipeline.executer(jusqua="decoupage")
    _poser_provenance(chaine, "000000000000")
    (tmp_path / "scripts" / "export_api_model.py").write_text(
        "print('export factice')", encoding="utf-8"
    )

    resultat = pipeline.lancer_export()

    assert resultat.agi


def test_l_entrainement_passe_par_son_script(chaine, tmp_path):
    """Jamais ignoré : un entraînement est une décision, pas une conséquence."""
    (tmp_path / "scripts" / "train.py").write_text(
        "print('entrainement factice')", encoding="utf-8"
    )

    resultat = pipeline.lancer_entrainement()

    assert resultat.agi
    assert "4 études" in resultat.detail


def test_un_script_en_echec_arrete_la_chaine(chaine, tmp_path):
    """Le code de sortie est repris dans le message : on sait quoi relire."""
    (tmp_path / "scripts" / "train.py").write_text(
        "raise SystemExit(3)", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="code 3"):
        pipeline.lancer_entrainement()

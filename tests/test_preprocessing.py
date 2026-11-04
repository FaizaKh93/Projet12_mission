"""Parité entre le pipeline d'entraînement et celui servi par l'API.

Ces tests répondent à une seule question : **l'artefact déployé prédit-il
exactement ce que prédirait le modèle entraîné ?** C'est le point où un projet
de ML se casse silencieusement — le modèle est bon, le service rend autre
chose, et rien ne le signale.

Deux niveaux, parce que l'un ne suffit pas :

- les colonnes reconstruites sont identiques à celles de l'entraînement, `NaN`
  et types compris ;
- les **prédictions** sont identiques de bout en bout. Des colonnes qui se
  ressemblent ne garantissent pas des prédictions égales : un booléen arrivé
  en `object` ou un flottant en `int` traverse l'encodeur sans erreur et change
  le résultat.

**Ce qui tourne sans `data/`.** Les règles de dérivation, la robustesse aux
champs absents et l'intégrité de l'artefact s'éprouvent sur l'**artefact
déployé**, qui embarque la table FAO ajustée à l'entraînement : aucune donnée
n'est nécessaire, et c'est la pièce effectivement livrée qui est testée. Seule
la parité exacte avec le dataset exige les partitions, puisqu'elle compare la
reconstruction aux colonnes que le dataset porte lui-même ; elle est réservée à
l'environnement qui les possède.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from itertools import combinations, product

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from agritech import training
from agritech.config import (
    API_MODEL_PATH,
    FEATURES_FAO,
    FEATURES_MODELE,
    FINAL_MODEL_PATH,
    MODALITES_CATEGORIELLES,
    TRAIN_PATH,
    VAL_PATH,
)
from agritech.config import (
    PANEL_COVARIATES,
    PANEL_FEATURES_MODELE,
    PANEL_FEATURES_PREVISION,
    PANEL_TARGET,
)
from agritech.preprocessing import (
    FEATURES_FACULTATIVES,
    FEATURES_OBLIGATOIRES,
    FEATURES_SAISIES,
    PANEL_SAISIES,
    DerivationPanel,
    DerivationVariables,
)

#: Assez de lignes pour couvrir les six cultures et toutes les modalités, assez
#: peu pour que la suite reste rapide.
TAILLE_ECHANTILLON = 5_000

#: Seuls les tests de **parité** exigent les partitions : eux seuls comparent la
#: reconstruction aux colonnes que le dataset porte lui-même. Les fabriquer
#: reviendrait à inventer la référence, puis à vérifier qu'on la retrouve.
exige_les_partitions = pytest.mark.skipif(
    not (TRAIN_PATH.exists() and VAL_PATH.exists()),
    reason="partitions absentes : parité réservée à l'environnement MLOps",
)


# --------------------------------------------------------------------------- #
# Fixtures autonomes — rien d'autre que ce que le dépôt contient
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def artefact_servi():
    """L'artefact réellement déployé, lu dans le dépôt.

    Il embarque la `DerivationVariables` ajustée à l'entraînement : sa table
    FAO est celle qui a servi, pas une reconstruction. Les tests de règles
    n'ont donc besoin d'aucune donnée — et ils éprouvent la pièce livrée
    plutôt qu'un artefact rebâti dans le test.
    """
    return joblib.load(API_MODEL_PATH)


@pytest.fixture(scope="module")
def derivation_servie(artefact_servi):
    """Le transformateur tel qu'il part dans l'image."""
    return artefact_servi.estimator.named_steps["derivation"]


@pytest.fixture(scope="module")
def saisies():
    """Une grille déterministe de saisies valides : ni tirage, ni données.

    Produit cartésien des quatre catégorielles — 432 lignes couvrant toutes les
    modalités en proportions égales. L'équilibre n'est pas décoratif : c'est lui
    qui rend l'omission d'un champ facultatif équivalente à sa marginalisation,
    propriété que mesure `test_categorielles_facultatives_absentes`.

    Quantitatives et booléens tournent par l'indice, à l'intérieur du domaine
    d'entraînement, pour couvrir leurs combinaisons sans multiplier les lignes.
    """
    pluies = (200.0, 450.0, 700.0, 950.0)
    temperatures = (18.0, 25.0, 33.0)
    return pd.DataFrame([
        {"Rainfall_mm": pluies[i % len(pluies)],
         "Temperature_Celsius": temperatures[i % len(temperatures)],
         "Fertilizer_Used": bool(i % 2),
         "Irrigation_Used": bool((i // 2) % 2),
         "Crop": culture, "Region": region,
         "Soil_Type": sol, "Weather_Condition": meteo}
        for i, (culture, region, sol, meteo) in enumerate(product(
            MODALITES_CATEGORIELLES["Crop"],
            MODALITES_CATEGORIELLES["Region"],
            MODALITES_CATEGORIELLES["Soil_Type"],
            MODALITES_CATEGORIELLES["Weather_Condition"],
        ))
    ])


# --------------------------------------------------------------------------- #
# Fixtures de parité — elles seules lisent `data/`
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def donnees():
    """Un échantillon de `train` et de `val`, partagé par tout le module.

    `train` sert à ajuster le transformateur, `val` à comparer ses sorties
    aux colonnes du dataset — jamais l'inverse.
    """
    train, val = training.charger_partitions(echantillon=50_000)
    return train, val.sample(TAILLE_ECHANTILLON, random_state=0)


@pytest.fixture(scope="module")
def derivation(donnees):
    """Le transformateur ajusté sur `train`, comme à l'entraînement.

    L'ajustement calcule la table FAO ; le refaire sur `val` fausserait la
    comparaison qu'on cherche justement à établir.
    """
    train, _ = donnees
    return DerivationVariables().fit(train)


@pytest.fixture(scope="module")
def artefact(derivation):
    """L'artefact tel qu'il sera déployé : gelé dans son intégralité.

    Le gel porte sur le Pipeline entier, pas sur son dernier étage. Geler le
    seul estimateur laisserait `fit()` réajuster les étapes situées devant.
    """
    modele = joblib.load(FINAL_MODEL_PATH)
    return FrozenEstimator(Pipeline([("derivation", derivation), ("modele", modele)]))


# --------------------------------------------------------------------------- #
# Parité
# --------------------------------------------------------------------------- #


@exige_les_partitions
def test_colonnes_reconstruites_identiques(donnees, derivation):
    """Les 13 colonnes dérivées des 8 saisies égalent celles du dataset."""
    _, val = donnees
    obtenu = derivation.transform(val[FEATURES_SAISIES]).reset_index(drop=True)
    attendu = val[FEATURES_MODELE].reset_index(drop=True)

    assert list(obtenu.columns) == list(attendu.columns)
    assert (obtenu.dtypes == attendu.dtypes).all()
    # `equals` compare NaN à NaN comme égaux, ce qu'on veut ici : l'absence de
    # référence FAO pour l'orge et le coton EST l'information à reproduire.
    assert obtenu.equals(attendu)


@exige_les_partitions
def test_predictions_identiques_de_bout_en_bout(donnees, artefact):
    """Même prédiction depuis les 13 colonnes et depuis les 8 saisies."""
    _, val = donnees
    modele = joblib.load(FINAL_MODEL_PATH)
    depuis_13 = modele.predict(val[FEATURES_MODELE])
    depuis_8 = artefact.predict(val[FEATURES_SAISIES])
    assert np.array_equal(depuis_13, depuis_8)


@exige_les_partitions
@pytest.mark.parametrize("culture", ["Barley", "Cotton", "Maize", "Rice",
                                     "Soybean", "Wheat"])
def test_parite_par_culture(donnees, artefact, culture):
    """La parité tient pour les six cultures, l'orge et le coton compris.

    Ces deux-là n'ont pas de référence FAO : leurs indicateurs valent `NaN` et
    passent par l'imputeur du Pipeline. C'est le chemin de code le plus fragile.
    """
    _, val = donnees
    lignes = val[val["Crop"] == culture]
    assert len(lignes) > 0, f"{culture} absente de l'échantillon"

    modele = joblib.load(FINAL_MODEL_PATH)
    assert np.array_equal(modele.predict(lignes[FEATURES_MODELE]),
                          artefact.predict(lignes[FEATURES_SAISIES]))


def test_nan_fao_preserve_pour_orge_et_coton(derivation_servie):
    """Les indicateurs absents restent `NaN` et ne deviennent pas zéro.

    Substituer zéro ne lèverait aucune erreur et fausserait les prédictions
    d'un tiers des cultures.
    """
    numeriques = [c for c in FEATURES_FAO if c != "crop_in_fao_reference"]
    saisie = pd.DataFrame([
        {"Rainfall_mm": 500.0, "Temperature_Celsius": 25.0, "Fertilizer_Used": True,
         "Irrigation_Used": False, "Crop": culture, "Region": "North",
         "Soil_Type": "Loam", "Weather_Condition": "Sunny"}
        for culture in ["Barley", "Cotton", "Wheat"]
    ])
    sortie = derivation_servie.transform(saisie)

    assert sortie.loc[[0, 1], numeriques].isna().all().all()
    assert not sortie.loc[[2], numeriques].isna().any().any()
    assert sortie["crop_in_fao_reference"].tolist() == [False, False, True]


# --------------------------------------------------------------------------- #
# Robustesse
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("absentes", [
    list(combo)
    for taille in range(len(FEATURES_FACULTATIVES) + 1)
    for combo in combinations(FEATURES_FACULTATIVES, taille)
])
def test_categorielles_facultatives_absentes(saisies, artefact_servi, absentes):
    """Omettre les champs facultatifs ne plante pas et ne biaise pas.

    Le seuil est volontairement large au regard de ce qui a été mesuré
    (0,0017 t/ha au pire) : il attrape une régression de comportement, pas une
    dérive au dernier chiffre.
    """
    reference = artefact_servi.predict(saisies)
    obtenu = artefact_servi.predict(saisies.drop(columns=absentes))

    ecart = np.abs(obtenu - reference)
    assert ecart.mean() < 0.01, f"dégradation inattendue sans {absentes}"
    assert abs((obtenu - reference).mean()) < 0.001, f"biais introduit sans {absentes}"


def test_culture_inconnue_rejetee(derivation_servie):
    """Une modalité invalide est une erreur, pas une information absente."""
    saisie = pd.DataFrame([{
        "Rainfall_mm": 500.0, "Temperature_Celsius": 25.0, "Fertilizer_Used": True,
        "Irrigation_Used": False, "Crop": "Quinoa", "Region": "North",
        "Soil_Type": "Loam", "Weather_Condition": "Sunny"}])
    with pytest.raises(ValueError, match="Culture inconnue"):
        derivation_servie.transform(saisie)


@pytest.mark.parametrize("manquante", FEATURES_OBLIGATOIRES)
def test_saisie_obligatoire_absente_rejetee(saisies, derivation_servie, manquante):
    """Une saisie obligatoire absente échoue explicitement."""
    with pytest.raises(ValueError, match="obligatoires absentes|Culture inconnue"):
        derivation_servie.transform(saisies.drop(columns=[manquante]))


def test_regle_du_seuil_hydrique(derivation_servie):
    """`low_rain_no_irrigation` vaut pluie sous le seuil ET absence d'irrigation."""
    cas = pd.DataFrame([
        {"Rainfall_mm": 300.0, "Irrigation_Used": False},   # attendu : True
        {"Rainfall_mm": 300.0, "Irrigation_Used": True},    # irriguée
        {"Rainfall_mm": 500.0, "Irrigation_Used": False},   # pluie suffisante
        {"Rainfall_mm": 400.0, "Irrigation_Used": False},   # seuil exclu
    ]).assign(Temperature_Celsius=25.0, Fertilizer_Used=True, Crop="Wheat",
              Region="North", Soil_Type="Loam", Weather_Condition="Sunny")

    assert derivation_servie.transform(cas)["low_rain_no_irrigation"].tolist() == \
        [True, False, False, False]


# --------------------------------------------------------------------------- #
# Intégrité de l'artefact
# --------------------------------------------------------------------------- #


def test_fit_ne_reentraine_pas_le_modele(saisies, artefact_servi):
    """Un `fit()` accidentel ne doit pas toucher aux coefficients.

    Un Pipeline scikit-learn ordinaire réajuste toutes ses étapes. Seul le gel
    de l'artefact entier garantit l'inverse.
    """
    interne = artefact_servi.estimator.named_steps["modele"].named_steps["modele"]
    avant = interne.coef_.copy()

    # La cible est sans importance : si le gel ne tenait pas, n'importe quel
    # `fit` suffirait à déplacer les coefficients.
    artefact_servi.fit(saisies, np.zeros(len(saisies)))

    assert np.array_equal(avant, interne.coef_)


def test_artefact_ne_reference_que_les_paquets_de_l_image(artefact_servi, tmp_path):
    """L'artefact sérialisé ne doit référencer aucun paquet absent de l'image.

    `pandas` 3 adosse ses chaînes à pyarrow dès que pyarrow est installé. Sans
    précaution, les deux axes de la table FAO deviennent des tableaux pyarrow
    et le pickle en porte la référence : le fichier se recharge sur le poste de
    développement et **échoue dans le conteneur**, où pyarrow n'est pas
    installé. La dépendance dépend alors de l'environnement d'export, pas du
    code — c'est ce qui la rend difficile à voir.

    **Portée de ce test.** C'est un indicateur peu coûteux, pas une preuve de
    portabilité : il cherche une signature textuelle dans les octets. Le
    contrôle décisif est de recharger l'artefact dans un environnement où le
    paquet est absent, puis de prédire — ce que fait le test de fumée du
    conteneur en intégration continue. Les deux sont complémentaires : celui-ci
    tourne en une seconde à chaque commit, l'autre à chaque image construite.
    """
    chemin = tmp_path / "artefact.joblib"
    joblib.dump(artefact_servi, chemin)

    # `pyarrow` est le cas rencontré ; les autres sont des paquets lourds de
    # l'entraînement, qu'un artefact de service n'a aucune raison de citer.
    #
    # Les frontières ne sont pas décoratives : sans elles, `shap` matcherait
    # `shape`, que joblib écrit dans l'en-tête de chaque tableau numpy, et le
    # test échouerait sur tous les artefacts du projet.
    motif = (rb"(?<![a-zA-Z_])(?:pyarrow|optuna|shap|lightgbm|mlflow)"
             rb"(?![a-zA-Z])[a-zA-Z_.]*")
    cites = sorted({m.decode() for m in re.findall(motif, chemin.read_bytes())})
    assert not cites, f"dépendances hors de l'image de l'API : {cites}"


def test_artefact_autonome_en_processus_neuf(artefact_servi, tmp_path):
    """Rechargé dans un interpréteur vierge, l'artefact prédit.

    Vérifie que la classe du transformateur est importable depuis le paquet
    installé et que rien ne dépend de l'état de la session courante.
    """
    chemin = tmp_path / "artefact.joblib"
    joblib.dump(artefact_servi, chemin)

    code = textwrap.dedent(f"""
        import joblib, pandas as pd
        artefact = joblib.load(r"{chemin}")
        saisie = pd.DataFrame([{{"Rainfall_mm": 500.0, "Temperature_Celsius": 25.0,
            "Fertilizer_Used": True, "Irrigation_Used": False, "Crop": "Wheat",
            "Region": "North", "Soil_Type": "Loam", "Weather_Condition": "Sunny"}}])
        print(float(artefact.predict(saisie)[0]))
    """)
    resultat = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True)

    assert resultat.returncode == 0, resultat.stderr
    assert 0 < float(resultat.stdout.strip()) < 15



# --------------------------------------------------------------------------- #
# Chaîne `real` : le transformateur du panel
#
# Éprouvé sur un **panel synthétique**, et non sur les données du projet. Ce
# qu'on vérifie ici est une règle — le décalage d'un cran, le refus d'une série
# trouée — et une règle se montre mieux sur quatre séries dont on connaît le
# contenu que sur 53 044 lignes. Ces tests tournent donc sans `data/`, comme
# ceux du transformateur de parcelle.
# --------------------------------------------------------------------------- #


def _panel(series: dict[tuple[str, str], dict[int, float]]) -> pd.DataFrame:
    """Un panel au format du dataset, retards compris.

    Les retards sont calculés comme la préparation les calcule : `lag1` est le
    rendement de l'année précédente **si elle existe**, sinon `NaN`. C'est ce
    qui permet aux tests de distinguer une série continue d'une série trouée
    sans écrire les valeurs à la main.
    """
    lignes = []
    for (aire, culture), rendements in series.items():
        for annee, rendement in sorted(rendements.items()):
            ligne = {"Area": aire, "Item": culture, "Year": annee,
                     PANEL_TARGET: rendement, "choc": 0.0}
            for rang, retard in enumerate(PANEL_FEATURES_PREVISION, start=1):
                ligne[retard] = rendements.get(annee - rang, float("nan"))
            for covariable in PANEL_COVARIATES:
                ligne[covariable] = 20.0
            lignes.append(ligne)
    return pd.DataFrame(lignes)


#: Quatre séries choisies pour couvrir les quatre cas qui comptent.
SERIES_ESSAI = {
    # Continue jusqu'à l'année de référence : servable.
    ("Atlantis", "Wheat"): {annee: 3.0 + annee / 1000
                            for annee in range(2010, 2017)},
    # S'interrompt en 2014 : son `lag1` ne serait pas le rendement de l'an
    # dernier, la série est écartée.
    ("Atlantis", "Maize"): {annee: 8.0 + annee / 1000
                            for annee in range(2010, 2015)},
    # Trop courte : en 2016, elle n'a pas trois ans d'historique derrière elle.
    ("Borduria", "Rice, paddy"): {2015: 4.0, 2016: 4.2},
    # Seule culture de son pays : il n'y a rien à classer.
    ("Syldavia", "Potatoes"): {annee: 20.0 + annee / 100
                               for annee in range(2008, 2017)},
}


@pytest.fixture(scope="module")
def derivation_panel():
    """Le transformateur ajusté sur le panel synthétique."""
    return DerivationPanel().fit(_panel(SERIES_ESSAI))


def test_panel_annee_de_reference_est_la_derniere_observee(derivation_panel):
    """L'année prédite est celle qui suit le dernier relevé, pas une autre."""
    assert derivation_panel.annee_reference_ == 2016
    assert derivation_panel.annee_predite_ == 2017


def test_panel_ne_sert_que_les_series_continues(derivation_panel):
    """Deux séries sur quatre sont servables, et ce sont les bonnes.

    `Atlantis/Maize` s'arrête en 2014 : servir son relevé de 2014 comme
    « l'an dernier » décalerait la prévision de deux ans sans le dire.
    `Borduria/Rice, paddy` n'a que deux années : son troisième retard manque.
    """
    assert derivation_panel.couples_servables_ == [("Atlantis", "Wheat"),
                                                   ("Syldavia", "Potatoes")]


def test_panel_decale_les_retards_d_un_cran(derivation_panel):
    """`lag1` est le relevé de l'année de référence, et non son propre `lag1`.

    C'est l'erreur que ce décalage évite : reprendre la ligne de 2016 telle
    quelle prédirait 2016, pas 2017. Les valeurs attendues se lisent
    directement dans la série d'essai.
    """
    serie = SERIES_ESSAI[("Atlantis", "Wheat")]
    assert derivation_panel.retards("Atlantis", "Wheat") == {
        "lag1": serie[2016], "lag2": serie[2015], "lag3": serie[2014]}


def test_panel_transform_rend_les_colonnes_du_modele(derivation_panel):
    """Les 8 colonnes attendues, dans l'ordre, avec les covariables à `NaN`.

    Les covariables portent sur l'année à prédire : les laisser vides fait
    jouer l'imputeur du Pipeline, qui y met la médiane de `train`. Y glisser
    une valeur reviendrait à inventer une température.
    """
    sortie = derivation_panel.transform(
        pd.DataFrame({"Area": ["Atlantis"], "Item": ["Wheat"]}))

    assert list(sortie.columns) == list(PANEL_FEATURES_MODELE)
    assert sortie[PANEL_COVARIATES].isna().all().all()
    assert sortie[PANEL_FEATURES_PREVISION].notna().all().all()
    assert sortie[list(PANEL_FEATURES_PREVISION)].dtypes.eq("float64").all()


def test_panel_get_feature_names_out_decrit_la_sortie(derivation_panel):
    """Le `ColumnTransformer` en aval route les colonnes par leur nom."""
    assert (list(derivation_panel.get_feature_names_out())
            == list(PANEL_FEATURES_MODELE))


@pytest.mark.parametrize("couple, motif", [
    (("Atlantis", "Quinoa"), "culture jamais cultivée là"),
    (("Narnia", "Wheat"), "pays absent du panel"),
    (("Atlantis", "Maize"), "série interrompue avant l'année de référence"),
])
def test_panel_refuse_un_couple_non_servable(derivation_panel, couple, motif):
    """Un couple hors table est refusé, pas servi avec une imputation.

    Le troisième cas est le plus instructif : la série existe dans le panel,
    mais pas jusqu'au bout. L'accepter ferait passer un relevé de 2014 pour
    celui de l'an dernier — une prédiction fausse sans aucun signal.
    """
    entree = pd.DataFrame([dict(zip(PANEL_SAISIES, couple))])
    with pytest.raises(ValueError, match="non servable"):
        derivation_panel.transform(entree)


def test_panel_refuse_une_saisie_absente(derivation_panel):
    """Sans la culture, il n'y a aucune série dont chercher l'historique."""
    with pytest.raises(ValueError, match="Saisies absentes"):
        derivation_panel.transform(pd.DataFrame({"Area": ["Atlantis"]}))


def test_panel_fit_exige_le_panel_complet():
    """Ajuster sur des données sans retards ne peut pas marcher à moitié."""
    with pytest.raises(ValueError, match="panel complet"):
        DerivationPanel().fit(pd.DataFrame({"Area": ["Atlantis"],
                                            "Item": ["Wheat"]}))


def test_panel_cultures_par_pays_est_le_vocabulaire_servi(derivation_panel):
    """C'est cette table que l'interface lit, d'où son ordre déterministe."""
    assert derivation_panel.cultures_par_pays() == {
        "Atlantis": ["Wheat"], "Syldavia": ["Potatoes"]}


def test_panel_table_ne_reference_pas_pyarrow(derivation_panel, tmp_path):
    """Sérialisée, la table ne doit citer aucun paquet absent de l'image.

    `MultiIndex.from_arrays` ré-infère les niveaux en chaînes adossées à
    pyarrow, **même nourri de tableaux `object`** : sans le `set_levels` du
    transformateur, l'artefact se recharge ici et échoue dans le conteneur,
    où le groupe `api` du lock n'installe pas pyarrow. Mesuré : `joblib.load`
    y lève alors `ImportError: pyarrow`.
    """
    chemin = tmp_path / "derivation_panel.joblib"
    joblib.dump(derivation_panel, chemin)

    motif = (rb"(?<![a-zA-Z_])(?:pyarrow|optuna|shap|lightgbm|mlflow)"
             rb"(?![a-zA-Z])[a-zA-Z_.]*")
    cites = sorted({m.decode() for m in re.findall(motif, chemin.read_bytes())})
    assert not cites, f"dépendances hors de l'image de l'API : {cites}"

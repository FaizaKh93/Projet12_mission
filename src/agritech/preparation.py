"""Construction du dataset consolidé, à partir des six fichiers sources.

**Ce que ce module résout.** La chaîne de préparation vivait dans
`notebooks/data_preparation.ipynb`, mêlée à l'analyse qui la justifie. Un
notebook ne se rejoue pas : il s'ouvre, se fait dérouler cellule après cellule,
et rien ne garantit l'ordre ni l'état des variables. Le dataset livré n'était
donc reproductible que par un humain attentif.

Les fonctions ci-dessous sont **pures** : elles prennent des tables, en rendent
d'autres, n'écrivent rien et n'affichent rien. Le notebook les appelle pour
commenter leurs sorties ; `scripts/prepare_data.py` les appelle pour produire le
livrable. Les deux chemins traversent le même code, donc ne peuvent pas diverger.

**Contrainte de reproductibilité.** Le fichier consolidé porte une empreinte,
`data_version`, que chaque run MLflow enregistre. Régénérer un dataset
différent — ne serait-ce que par l'ordre des colonnes — rendrait tous les runs
passés incomparables et périmerait la provenance de l'artefact déployé. L'ordre
des opérations de ce module reproduit donc exactement celui du notebook, et
`pipeline.verifier_preparation` le contrôle par comparaison d'empreintes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agritech.config import (
    CROP_MAPPING,
    CROP_YIELD_PATH,
    CROPS_WITHOUT_REFERENCE,
    HG_PER_HA_TO_T_PER_HA,
    PESTICIDES_PATH,
    RAINFALL_PATH,
    SEUIL_FAIBLE_PLUIE_MM,
    TARGET,
    TEMP_PATH,
    YIELD_DF_PATH,
    YIELD_RAW_PATH,
)

#: Clé du panel FAO : un rendement est identifié par pays, culture et année.
CLE_FAO = ["Area", "Item", "Year"]

#: Les cinq variables de référence calculées par culture, dans l'ordre où le
#: livrable les porte. Deux seront retirées plus bas.
COLONNES_REFERENCE = [
    "crop_productivity_index",
    "crop_yield_ref_t_ha",
    "crop_yield_geo_cv",
    "crop_yield_trend_decade",
    "crop_countries",
]

#: Retirées du livrable pour redondance, conservées dans `crop_reference.csv` :
#: l'indice intra-pays est mieux fondé que la médiane brute, et la variabilité
#: géographique plus informative que le simple nombre de pays.
COLONNES_REDONDANTES = ["crop_yield_ref_t_ha", "crop_countries"]

#: Retirées après analyse : `inputs_level` résume deux variables déjà présentes,
#: `Days_to_Harvest` n'est pas connue au moment où l'agriculteur décide.
COLONNES_NON_RETENUES = ["inputs_level", "Days_to_Harvest"]

#: Ce qui reste du référentiel FAO dans le dataset livré.
COLONNES_DIFFUSEES = [c for c in COLONNES_REFERENCE if c not in COLONNES_REDONDANTES]

#: Dictionnaire des variables du livrable : provenance, unité, rôle.
DICTIONNAIRE = [
    ("Region", "crop_yield.csv", "—", "Zone générique du jeu simulé, sans correspondance géographique", "oui"),
    ("Soil_Type", "crop_yield.csv", "—", "Type de sol", "oui"),
    ("Crop", "crop_yield.csv", "—", "Culture de la parcelle", "oui"),
    ("Rainfall_mm", "crop_yield.csv", "mm", "Pluviométrie de la parcelle, période non documentée", "oui"),
    ("Temperature_Celsius", "crop_yield.csv", "°C", "Température de la parcelle, période non documentée", "oui"),
    ("Fertilizer_Used", "crop_yield.csv", "booléen", "Fertilisation appliquée", "oui"),
    ("Irrigation_Used", "crop_yield.csv", "booléen", "Irrigation appliquée", "oui"),
    ("Weather_Condition", "crop_yield.csv", "—", "Condition météo dominante", "oui"),
    (TARGET, "crop_yield.csv", "t/ha", "CIBLE — rendement, non modifié", "cible"),
    ("crop_productivity_index", "yield.csv", "ratio", "Rendement relatif, normalisé par la médiane du pays", "oui"),
    ("crop_yield_geo_cv", "yield.csv", "ratio", "Dispersion ENTRE PAYS — variabilité géographique, pas risque parcellaire", "oui"),
    ("crop_yield_trend_decade", "yield.csv", "t/ha/décennie", "Pente historique, sensible au périmètre", "oui"),
    ("crop_in_fao_reference", "déduit", "booléen", "Culture présente dans le référentiel FAO", "oui"),
    ("low_rain_no_irrigation", "déduit", "booléen", "Pluie < 400 mm sans irrigation — seuil conventionnel", "oui"),
]


@dataclass(frozen=True)
class Sources:
    """Les six fichiers d'entrée, tels qu'ils sortent du disque."""

    parcelles: pd.DataFrame
    fao: pd.DataFrame
    pesticides: pd.DataFrame
    pluie: pd.DataFrame
    temperatures: pd.DataFrame
    fusion_fournie: pd.DataFrame


@dataclass(frozen=True)
class Preparation:
    """Tout ce que produit la chaîne, avant écriture."""

    dataset: pd.DataFrame
    panel: pd.DataFrame
    reference: pd.DataFrame
    dictionnaire: pd.DataFrame


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #


def charger_sources() -> Sources:
    """Lit les six CSV sans rien transformer.

    `yield_df.csv` est le fichier pré-assemblé fourni avec le jeu. Il n'entre
    pas dans la construction — il sert de contre-épreuve au panel reconstruit.
    """
    return Sources(
        parcelles=pd.read_csv(CROP_YIELD_PATH),
        fao=pd.read_csv(YIELD_RAW_PATH),
        pesticides=pd.read_csv(PESTICIDES_PATH),
        pluie=pd.read_csv(RAINFALL_PATH),
        temperatures=pd.read_csv(TEMP_PATH),
        fusion_fournie=pd.read_csv(YIELD_DF_PATH, index_col=0),
    )


def controle(df: pd.DataFrame, nom: str, cle: list[str] | None = None) -> dict:
    """Fiche de contrôle commune à tous les fichiers : volume, manquants, doublons."""
    ligne = {
        "fichier": nom,
        "lignes": len(df),
        "colonnes": df.shape[1],
        "cellules_nan": int(df.isna().sum().sum()),
        "colonnes_avec_nan": int((df.isna().sum() > 0).sum()),
        "doublons_stricts": int(df.duplicated().sum()),
    }
    ligne["doublons_sur_cle"] = int(df.duplicated(subset=cle).sum()) if cle else 0
    return ligne


def colonnes_constantes(df: pd.DataFrame) -> list[str]:
    """Colonnes à valeur unique : aucune information, donc supprimables."""
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


# --------------------------------------------------------------------------- #
# Nettoyage, fichier par fichier
# --------------------------------------------------------------------------- #


def nettoyer_parcelles(parcelles: pd.DataFrame) -> pd.DataFrame:
    """Retire les rendements négatifs, physiquement impossibles.

    Le filtre est écrit `~(< 0)` et non `>= 0` : les deux diffèrent dès qu'une
    valeur manque, puisqu'une comparaison avec `NaN` est toujours fausse. La
    première forme conserve les manquants, la seconde les supprimerait.
    """
    return parcelles[~(parcelles[TARGET] < 0)].reset_index(drop=True)


def agreger_temperatures(brut: pd.DataFrame) -> pd.DataFrame:
    """Une seule température par couple (pays, année).

    Le fichier en porte plusieurs par an. Sans agrégation, la jointure
    dupliquerait chaque rendement autant de fois qu'il y a de relevés. La
    médiane résiste aux valeurs aberrantes mieux que la moyenne.
    """
    temperatures = (
        brut.rename(columns={"year": "Year", "country": "Area"})
        .groupby(["Area", "Year"], as_index=False)["avg_temp"]
        .median()
    )
    return temperatures.dropna(subset=["avg_temp"]).reset_index(drop=True)


def nettoyer_pluie(brut: pd.DataFrame) -> pd.DataFrame:
    """Révèle les valeurs manquantes masquées en texte.

    La colonne est lue comme du texte parce qu'elle contient des chaînes non
    numériques. `to_numeric(errors='coerce')` les transforme en `NaN`, ce qui
    les rend comptables au lieu de les laisser passer pour des valeurs.
    """
    pluie = brut.copy()
    pluie.columns = [col.strip() for col in pluie.columns]  # ' Area' porte un espace
    pluie["rain_mm"] = pd.to_numeric(
        pluie["average_rain_fall_mm_per_year"], errors="coerce"
    )
    return (
        pluie.loc[pluie["rain_mm"].notna(), ["Area", "Year", "rain_mm"]]
        .reset_index(drop=True)
    )


def nettoyer_rendements(brut: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les colonnes porteuses d'information, et convertit l'unité."""
    rendements = brut[["Area", "Item", "Year", "Value"]].rename(
        columns={"Value": "yield_hg_ha"}
    )
    rendements["yield_t_ha"] = rendements["yield_hg_ha"] / HG_PER_HA_TO_T_PER_HA
    return rendements[rendements["yield_t_ha"] > 0].reset_index(drop=True)


def nettoyer_pesticides(brut: pd.DataFrame) -> pd.DataFrame:
    """Trois colonnes suffisent : le reste est constant."""
    return brut[["Area", "Year", "Value"]].rename(columns={"Value": "pesticides_t"})


# --------------------------------------------------------------------------- #
# Panel FAO et référentiel par culture
# --------------------------------------------------------------------------- #


def construire_panel(
    rendements: pd.DataFrame,
    temperatures: pd.DataFrame,
    pluie: pd.DataFrame,
    pesticides: pd.DataFrame,
) -> pd.DataFrame:
    """Joint les trois tables annexes aux rendements, sans dupliquer une ligne.

    `validate='many_to_one'` fait échouer la jointure si une table annexe
    portait deux lignes pour un même couple — une duplication silencieuse
    gonflerait le panel sans que rien ne le signale.
    """
    for nom, table in [
        ("temperatures", temperatures),
        ("pluie", pluie),
        ("pesticides", pesticides),
    ]:
        doublons = int(table.duplicated(subset=["Area", "Year"]).sum())
        if doublons:
            raise ValueError(f"{nom} : {doublons} doublons sur (Area, Year)")

    panel = (
        rendements.merge(temperatures, on=["Area", "Year"], how="left", validate="many_to_one")
        .merge(pluie, on=["Area", "Year"], how="left", validate="many_to_one")
        .merge(pesticides, on=["Area", "Year"], how="left", validate="many_to_one")
    )

    if len(panel) != len(rendements):
        raise ValueError("La jointure a modifié le nombre de lignes")
    return panel


def construire_reference(panel: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par culture : productivité, dispersion, tendance, climat.

    L'indice de productivité est **intra-pays** : le rendement d'une culture
    rapporté à la médiane de son pays, puis la médiane de ces ratios. Un
    classement mondial brut refléterait surtout la richesse agricole des pays
    où chaque culture est répandue, pas la culture elle-même.
    """
    cultures_fao = list(CROP_MAPPING.values())
    cultures = panel[panel["Item"].isin(cultures_fao)]

    mediane_par_pays = cultures.groupby("Area")["yield_t_ha"].transform("median")
    ratios = cultures.assign(ratio=cultures["yield_t_ha"] / mediane_par_pays)

    # Pays cultivant les quatre cultures : base de comparaison de la dispersion
    par_pays = cultures.pivot_table(
        index="Area", columns="Item", values="yield_t_ha", aggfunc="median"
    ).dropna()

    reference = pd.DataFrame(
        {
            "crop_productivity_index": ratios.groupby("Item")["ratio"].median(),
            "crop_yield_ref_t_ha": cultures.groupby("Item")["yield_t_ha"].median(),
            # Variabilité ENTRE PAYS : géographique, et non risque parcellaire
            "crop_yield_geo_cv": par_pays.std() / par_pays.mean(),
            "crop_countries": cultures.groupby("Item")["Area"].nunique(),
        }
    )

    # Tendance historique, en t/ha par décennie
    tendances = {}
    for culture, groupe in cultures.groupby("Item"):
        serie = groupe.groupby("Year")["yield_t_ha"].median()
        tendances[culture] = np.polyfit(serie.index, serie.to_numpy(), 1)[0] * 10
    reference["crop_yield_trend_decade"] = pd.Series(tendances)

    # Plages climatiques observées : où chaque culture est réellement cultivée
    climat = cultures.dropna(subset=["avg_temp"]).groupby("Item")["avg_temp"]
    reference["crop_temp_p10"] = climat.quantile(0.10)
    reference["crop_temp_p90"] = climat.quantile(0.90)
    pluviometrie = cultures.dropna(subset=["rain_mm"]).groupby("Item")["rain_mm"]
    reference["crop_rain_p10"] = pluviometrie.quantile(0.10)
    reference["crop_rain_p90"] = pluviometrie.quantile(0.90)

    # Contexte d'intensification, conservé pour contrôle — non diffusé
    reference["crop_pesticides_context_t"] = cultures.groupby("Item")["pesticides_t"].median()

    return reference


def traduire_reference(reference: pd.DataFrame) -> pd.DataFrame:
    """Passe du vocabulaire FAO à celui du jeu de parcelles.

    « Rice, paddy » devient « Rice ». Les cultures FAO sans équivalent sont
    écartées : elles n'ont aucune parcelle à enrichir.
    """
    diffusee = reference[COLONNES_REFERENCE].copy()
    diffusee.index = diffusee.index.map({v: k for k, v in CROP_MAPPING.items()})
    return diffusee.loc[diffusee.index.notna()]


# --------------------------------------------------------------------------- #
# Enrichissement des parcelles
# --------------------------------------------------------------------------- #


def enrichir(parcelles: pd.DataFrame, reference_diffusee: pd.DataFrame) -> pd.DataFrame:
    """Complète chaque parcelle par le référentiel de sa culture.

    Les valeurs manquantes sont **conservées** : l'orge et le coton n'ont pas
    de référence FAO, et fabriquer un chiffre pour une statistique inexistante
    fausserait les prédictions d'un tiers des cultures. `crop_in_fao_reference`
    rend cette absence lisible.
    """
    avant = len(parcelles)
    dataset = parcelles.merge(
        reference_diffusee, left_on="Crop", right_index=True, how="left"
    )
    if len(dataset) != avant:
        raise ValueError(
            f"L'enrichissement a dupliqué des lignes : {avant} -> {len(dataset)}"
        )

    dataset["crop_in_fao_reference"] = ~dataset["Crop"].isin(CROPS_WITHOUT_REFERENCE)

    # Niveau d'intensification : 0, 1 ou 2 intrants. Construit puis retiré plus
    # bas — il résume deux variables que le modèle garde séparées.
    dataset["inputs_level"] = dataset["Fertilizer_Used"].astype(int) + dataset[
        "Irrigation_Used"
    ].astype(int)

    # Indicateur DESCRIPTIF, pas un diagnostic de stress hydrique : celui-ci
    # dépendrait de la culture, du sol et de l'évapotranspiration.
    dataset["low_rain_no_irrigation"] = (
        dataset["Rainfall_mm"] < SEUIL_FAIBLE_PLUIE_MM
    ) & ~dataset["Irrigation_Used"]

    return dataset


def reduire_au_livrable(dataset: pd.DataFrame) -> pd.DataFrame:
    """Retire les colonnes que l'analyse a conduit à écarter.

    Séparé d'`enrichir` à dessein : le notebook justifie ces retraits par une
    étude de redondance qui a besoin des colonnes en question. Il analyse donc
    le dataset enrichi, puis appelle cette fonction — dans le même ordre que
    `preparer`, et c'est cet ordre qui fixe la disposition des colonnes
    restantes, donc les octets du livrable et `data_version`.
    """
    return dataset.drop(columns=COLONNES_REDONDANTES).drop(
        columns=COLONNES_NON_RETENUES
    )


def dictionnaire_des_variables() -> pd.DataFrame:
    """Le dictionnaire du livrable, indexé par nom de colonne."""
    return pd.DataFrame(
        DICTIONNAIRE,
        columns=["colonne", "provenance", "unité", "signification", "admissible_entraînement"],
    ).set_index("colonne")


# --------------------------------------------------------------------------- #
# Chaîne complète
# --------------------------------------------------------------------------- #


def preparer(sources: Sources | None = None) -> Preparation:
    """Enchaîne le tout : des six CSV au dataset consolidé.

    `sources` permet d'injecter des tables en mémoire — c'est ainsi que les
    tests éprouvent la chaîne sans toucher au disque.
    """
    sources = sources or charger_sources()

    parcelles = nettoyer_parcelles(sources.parcelles)
    panel = construire_panel(
        nettoyer_rendements(sources.fao),
        agreger_temperatures(sources.temperatures),
        nettoyer_pluie(sources.pluie),
        nettoyer_pesticides(sources.pesticides),
    )
    reference = construire_reference(panel)
    dataset = reduire_au_livrable(enrichir(parcelles, traduire_reference(reference)))

    return Preparation(
        dataset=dataset,
        panel=panel,
        reference=reference,
        dictionnaire=dictionnaire_des_variables(),
    )


def controler(preparation: Preparation) -> dict:
    """Vérifie le livrable, et dit pourquoi chaque chiffre est celui-là.

    Lève si une valeur manquante inattendue subsiste : les seules admises sont
    celles des cultures hors référentiel FAO, et leur nombre est calculable.
    """
    dataset = preparation.dataset
    hors_referentiel = int((~dataset["crop_in_fao_reference"]).sum())
    attendus = hors_referentiel * len(COLONNES_DIFFUSEES)
    constates = int(dataset.isna().sum().sum())

    if constates != attendus:
        raise ValueError(
            f"Valeurs manquantes inattendues : {constates} constatées, "
            f"{attendus} attendues ({hors_referentiel} lignes hors référentiel "
            f"× {len(COLONNES_DIFFUSEES)} colonnes)"
        )

    manquant_au_dictionnaire = set(dataset.columns) - set(preparation.dictionnaire.index)
    if manquant_au_dictionnaire:
        raise ValueError(f"Colonnes non documentées : {sorted(manquant_au_dictionnaire)}")

    return {
        "lignes": len(dataset),
        "colonnes": dataset.shape[1],
        "doublons_stricts": int(dataset.duplicated().sum()),
        "rendements_negatifs": int((dataset[TARGET] < 0).sum()),
        "cellules_manquantes": constates,
        "lignes_hors_referentiel": hors_referentiel,
    }

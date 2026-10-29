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
from agritech.preprocessing import (
    FEATURES_FACULTATIVES,
    FEATURES_OBLIGATOIRES,
    FEATURES_SAISIES,
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

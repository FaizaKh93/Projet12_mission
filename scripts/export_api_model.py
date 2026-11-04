"""Assemble, vérifie et publie l'artefact que l'API servira.

**Ce que ce script résout.** Le modèle retenu à l'étape 2 attend plus de
colonnes que l'appelant n'en connaît. Déployer tel quel échouerait sur
`columns are missing`. Ce script greffe le transformateur de dérivation devant
le modèle, gèle l'ensemble, et publie le résultat comme un artefact de service
à part entière.

**Deux chaînes, un seul script.** `--source simulated` sert les parcelles : 13
colonnes attendues, 8 saisies fournies, les 5 autres *calculées*.
`--source real` sert le panel FAO : 8 colonnes attendues, 2 saisies fournies —
le pays et la culture — les 3 retards étant *cherchés* dans l'historique
embarqué et les 3 covariables de l'année cible restant inconnues avant la
récolte. Ce qui distingue les deux chaînes est déclaré une fois dans
`CHAINES_DE_SERVICE` ; le reste du script lit le descripteur, exactement comme
l'entraînement lit `JEUX_DE_VARIABLES`.

**Ce qu'il ne réentraîne pas : rien.** Le modèle vient du registre, inchangé.
`FrozenEstimator` porte sur le Pipeline **entier**, pas sur son dernier étage :
geler le seul estimateur laisserait un `fit()` accidentel réajuster les étapes
situées devant lui.

**L'ordre des opérations est délibéré.** Toutes les vérifications de parité
passent *avant* la première écriture. Un artefact qui ne prédirait pas
exactement comme sa source ne doit pas atteindre le disque, et encore moins le
registre : c'est le point précis où un projet de ML se casse sans bruit.

**Ce qu'il ne fait pas : promouvoir.** La version est publiée sans alias. Le
`champion` du registre de service ne se pose qu'après les tests d'intégration
de l'API, et la commande pour le faire est affichée en fin d'exécution.
Publier n'est pas déclarer prêt à servir.

Exemples :

    # Vérifier sans rien écrire — ce que la CI exécute sur une pull request
    uv run python scripts/export_api_model.py --verifier-seulement

    # Export complet : .joblib déployable + version au registre
    uv run python scripts/export_api_model.py

    # La chaîne du panel FAO : (pays, culture) -> rendement national
    uv run python scripts/export_api_model.py --source real

    # Promotion : pose l'alias champion, sans rien réexporter
    uv run python scripts/export_api_model.py --source real --promouvoir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from agritech.config import (
    ALIAS_CHAMPION,
    API_METADATA_PATH,
    API_MODEL_PATH,
    API_PANEL_METADATA_PATH,
    API_PANEL_MODEL_PATH,
    FEATURES_BOOLEENNES,
    FEATURES_MODELE,
    FEATURES_NUMERIQUES,
    FINAL_MODEL_PATH,
    PANEL_ANNEE_PREDITE,
    PANEL_ANNEE_REFERENCE,
    PANEL_COVARIATES,
    PANEL_DATASET_PARQUET_PATH,
    PANEL_FEATURES_MODELE,
    PANEL_FEATURES_PREVISION,
    PANEL_FINAL_MODEL_PATH,
    PANEL_INTERVALLE_PLANCHER_95_T_HA,
    PANEL_INTERVALLE_RELATIF_95,
    PANEL_PREDICTION_MINIMALE_T_HA,
    PANEL_TARGET,
    PROJECT_ROOT,
    REGISTERED_MODEL_API,
    REGISTERED_MODEL_NAME,
    REGISTERED_MODEL_PANEL,
    REGISTERED_MODEL_PANEL_API,
    SOURCES,
    TARGET,
)
from agritech.preprocessing import (
    FEATURES_FACULTATIVES,
    FEATURES_OBLIGATOIRES,
    FEATURES_SAISIES,
    PANEL_SAISIES,
    DerivationPanel,
    DerivationVariables,
)
from agritech.training import (
    JEUX_DE_VARIABLES,
    SOURCE_PAR_DEFAUT,
    charger_partitions,
    configurer_mlflow,
    journaliser_jeux,
    versions_des_donnees,
)

#: Groupe de dépendances déclaré dans `pyproject.toml` dont l'artefact a besoin
#: pour être **désérialisé**. Distinct du groupe `api` : l'API charge un
#: `.joblib` (pickle standard) et n'a que faire de cloudpickle, là où l'artefact
#: du registre en dépend sans rien connaître de FastAPI.
GROUPE_DEPENDANCES = "artefact"


class ChaineDeService(NamedTuple):
    """Tout ce qui distingue l'export d'une chaîne de celui de l'autre.

    Même intention que `JeuDeVariables` du côté entraînement : les différences
    sont déclarées **une fois**, et les fonctions du script lisent le
    descripteur plutôt que de tester la source. Ce qui ne peut pas se réduire à
    une valeur — comment ajuster le transformateur, comment prouver la parité —
    est porté par les quatre appelables du bas, dont les implémentations sont
    propres à chaque chaîne.
    """

    #: Le mot de `--source`, celui qu'emploient déjà les autres scripts
    source: str
    #: Registre d'où vient le modèle retenu à l'étape 2
    registre_source: str
    #: Registre où la version **de service** est publiée
    registre_api: str
    #: Le `.joblib` du modèle retenu, dont on vérifie qu'il dit la même chose
    #: que le registre
    modele_final: Path
    #: L'artefact déployable, et sa provenance, que l'image embarque
    chemin_artefact: Path
    chemin_provenance: Path
    #: Les saisies du contrat d'entrée, et leur partage obligatoire/facultatif
    saisies: list[str]
    obligatoires: list[str]
    facultatives: list[str]
    #: Les colonnes que le modèle d'étude attend, et sa cible
    features_modele: list[str]
    cible: str
    #: Valeur du tag qui résume le contrat, reprise dans la provenance
    schema_entree: str
    #: Variable d'environnement qui fait servir cette chaîne depuis le
    #: registre plutôt que depuis le `.joblib` embarqué. Écrite ici pour que
    #: la commande affichée après une promotion soit la bonne ; `api/service`
    #: lit les mêmes noms, et un test confronte les deux.
    variable_uri: str
    #: Nom de la métrique MLflow qui chiffre ce que l'appelant ne fournit pas.
    #: Distinct d'une chaîne à l'autre parce qu'il ne mesure pas la même
    #: chose : l'omission de champs facultatifs d'un côté, l'imputation de
    #: covariables inconnues de l'autre.
    metrique_ecart: str
    #: Clés que la provenance porte **en plus** des clés communes, et que la
    #: promotion doit donc contrôler aussi. Vides sur les parcelles : leur
    #: artefact n'a rien de plus à déclarer que son modèle d'origine.
    cles_supplementaires: tuple[str, ...]
    #: Les données du fit du transformateur, et le témoin de parité
    charger: Callable
    #: Greffe le transformateur devant le modèle et gèle l'ensemble
    construire: Callable
    #: Toutes les vérifications, avant la moindre écriture
    verifier: Callable
    #: L'entrée de service sur laquelle les rechargements sont éprouvés
    entree: Callable
    #: Les trois lignes d'exemple attachées à la version du registre
    exemple: Callable
    #: Ce que l'artefact ajoute à sa provenance, lu dans son propre état
    metadonnees: Callable


def empreinte(chemin) -> str:
    """SHA-256 du fichier, en hexadécimal.

    Scelle le `.joblib` que l'image embarquera. C'est la seule mesure qui
    réponde, depuis le conteneur, à « sert-on bien l'artefact promu ? » — un
    fichier périmé figé dans une couche d'image ne se détecte pas autrement.

    **Elle ne compare pas l'artefact local à celui du registre.** Le premier
    sort de `joblib.dump`, le second de cloudpickle : deux formats, deux
    suites d'octets (7 197 contre 5 294 à la mesure), donc deux empreintes
    différentes par construction. C'est la parité **des prédictions** qui
    atteste qu'il s'agit du même modèle, et elle est vérifiée ailleurs.

    **Ce qu'elle n'atteste pas : un procédé.** Elle identifie un fichier.
    L'export, lui, s'est révélé reproductible à l'octet près — réexécuté pour
    produire la v2, il a rendu exactement l'empreinte de l'artefact précédent.
    En revanche un cycle `load`/`dump` ne reproduit pas les octets d'origine
    (7 197 puis 7 472 puis 7 488) : re-sérialiser un objet déjà désérialisé
    n'est pas le même geste que l'exporter depuis sa source.
    """
    return hashlib.sha256(chemin.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 1. Dépendances : une seule source de vérité, le fichier de verrouillage
# --------------------------------------------------------------------------- #


def dependances_verrouillees(groupe: str) -> list[str]:
    """Extrait d'`uv.lock` les dépendances épinglées d'un groupe.

    Pourquoi ne pas laisser MLflow deviner : il détecte le projet uv et capture
    **171 paquets** — jupyterlab, optuna et shap compris. Pourquoi ne pas écrire
    la liste ici : elle deviendrait une seconde source de vérité, qui dériverait
    du lock au premier `uv add`. Le lock tranche, et lui seul.
    """
    acheve = subprocess.run(
        ["uv", "export", "--no-default-groups", "--group", groupe,
         "--no-emit-project", "--no-hashes", "--format", "requirements.txt"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    if acheve.returncode != 0:
        raise RuntimeError(f"`uv export` a échoué :\n{acheve.stderr}")

    # On ne garde que les exigences : ni commentaires, ni lignes d'option
    # (`--index-url`, `-e .`), ni lignes vides.
    exigences = [ligne.strip() for ligne in acheve.stdout.splitlines()
                 if ligne.strip() and not ligne.lstrip().startswith(("#", "-"))]
    if not exigences:
        raise RuntimeError(f"`uv export --group {groupe}` n'a rien renvoyé.")
    return exigences


# --------------------------------------------------------------------------- #
# 2. Contrat d'entrée
# --------------------------------------------------------------------------- #


def construire_signature(chaine: ChaineDeService):
    """Déclare le contrat : les saisies exigées, puis les facultatives.

    MLflow n'impose que les premières ; une facultative absente traverse le
    contrôle de schéma sans erreur et la sentinelle du transformateur s'en
    charge. Une obligatoire absente, elle, est rejetée avant d'atteindre le
    modèle. Sur la chaîne du panel, tout est obligatoire : sans le pays ni la
    culture, il n'y a aucun historique à chercher.

    `infer_signature` ne saurait pas produire ceci : un exemple complet ne dit
    pas quelles colonnes étaient facultatives. D'où la construction explicite,
    mais **dérivée des constantes** — ajouter une variable au modèle ne peut
    donc pas laisser la signature en arrière.
    """
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, DataType, Schema

    def type_mlflow(colonne: str) -> DataType:
        """Le type MLflow d'une saisie, déduit de sa famille.

        Déduit plutôt qu'énuméré : une table recopiée à la main divergerait
        des constantes dès le premier ajout de variable.
        """
        if colonne in FEATURES_NUMERIQUES:
            return DataType.double
        if colonne in FEATURES_BOOLEENNES:
            return DataType.boolean
        return DataType.string

    colonnes = [ColSpec(type_mlflow(c), c) for c in chaine.obligatoires]
    colonnes += [ColSpec(type_mlflow(c), c, required=False)
                 for c in chaine.facultatives]

    return ModelSignature(
        inputs=Schema(colonnes),
        outputs=Schema([ColSpec(DataType.double, chaine.cible)]))


def exemple_parcelle(artefact, temoin: pd.DataFrame) -> pd.DataFrame:
    """Trois lignes réelles, pour que l'interface MLflow montre un appel valide.

    Les colonnes entières sont converties en flottantes : un entier ne porte pas
    de valeur manquante, et le contrôle de schéma refuserait un appel légitime.
    """
    exemple = temoin[FEATURES_SAISIES].head(3).copy()
    for colonne in FEATURES_NUMERIQUES:
        exemple[colonne] = exemple[colonne].astype("float64")
    return exemple


def exemple_panel(artefact, temoin: pd.DataFrame) -> pd.DataFrame:
    """Trois couples (pays, culture) **pris dans la table embarquée**.

    Pas dans `val` : ses lignes sont des années 2003-2009, et un exemple tiré
    de là laisserait croire que l'appelant choisit l'année. Il ne la choisit
    pas — le service en sert une seule, celle qui suit l'année de référence.
    """
    couples = _derivation(artefact).couples_servables_
    indices = [0, len(couples) // 2, len(couples) - 1]
    return pd.DataFrame([dict(zip(PANEL_SAISIES, couples[indice]))
                         for indice in indices])


def entree_parcelle(artefact, val: pd.DataFrame) -> pd.DataFrame:
    """Ce qu'un appelant envoie : les 8 saisies, sur toutes les lignes de val."""
    return val[FEATURES_SAISIES]


def entree_panel(artefact, val: pd.DataFrame) -> pd.DataFrame:
    """Tous les couples servables, c'est-à-dire tout ce que l'API acceptera.

    Pas `val[["Area", "Item"]]` : l'artefact servirait alors les retards de
    l'année de référence à des lignes de 2003-2009, et la comparaison ne
    voudrait rien dire. L'entrée de service **est** la liste des couples.
    """
    return pd.DataFrame(_derivation(artefact).couples_servables_,
                        columns=PANEL_SAISIES)


def metadonnees_parcelle(artefact) -> dict:
    """Rien à ajouter : la provenance du modèle d'origine dit tout."""
    return {}


def metadonnees_panel(artefact) -> dict:
    """Ce que l'artefact du panel doit déclarer de lui-même.

    L'année prédite est une **propriété de la table embarquée**, pas une
    option de l'API : sans elle dans la provenance, un conteneur ne pourrait
    pas dire sur quelle année porte ce qu'il sert. Le nombre de couples, lui,
    fait qu'une table remplacée ne passe pas inaperçue au contrôle de
    concordance.
    """
    derivation = _derivation(artefact)
    return {
        "annee_reference": derivation.annee_reference_,
        "annee_predite": derivation.annee_predite_,
        "couples_servables": len(derivation.couples_servables_),
    }


# --------------------------------------------------------------------------- #
# 3. Source : le modèle retenu à l'étape 2
# --------------------------------------------------------------------------- #


def resoudre_source(mlflow, chaine: ChaineDeService):
    """Récupère au registre la version que l'alias `champion` désigne.

    La source est le **registre**, pas le `.joblib` du modèle retenu : c'est la
    version taguée qui porte la traçabilité — run d'origine, empreintes de
    données, métriques. Partir du fichier laisserait une dérive silencieuse
    possible entre les deux ; on vérifie d'ailleurs qu'ils coïncident.
    """
    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(chaine.registre_source,
                                                    ALIAS_CHAMPION)
    except Exception as erreur:
        raise RuntimeError(
            f"Aucun alias « {ALIAS_CHAMPION} » sur {chaine.registre_source}. "
            f"Le choix du modèle se fait dans le notebook de comparaison de "
            f"cette chaîne, qui appelle promouvoir_champion("
            f"source={chaine.source!r}). ({erreur})"
        ) from erreur

    uri = f"models:/{chaine.registre_source}@{ALIAS_CHAMPION}"
    print(f"=== Source : le modèle retenu à l'étape 2 ({chaine.source}) ===")
    print(f"  {chaine.registre_source} version {version.version} "
          f"({version.tags.get('modele', '?')}) <- alias {ALIAS_CHAMPION}")
    print(f"  run d'origine : {version.run_id}")
    print(f"  données       : {version.tags.get('data_version')} / "
          f"split {version.tags.get('split_version')}")
    return mlflow.sklearn.load_model(uri), version


# --------------------------------------------------------------------------- #
# 4. Assemblage et vérifications
# --------------------------------------------------------------------------- #


def _derivation(artefact: FrozenEstimator):
    """Le transformateur, sous les deux enveloppes du gel et du Pipeline.

    L'API descend par le même chemin pour lire la table embarquée : c'est le
    seul endroit où l'ordre des étages est connu, et le nommer ici évite de le
    recopier ailleurs.
    """
    return artefact.estimator.named_steps["derivation"]


def charger_parcelle(echantillon: int | None):
    """Les données du fit du transformateur, et le témoin de parité."""
    return charger_partitions(echantillon=echantillon)


def charger_panel(echantillon: int | None):
    """Le panel **complet** pour la table de retards, `val` pour la parité.

    Le transformateur ne s'ajuste pas sur `train` ici, et ce n'est pas une
    entorse : les retards sont une entrée de service, et `train` s'arrête en
    2002. Ajuster la table sur lui ferait servir des retards vieux de quatorze
    ans. Le modèle, lui, reste celui du registre et n'a rien vu après 2002.

    `echantillon` ne tronque que le témoin de parité : une table de service
    construite sur un panel partiel n'aurait pas les mêmes couples, et son
    artefact ne doit de toute façon jamais être publié — d'où `--sample` qui
    implique `--verifier-seulement`.
    """
    panel = pd.read_parquet(PANEL_DATASET_PARQUET_PATH)
    (_, val) = charger_partitions(echantillon=echantillon, source="real")
    return panel, val


def construire_parcelle(source, ajustement: pd.DataFrame) -> FrozenEstimator:
    """Greffe le transformateur devant le modèle et gèle l'ensemble.

    Le gel porte sur le Pipeline entier. Geler le seul estimateur laisserait
    `fit()` réajuster la dérivation — la table FAO serait réapprise sur les
    données d'appel, et les prédictions changeraient sans qu'aucune erreur ne
    le signale.
    """
    derivation = DerivationVariables().fit(ajustement)
    print(f"  table FAO ajustée sur {len(ajustement):,} lignes, "
          f"{len(derivation.cultures_connues_)} cultures")
    return FrozenEstimator(Pipeline([("derivation", derivation),
                                     ("modele", source)]))


def construire_panel(source, ajustement: pd.DataFrame) -> FrozenEstimator:
    """Même greffe, avec la table des derniers retards connus.

    Le gel compte davantage encore ici : un `fit()` accidentel réajusterait la
    table sur les données d'appel, c'est-à-dire sur deux colonnes dépourvues
    d'historique, et le transformateur refuserait ensuite tous les couples.
    """
    derivation = DerivationPanel().fit(ajustement)
    print(f"  table de retards ajustée sur {len(ajustement):,} lignes : "
          f"{len(derivation.couples_servables_)} couples servables, "
          f"{len(derivation.cultures_par_pays())} pays")
    print(f"  année de référence {derivation.annee_reference_} -> "
          f"prévision {derivation.annee_predite_}")
    return FrozenEstimator(Pipeline([("derivation", derivation),
                                     ("modele", source)]))


def exiger_parite(libelle: str, obtenu: np.ndarray, reference: np.ndarray) -> None:
    """Interrompt tout si les prédictions ne sont pas **exactement** égales.

    L'égalité exacte, pas une tolérance : les deux chemins traversent le même
    modèle ajusté, donc le moindre écart signale une différence de
    prétraitement, pas une imprécision de calcul.
    """
    ecart = float(np.abs(obtenu - reference).max())
    if not np.array_equal(obtenu, reference):
        raise RuntimeError(
            f"PARITÉ ROMPUE — {libelle} : écart maximal {ecart:.3e}. "
            f"Rien n'a été écrit."
        )
    print(f"  {libelle:<44} écart max {ecart:.2e}")


def verifier_parcelle(chaine: ChaineDeService, artefact, source,
                      ajustement: pd.DataFrame, val: pd.DataFrame) -> float:
    """Toutes les vérifications, avant la moindre écriture.

    Renvoie l'écart moyen introduit par l'omission des trois champs
    facultatifs — la seule mesure qui ne soit pas censée valoir zéro, et qui
    est journalisée comme métrique.
    """
    reference = source.predict(val[FEATURES_MODELE])

    print("\n=== Intégrité de la source ===")
    exiger_parite(f"registre == models/{chaine.modele_final.name}",
                  joblib.load(chaine.modele_final).predict(val[FEATURES_MODELE]),
                  reference)

    print("\n=== Parité de l'artefact de service ===")
    print(f"  13 colonnes attendues -> {len(FEATURES_SAISIES)} saisies "
          f"({len(FEATURES_OBLIGATOIRES)} obligatoires, "
          f"{len(FEATURES_FACULTATIVES)} facultatives)")
    exiger_parite(f"parité sur {len(val):,} lignes de val",
                  artefact.predict(val[FEATURES_SAISIES]), reference)

    # Seule mesure attendue non nulle : la sentinelle met le bloc encodé à
    # zéro, la variable ne contribue plus. Mesuré à 0,0017 t/ha, soit 0,34 %
    # du RMSE — un ordre de grandeur à surveiller, pas une égalité à exiger.
    sans = artefact.predict(val[FEATURES_SAISIES].drop(columns=FEATURES_FACULTATIVES))
    ecart_sans = float(np.abs(sans - reference).mean())
    print(f"  {'sans les 3 champs facultatifs':<44} "
          f"écart moyen {ecart_sans:.4f} t/ha")

    # Un `fit()` accidentel ne doit pas toucher aux coefficients.
    interne = artefact.estimator.named_steps["modele"].named_steps["modele"]
    avant = interne.coef_.copy()
    artefact.fit(val[FEATURES_SAISIES], val[chaine.cible])
    if not np.array_equal(avant, interne.coef_):
        raise RuntimeError("Le gel ne protège pas le modèle. Rien n'a été écrit.")
    print(f"  {'fit() accidentel : coefficients inchangés':<44} OK")

    return ecart_sans


def verifier_panel(chaine: ChaineDeService, artefact, source,
                   panel: pd.DataFrame, val: pd.DataFrame) -> float:
    """Les mêmes exigences, sur la chaîne où l'historique est l'entrée.

    Renvoie l'écart moyen dû à l'imputation des trois covariables, mesuré sur
    `val` — l'équivalent ici de l'omission des champs facultatifs.

    **La parité ne peut pas se vérifier sur l'année prédite** : elle n'est pas
    observée, et aucune ligne du panel ne porte ses colonnes. On la vérifie
    donc un cran plus tôt. Une table ajustée sur le panel privé de sa dernière
    année doit reproduire, pour cette dernière année, exactement les retards
    que le dataset y porte — et donc exactement les prédictions du champion.
    C'est ce contrôle qui atteste du décalage d'un cran ; une table qui
    reprendrait les lignes telles quelles échouerait ici.
    """
    reference = source.predict(_sans_covariables(val[chaine.features_modele]))
    derivation = _derivation(artefact)

    print("\n=== Intégrité de la source ===")
    exiger_parite(f"registre == models/{chaine.modele_final.name}",
                  joblib.load(chaine.modele_final).predict(
                      _sans_covariables(val[chaine.features_modele])),
                  reference)

    if derivation.annee_reference_ != PANEL_ANNEE_REFERENCE:
        raise RuntimeError(
            f"Le panel s'arrête en {derivation.annee_reference_}, "
            f"`PANEL_ANNEE_REFERENCE` annonce {PANEL_ANNEE_REFERENCE}. Le "
            "contrat de l'API et la table servie décriraient deux années "
            "différentes : corriger `config.py` avant d'exporter."
        )

    print("\n=== Parité de l'artefact de service ===")
    print(f"  {len(chaine.features_modele)} colonnes attendues -> "
          f"{len(chaine.saisies)} saisies ({', '.join(chaine.saisies)}), "
          f"{len(PANEL_FEATURES_PREVISION)} retards cherchés, "
          f"{len(PANEL_COVARIATES)} covariables imputées")

    # Le décalage d'un cran, éprouvé contre le dataset lui-même.
    veille = DerivationPanel().fit(
        panel[panel["Year"] < derivation.annee_reference_])
    attendu = panel[panel["Year"] == derivation.annee_reference_].set_index(
        PANEL_SAISIES)
    couples = [couple for couple in veille.couples_servables_
               if couple in attendu.index]
    entree = pd.DataFrame(couples, columns=PANEL_SAISIES)
    reconstitue = veille.transform(entree)

    exiger_parite(
        f"retards reconstitués pour {derivation.annee_reference_}",
        reconstitue[PANEL_FEATURES_PREVISION].to_numpy(),
        attendu.loc[couples, PANEL_FEATURES_PREVISION].to_numpy())
    exiger_parite(
        f"prédictions sur {len(couples):,} couples de "
        f"{derivation.annee_reference_}",
        artefact.estimator.named_steps["modele"].predict(reconstitue),
        source.predict(_sans_covariables(
            attendu.loc[couples].reset_index()[chaine.features_modele])))

    # Seule mesure attendue non nulle : les trois covariables portent sur
    # l'année à prédire, inconnue avant la récolte. Mesuré à 0,049 t/ha sur
    # `val` — pour une contribution SHAP de 1,1 %.
    avec = source.predict(val[chaine.features_modele])
    ecart_imputation = float(np.abs(reference - avec).mean())
    print(f"  {'covariables de l année cible imputées':<44} "
          f"écart moyen {ecart_imputation:.4f} t/ha")

    _exiger_calibration(source, val, chaine)

    # Un couple absent de la table doit être refusé, non servi avec une
    # imputation : une médiane mondiale passée pour l'historique d'un pays
    # serait une prédiction sans fondement.
    try:
        artefact.predict(pd.DataFrame([{"Area": "Narnia", "Item": "Wheat"}]))
    except ValueError:
        print(f"  {'couple hors table : refusé':<44} OK")
    else:
        raise RuntimeError("Le transformateur sert un couple qu'il ignore.")

    # Un `fit()` accidentel ne doit toucher ni le modèle ni la table. On
    # compare les **prédictions** et non les coefficients : le champion de
    # cette chaîne pourrait demain être un modèle qui n'en a pas.
    avant = artefact.predict(entree.head(50))
    artefact.fit(entree.head(50), np.zeros(50))
    if not np.array_equal(avant, artefact.predict(entree.head(50))):
        raise RuntimeError("Le gel ne protège pas l'artefact. Rien n'a été écrit.")
    print(f"  {'fit() accidentel : prédictions inchangées':<44} OK")

    return ecart_imputation


def _sans_covariables(lignes: pd.DataFrame) -> pd.DataFrame:
    """Les colonnes du modèle, privées de ce que le service ne saura pas.

    Sert de témoin : c'est l'entrée que le champion recevra en production, et
    c'est donc à elle que la parité de l'artefact doit se mesurer.
    """
    prive = lignes.copy()
    prive[PANEL_COVARIATES] = np.nan
    return prive


def _exiger_calibration(source, val: pd.DataFrame,
                        chaine: ChaineDeService) -> None:
    """Les bornes de `config.py` décrivent-elles encore ce modèle ?

    Les quatre constantes de l'intervalle empirique ont été calibrées sur `val`
    pour un modèle donné. Un réentraînement les laisserait en place en silence,
    et l'API publierait une couverture qu'elle n'a plus. Ce contrôle les
    recalcule et refuse d'exporter si elles ont dérivé — avec les valeurs à
    inscrire, pour que la correction ne demande pas de deviner.
    """
    prediction = source.predict(_sans_covariables(val[chaine.features_modele]))
    residu = val[chaine.cible].to_numpy() - prediction

    base = np.clip(prediction, PANEL_PREDICTION_MINIMALE_T_HA, None)
    relatif = tuple(np.quantile(residu / base, [0.025, 0.975]))
    bas = prediction < 1.0
    plancher = tuple(np.quantile(residu[bas], [0.025, 0.975]))

    ecarts = {
        "PANEL_INTERVALLE_RELATIF_95": (
            relatif, PANEL_INTERVALLE_RELATIF_95,
            max(abs(obtenu - attendu) for obtenu, attendu
                in zip(relatif, PANEL_INTERVALLE_RELATIF_95)), 0.01),
        "PANEL_INTERVALLE_PLANCHER_95_T_HA": (
            plancher, PANEL_INTERVALLE_PLANCHER_95_T_HA,
            max(abs(obtenu - attendu) for obtenu, attendu
                in zip(plancher, PANEL_INTERVALLE_PLANCHER_95_T_HA)), 0.02),
    }
    derive = {nom: valeurs for nom, valeurs in ecarts.items()
              if valeurs[2] > valeurs[3]}
    if derive:
        details = "\n".join(
            f"  {nom} : config {attendu}, recalculé "
            f"({obtenu[0]:+.4f}, {obtenu[1]:+.4f}) — écart {ecart:.4f} "
            f"> {tolerance}"
            for nom, (obtenu, attendu, ecart, tolerance) in derive.items())
        raise RuntimeError(
            "CALIBRATION PÉRIMÉE — les bornes de l'intervalle ne décrivent "
            f"plus ce modèle.\n{details}\n"
            "  Mettre `config.py` à jour avec les valeurs recalculées, puis "
            "remesurer la couverture sur `test` avant d'exporter.")

    print(f"  {'bornes de l intervalle : conformes à config':<44} "
          f"relatif ({relatif[0]:+.3f}, {relatif[1]:+.3f}) | "
          f"plancher ({plancher[0]:+.3f}, {plancher[1]:+.3f})")


# --------------------------------------------------------------------------- #
# Les deux chaînes de service
#
# Déclarées **après** les fonctions qu'elles désignent : un descripteur qui
# porte des appelables ne peut pas précéder leur définition. Tout ce que le
# reste du script a besoin de savoir d'une chaîne est ici.
#
# Et déclarées par des **fabriques** plutôt que par une table figée : les
# chemins sont alors relus dans les constantes du module à chaque appel. Une
# table construite à l'import les capturerait une fois pour toutes, et un test
# qui détourne `API_MODEL_PATH` vers un dossier temporaire écrirait malgré
# tout dans `models/` — éprouvé, et c'est exactement ce qui s'est produit.
# --------------------------------------------------------------------------- #


def _chaine_parcelle() -> ChaineDeService:
    """La chaîne des parcelles simulées : 13 colonnes, 8 saisies."""
    return ChaineDeService(
        source="simulated",
        registre_source=REGISTERED_MODEL_NAME,
        registre_api=REGISTERED_MODEL_API,
        modele_final=FINAL_MODEL_PATH,
        chemin_artefact=API_MODEL_PATH,
        chemin_provenance=API_METADATA_PATH,
        saisies=FEATURES_SAISIES,
        obligatoires=FEATURES_OBLIGATOIRES,
        facultatives=FEATURES_FACULTATIVES,
        features_modele=FEATURES_MODELE,
        cible=TARGET,
        schema_entree=f"{len(FEATURES_SAISIES)}-saisies",
        variable_uri="AGRITECH_MODEL_URI",
        metrique_ecart="ecart_sans_facultatives",
        cles_supplementaires=(),
        charger=charger_parcelle,
        construire=construire_parcelle,
        verifier=verifier_parcelle,
        entree=entree_parcelle,
        exemple=exemple_parcelle,
        metadonnees=metadonnees_parcelle,
    )


def _chaine_panel() -> ChaineDeService:
    """La chaîne du panel FAO : 8 colonnes, 2 saisies, les retards cherchés."""
    return ChaineDeService(
        source="real",
        registre_source=REGISTERED_MODEL_PANEL,
        registre_api=REGISTERED_MODEL_PANEL_API,
        modele_final=PANEL_FINAL_MODEL_PATH,
        chemin_artefact=API_PANEL_MODEL_PATH,
        chemin_provenance=API_PANEL_METADATA_PATH,
        saisies=PANEL_SAISIES,
        # Rien n'est facultatif ici : sans le pays ni la culture, il n'y a
        # aucune série dont chercher l'historique.
        obligatoires=PANEL_SAISIES,
        facultatives=[],
        features_modele=PANEL_FEATURES_MODELE,
        cible=PANEL_TARGET,
        schema_entree=f"{len(PANEL_SAISIES)}-saisies",
        variable_uri="AGRITECH_PANEL_MODEL_URI",
        metrique_ecart="ecart_covariables_imputees",
        cles_supplementaires=("annee_reference", "annee_predite",
                              "couples_servables"),
        charger=charger_panel,
        construire=construire_panel,
        verifier=verifier_panel,
        entree=entree_panel,
        exemple=exemple_panel,
        metadonnees=metadonnees_panel,
    )


#: Une fabrique par chaîne, sous le mot que `--source` emploie.
FABRIQUES = {"simulated": _chaine_parcelle, "real": _chaine_panel}

# Les fabriques et le vocabulaire de `--source` doivent rester solidaires :
# une chaîne de données sans chaîne de service exportable serait une option de
# ligne de commande qui échoue à l'exécution.
assert set(FABRIQUES) == set(SOURCES), \
    "FABRIQUES et SOURCES décrivent des chaînes différentes"


def chaine_de_service(source: str = SOURCE_PAR_DEFAUT) -> ChaineDeService:
    """Le descripteur de la chaîne demandée, construit maintenant."""
    return FABRIQUES[source]()


def etiquettes_de_version(chaine: ChaineDeService, version_source,
                          supplements: dict, sceau: str) -> dict:
    """Ce que portent **à l'identique** la version du registre et le fichier.

    Une seule construction pour les deux, et c'est le correctif d'un défaut
    réel : les métadonnées propres à une chaîne — l'année prédite, la taille de
    la table — n'étaient écrites que dans le fichier. La promotion, qui compare
    les deux déclarations clé par clé, trouvait alors `2016` d'un côté et
    `None` de l'autre et refusait de poser l'alias. Elle avait raison : une
    version du registre qui ne porte pas ce qu'annonce l'artefact n'est pas
    traçable.
    """
    return {
        **versions_des_donnees(chaine.source),
        "phase": "export",
        "artefact": "service",
        # Le mot de `--source` et le jeu de variables de l'étude dont cet
        # artefact descend : sans eux, deux versions de service seraient
        # indiscernables au registre.
        "source": chaine.source,
        "jeu_de_variables": JEUX_DE_VARIABLES[chaine.source].artefact,
        "schema_entree": chaine.schema_entree,
        "modele": version_source.tags.get("modele", "inconnu"),
        # Traçabilité vers l'étape 2 : de quelle version du registre
        # d'entraînement cet artefact descend, et par quel run elle a été
        # produite.
        "registre_source": chaine.registre_source,
        "version_source": str(version_source.version),
        "run_source": version_source.run_id,
        # Versions effectives au moment de la sérialisation : un pickle ne se
        # recharge pas sous n'importe quelle version de scikit-learn.
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        # Empreinte du fichier que l'image embarquera. Enregistrée ici pour
        # que la promotion puisse refuser un `.joblib` remplacé depuis
        # l'export, et pour qu'un conteneur puisse vérifier ce qu'il sert.
        "artefact_sha256": sceau,
        # Et ce que cette chaîne ajoute, lu dans l'état de l'artefact.
        **supplements,
    }


# --------------------------------------------------------------------------- #
# 5. Publication
# --------------------------------------------------------------------------- #


def publier(mlflow, chaine: ChaineDeService, artefact, source, version_source,
            val: pd.DataFrame, ecart: float) -> int:
    """Écrit le `.joblib`, publie au registre, puis **revérifie les trois
    chemins de rechargement**.

    Trois chemins, parce qu'ils ne contrôlent pas la même chose :

    - le `.joblib` est ce que l'image Docker embarquera ;
    - `mlflow.sklearn.load_model` rend l'objet brut, **sans** appliquer la
      signature : il passerait même si celle-ci était fausse ;
    - `mlflow.pyfunc.load_model` l'applique. C'est le seul qui valide le
      contrat d'entrée, et c'est aussi ce que chargerait un serveur MLflow.
    """
    # Les rechargements se comparent à l'artefact **en mémoire**, dont la
    # parité avec le modèle du registre vient d'être établie : ce qui est
    # éprouvé ici est la sérialisation, pas le modèle.
    entree = chaine.entree(artefact, val)
    reference = artefact.predict(entree)
    exigences = dependances_verrouillees(GROUPE_DEPENDANCES)

    print(f"\n=== Dépendances de désérialisation (groupe « "
          f"{GROUPE_DEPENDANCES} » du lock) ===")
    print(f"  {len(exigences)} paquets épinglés : "
          f"{', '.join(exigences[:4])}...")

    print("\n=== Écriture ===")
    chaine.chemin_artefact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, chaine.chemin_artefact)
    taille_ko = chaine.chemin_artefact.stat().st_size / 1024
    sceau = empreinte(chaine.chemin_artefact)
    print(f"  {chaine.chemin_artefact.name:<28} {taille_ko:>7.1f} Ko  "
          f"sha256 {sceau[:12]}…")

    supplements = chaine.metadonnees(artefact)
    # Les clés produites et celles que la promotion contrôlera sont la même
    # liste, vue de deux endroits : les laisser diverger ferait sortir une
    # métadonnée du contrôle de concordance sans que rien ne le signale.
    if set(supplements) != set(chaine.cles_supplementaires):
        raise RuntimeError(
            f"Métadonnées produites {sorted(supplements)} ≠ clés déclarées "
            f"{sorted(chaine.cles_supplementaires)} dans le descripteur.")

    etiquettes = etiquettes_de_version(chaine, version_source, supplements,
                                       sceau)

    with mlflow.start_run(
            run_name=f"export-api-{chaine.source}-{etiquettes['modele']}"):
        mlflow.set_tags(etiquettes)
        mlflow.log_params({"version_source": version_source.version,
                           "n_val": len(val),
                           "n_saisies": len(chaine.saisies),
                           "n_colonnes_modele": len(chaine.features_modele)})
        mlflow.log_metrics({"parite_ecart_max": 0.0,
                            chaine.metrique_ecart: ecart,
                            "taille_artefact_ko": taille_ko})
        journaliser_jeux(mlflow, chaine.source, val=val)

        info = mlflow.sklearn.log_model(
            artefact,
            name=chaine.chemin_artefact.stem,
            # cloudpickle plutôt que skops : skops refuse `numpy.dtype` comme
            # type non fiable et ne saurait pas sérialiser le transformateur
            # maison greffé devant le modèle.
            serialization_format="cloudpickle",
            signature=construire_signature(chaine),
            input_example=chaine.exemple(artefact, val),
            pip_requirements=exigences,
        )
        version = mlflow.register_model(f"models:/{info.model_id}",
                                        chaine.registre_api, tags=etiquettes)

    print(f"  {chaine.registre_api} version {version.version} publiée "
          f"(sans alias)")

    # Provenance écrite à côté de l'artefact. Elle existe parce que l'image
    # n'embarque pas mlflow : sans ce fichier, le conteneur servirait un
    # modèle sans pouvoir dire lequel. `/model/info` le lit quand le modèle
    # vient du `.joblib` plutôt que du registre.
    provenance = {
        "registre": chaine.registre_api,
        "version": int(version.version),
        # Volontairement nul : l'alias se pose après les tests d'intégration.
        # Un artefact qui s'annoncerait champion avant validation mentirait.
        "alias": None,
        **{cle: etiquettes[cle] for cle in CLES_CONCORDANCE},
        **supplements,
        "saisies_obligatoires": chaine.obligatoires,
        "saisies_facultatives": chaine.facultatives,
        "exporte_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    chaine.chemin_provenance.write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"  {chaine.chemin_provenance.name:<28} provenance -> "
          f"{chaine.registre_source} v{version_source.version} "
          f"({etiquettes['modele']})")

    print("\n=== Rechargement : les trois chemins ===")
    exiger_parite(".joblib (ce que l'image embarque)",
                  joblib.load(chaine.chemin_artefact).predict(entree),
                  reference)

    uri = f"models:/{chaine.registre_api}/{version.version}"
    exiger_parite("registre, mlflow.sklearn (objet brut)",
                  mlflow.sklearn.load_model(uri).predict(entree),
                  reference)

    pyfunc = mlflow.pyfunc.load_model(uri)
    exiger_parite("registre, mlflow.pyfunc (signature appliquée)",
                  np.asarray(pyfunc.predict(entree)).ravel(),
                  reference)

    # Le contrat doit aussi **refuser** ce qu'il déclare obligatoire : une
    # signature qui laisse tout passer ne protège de rien. N'importe quelle
    # saisie obligatoire ferait l'épreuve ; la dernière déclarée, par
    # convention.
    exigee = chaine.obligatoires[-1]
    try:
        pyfunc.predict(entree.drop(columns=[exigee]))
    except Exception:
        print(f"  {'saisie obligatoire absente : rejetée':<44} OK ({exigee})")
    else:
        raise RuntimeError(
            f"La signature n'exige pas `{exigee}`. Contrat invalide.")

    return int(version.version)


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #

#: Clés que la provenance du disque et la version enregistrée doivent porter à
#: l'identique. Un écart signifie que le fichier embarqué ne décrit pas la
#: version que l'alias s'apprête à désigner — donc que l'image servirait un
#: modèle en annonçant la provenance d'un autre.
#:
#: Les versions de bibliothèques en font partie, et pas par excès de zèle :
#: `run_source` identifie le modèle *d'origine*, pas l'export. Deux exports du
#: même Ridge faits dans des environnements différents produiraient des
#: artefacts différents tout en partageant `run_source`. Ce sont ces trois
#: clés-là qui les distinguent.
#:
#: `artefact_sha256` y joue un rôle à part : les autres clés se comparent
#: entre deux déclarations, celle-ci se recalcule sur le fichier présent. Elle
#: est la seule à détecter un `.joblib` remplacé après l'export.
#:
#: Un test de la suite vérifie que cette liste couvre **toutes** les clés
#: descriptives du fichier de provenance : en ajouter une à l'export sans
#: l'inscrire ici affaiblirait le contrôle en silence.
CLES_CONCORDANCE = ("modele", "registre_source", "version_source",
                    "run_source", "schema_entree", "data_version",
                    "split_version", "hypotheses_economiques",
                    "sklearn_version", "pandas_version", "numpy_version",
                    "artefact_sha256")


def _ecrire_provenance(provenance: dict,
                       chemin: Path = API_METADATA_PATH) -> None:
    """Écrit le fichier de provenance par remplacement, non par troncature.

    `write_text` ouvre en écrasant : une interruption en cours d'écriture
    laisserait un JSON tronqué, que l'API refuserait ensuite de charger. Le
    fichier temporaire est créé dans le même dossier — `os.replace` peut
    échouer entre deux systèmes de fichiers.

    La doc Python attribue l'atomicité du remplacement à une **exigence
    POSIX** et n'en donne aucune sous Windows. Ce détour supprime donc le cas
    du fichier à moitié écrit ; il ne rend pas l'opération transactionnelle.
    """
    temporaire = chemin.with_suffix(".json.tmp")
    temporaire.write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(temporaire, chemin)


def promouvoir(mlflow, chaine: ChaineDeService | None = None) -> None:
    """Pose l'alias champion et met la provenance à jour.

    **Ce n'est pas une transaction.** Le registre MLflow et le fichier du
    disque sont deux systèmes, écrits l'un après l'autre : si la seconde
    écriture échoue, l'alias est posé alors que le fichier annonce encore
    `null`. Le script détecte cet état, le nomme, et affiche la reprise — il
    ne prétend pas l'empêcher.

    La propriété sur laquelle on peut compter n'est pas l'atomicité mais
    l'**idempotence** : relancer la commande repose le même alias sans effet,
    réécrit le fichier, et résorbe donc l'écart **sans créer de version**.
    Éprouvé en registre jetable, y compris après un échec d'écriture provoqué.

    Trois refus, tous avant la moindre écriture :

    - la provenance est antérieure au contrat de traçabilité courant ;
    - elle ne décrit pas la version enregistrée — désigner une version dont on
      n'a pas le binaire promouvrait autre chose que ce qui a été testé ;
    - le `.joblib` présent n'est pas celui dont l'empreinte a été enregistrée,
      c'est-à-dire qu'il a été remplacé depuis l'export.

    Rien n'est réexporté ici. L'option ne fait que désigner.
    """
    from mlflow.tracking import MlflowClient

    chaine = chaine or chaine_de_service()
    # Les clés communes, plus celles que cette chaîne ajoute : sur le panel,
    # l'année prédite et la taille de la table embarquée en font partie.
    cles = (*CLES_CONCORDANCE, *chaine.cles_supplementaires)

    if not chaine.chemin_provenance.exists():
        raise FileNotFoundError(
            f"Provenance absente : {chaine.chemin_provenance}. Exécuter "
            f"l'export de la chaîne {chaine.source!r} avant toute promotion.")

    provenance = json.loads(
        chaine.chemin_provenance.read_text(encoding="utf-8"))

    # Une provenance écrite par une version antérieure du script n'a pas
    # toutes les clés contrôlées. Le dire vaut mieux qu'un KeyError nu.
    manquantes = [cle for cle in cles if cle not in provenance]
    if manquantes:
        raise RuntimeError(
            f"{chaine.chemin_provenance.name} est antérieur à ces clés de "
            f"traçabilité : {', '.join(manquantes)}.\n"
            "  Réexécuter l'export pour produire une provenance complète.")

    registre, version = provenance["registre"], int(provenance["version"])
    client = MlflowClient()
    enregistree = client.get_model_version(registre, str(version))

    ecarts = {cle: (provenance[cle], enregistree.tags.get(cle))
              for cle in cles
              if str(provenance[cle]) != str(enregistree.tags.get(cle))}
    if ecarts:
        raise RuntimeError(
            f"La provenance du disque ne décrit pas {registre} v{version}.\n"
            f"  écarts (disque, registre) : {ecarts}\n"
            "  Réexécuter l'export avant de promouvoir.")
    print(f"\n=== Concordance ===")
    print(f"  {len(cles)} clés identiques entre "
          f"{chaine.chemin_provenance.name} et {registre} v{version}")

    # L'empreinte est recalculée sur le fichier **présent**. Comparer la
    # provenance au registre ne dirait rien d'un `.joblib` remplacé depuis
    # l'export : les deux porteraient la même valeur périmée.
    if not chaine.chemin_artefact.exists():
        raise FileNotFoundError(
            f"Artefact absent : {chaine.chemin_artefact}. Promouvoir une "
            "version dont le binaire manque laisserait l'image sans modèle "
            "à servir.")
    constatee = empreinte(chaine.chemin_artefact)
    if constatee != provenance["artefact_sha256"]:
        raise RuntimeError(
            f"{chaine.chemin_artefact.name} n'est pas le fichier exporté.\n"
            f"  enregistrée : {provenance['artefact_sha256']}\n"
            f"  constatée   : {constatee}\n"
            "  Réexécuter l'export avant de promouvoir.")
    print(f"  {chaine.chemin_artefact.name} scellé  sha256 {constatee[:12]}…")

    print(f"\n=== Promotion ===")
    try:
        precedente = int(client.get_model_version_by_alias(
            registre, ALIAS_CHAMPION).version)
        print(f"  « {ALIAS_CHAMPION} » désignait la version {precedente}")
    except Exception:
        precedente = None
        print(f"  « {ALIAS_CHAMPION} » n'était posé sur aucune version")

    client.set_registered_model_alias(registre, ALIAS_CHAMPION, str(version))

    provenance["alias"] = ALIAS_CHAMPION
    # Conservé pour le retour arrière : le registre garde l'historique des
    # versions, pas celui des déplacements d'alias. Sans cette trace, revenir
    # à la version précédemment championne supposerait de la retrouver à la
    # main.
    provenance["alias_precedent"] = precedente
    provenance["promu_le"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")

    try:
        _ecrire_provenance(provenance, chaine.chemin_provenance)
    except OSError as erreur:
        # L'alias est posé, le fichier ne l'est pas : c'est précisément la
        # divergence que cette fonction ne peut pas empêcher. On la nomme.
        raise RuntimeError(
            f"DIVERGENCE — « {ALIAS_CHAMPION} » désigne {registre} v{version} "
            f"au registre, mais {chaine.chemin_provenance.name} n'a pas "
            f"pu être écrit ({erreur}).\n"
            "  L'artefact du disque annonce donc un statut faux.\n"
            "  Reprise : corriger la cause, puis relancer la même commande. "
            "Elle est idempotente et résorbe l'écart sans créer de version.\n"
            + (f"  Retour arrière : l'alias désignait v{precedente}.\n"
               if precedente is not None else "")
        ) from erreur

    # Les deux écritures ont rendu la main : on relit plutôt que de supposer.
    relue = json.loads(
        chaine.chemin_provenance.read_text(encoding="utf-8"))
    pose = int(client.get_model_version_by_alias(registre,
                                                 ALIAS_CHAMPION).version)
    if relue["alias"] != ALIAS_CHAMPION or pose != version:
        raise RuntimeError(
            f"Vérification finale : le registre désigne v{pose}, le fichier "
            f"porte alias={relue['alias']!r}. Relancer la commande.")

    print(f"  {registre} v{version}  <-  « {ALIAS_CHAMPION} »")
    print(f"  {chaine.chemin_provenance.name:<28} alias = {ALIAS_CHAMPION!r}")
    print(f"  {'relecture des deux côtés':<28} concordent")
    print("\n  L'API peut désormais servir cette chaîne depuis le registre :")
    print(f"    $env:{chaine.variable_uri} = "
          f"\"models:/{registre}@{ALIAS_CHAMPION}\"")
    print("    uv run uvicorn api.main:app")


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def analyser_arguments():
    """Les options de la ligne de commande.

    `--sample` implique `--verifier-seulement` : les tables embarquées — la
    correspondance FAO d'un côté, les derniers retards de l'autre — doivent
    être ajustées sur l'intégralité de leurs données. Un artefact construit sur
    un échantillon ne doit donc jamais être publié.

    **Une seule chaîne par exécution**, là où `train.py` accepte les deux : un
    export écrit un artefact nommé et publie une version dans un registre
    précis, deux gestes qui ne se cumulent pas.
    """
    parseur = argparse.ArgumentParser(
        description="Assemble et publie l'artefact de service de l'API.")
    parseur.add_argument(
        "--source", choices=list(SOURCES), default=SOURCE_PAR_DEFAUT,
        help=f"chaîne à exporter (défaut : {SOURCE_PAR_DEFAUT}, les 8 saisies "
             "de parcelle). `real` exporte le service national du panel FAO : "
             "le pays et la culture, les retards cherchés dans la table "
             "embarquée. Le même mot qu'attendent les autres scripts")
    parseur.add_argument(
        "--verifier-seulement", action="store_true",
        help="exécute toutes les vérifications sans rien écrire")
    parseur.add_argument(
        "--promouvoir", action="store_true",
        help="pose l'alias champion sur la version décrite par la provenance "
             "et met ce fichier à jour — sans rien réexporter")
    parseur.add_argument(
        "--sample", type=int, default=None,
        help="sous-échantillonne le témoin de parité — implique "
             "--verifier-seulement, les tables embarquées devant être "
             "ajustées sur l'intégralité de leurs données")
    return parseur.parse_args()


def main():
    """Vérifie, puis publie — jamais l'inverse.

    Rien n'est écrit tant que la parité avec le modèle d'origine n'est pas
    établie, et la publication ne déplace pas l'alias `champion` : la
    promotion reste un geste explicite, postérieur aux tests de l'API.
    """
    arguments = analyser_arguments()
    chaine = chaine_de_service(arguments.source)

    # La promotion est un geste à part : elle ne réexporte rien, elle désigne.
    # La combiner à un export ferait promouvoir une version dans le même souffle
    # qu'on la publie, c'est-à-dire avant de l'avoir testée.
    if arguments.promouvoir:
        if arguments.verifier_seulement or arguments.sample:
            raise SystemExit("--promouvoir ne se combine ni avec "
                             "--verifier-seulement ni avec --sample.")
        promouvoir(configurer_mlflow(chaine.source), chaine)
        return

    verifier_seulement = arguments.verifier_seulement or bool(arguments.sample)
    if arguments.sample and not arguments.verifier_seulement:
        print("--sample implique --verifier-seulement : rien ne sera publié.\n")

    mlflow = configurer_mlflow(chaine.source)
    source, version_source = resoudre_source(mlflow, chaine)

    ajustement, val = chaine.charger(arguments.sample)
    print(f"\n=== Assemblage ({chaine.source}) ===")
    artefact = chaine.construire(source, ajustement)

    ecart = chaine.verifier(chaine, artefact, source, ajustement, val)

    if verifier_seulement:
        print("\nToutes les vérifications passent. Rien n'a été écrit "
              "(--verifier-seulement).")
        return

    version = publier(mlflow, chaine, artefact, source, version_source, val,
                      ecart)

    print("\n=== Publié, pas promu ===")
    print(f"  {chaine.registre_api} version {version} n'a aucun alias.")
    print(f"  L'alias « {ALIAS_CHAMPION} » se pose après les tests "
          f"d'intégration de l'API :")
    print(f"    uv run python scripts/export_api_model.py "
          f"--source {chaine.source} --promouvoir")
    # Une commande de ce script, et non `mlflow` en direct : poser l'alias à la
    # main laisserait `alias` à `null` dans le fichier de provenance, et
    # l'artefact embarqué mentirait sur son propre statut. Les deux doivent
    # bouger ensemble.


if __name__ == "__main__":
    main()

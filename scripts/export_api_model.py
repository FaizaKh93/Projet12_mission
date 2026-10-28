"""Assemble, vérifie et publie l'artefact que l'API servira.

**Ce que ce script résout.** Le modèle retenu à l'étape 2 attend les 13
colonnes du dataset consolidé ; l'agriculteur n'en saisira que 8. Déployer tel
quel échouerait sur `columns are missing`. Ce script greffe le transformateur
de dérivation devant le modèle, gèle l'ensemble, et publie le résultat comme un
artefact de service à part entière.

**Ce qu'il ne réentraîne pas : rien.** Le modèle vient du registre, inchangé.
`FrozenEstimator` porte sur le Pipeline **entier**, pas sur son dernier étage :
geler le seul estimateur laisserait un `fit()` accidentel réajuster les étapes
situées devant lui.

**L'ordre des opérations est délibéré.** Toutes les vérifications de parité
passent *avant* la première écriture. Un artefact qui ne prédirait pas
exactement comme sa source ne doit pas atteindre le disque, et encore moins le
registre : c'est le point précis où un projet de ML se casse sans bruit.

**Ce qu'il ne fait pas : promouvoir.** La version est publiée sans alias. Le
`champion` de `agritech-rendement-api` ne se pose qu'après les tests
d'intégration de l'API, et la commande pour le faire est affichée en fin
d'exécution. Publier n'est pas déclarer prêt à servir.

Exemples :

    # Vérifier sans rien écrire — ce que la CI exécute sur une pull request
    uv run python scripts/export_api_model.py --verifier-seulement

    # Export complet : .joblib déployable + version au registre
    uv run python scripts/export_api_model.py

    # Promotion : pose l'alias champion, sans rien réexporter
    uv run python scripts/export_api_model.py --promouvoir
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

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
    FEATURES_BOOLEENNES,
    FEATURES_MODELE,
    FEATURES_NUMERIQUES,
    FINAL_MODEL_PATH,
    PROJECT_ROOT,
    REGISTERED_MODEL_API,
    REGISTERED_MODEL_NAME,
    TARGET,
)
from agritech.preprocessing import (
    FEATURES_FACULTATIVES,
    FEATURES_OBLIGATOIRES,
    FEATURES_SAISIES,
    DerivationVariables,
)
from agritech.training import (
    JEU_PAR_DEFAUT,
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


def construire_signature():
    """Déclare le contrat : 5 saisies exigées, 3 facultatives.

    MLflow n'impose que les premières ; une facultative absente traverse le
    contrôle de schéma sans erreur et la sentinelle du transformateur s'en
    charge. Une obligatoire absente, elle, est rejetée avant d'atteindre le
    modèle.

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

    colonnes = [ColSpec(type_mlflow(c), c) for c in FEATURES_OBLIGATOIRES]
    colonnes += [ColSpec(type_mlflow(c), c, required=False)
                 for c in FEATURES_FACULTATIVES]

    return ModelSignature(inputs=Schema(colonnes),
                          outputs=Schema([ColSpec(DataType.double, TARGET)]))


def exemple_entree(val: pd.DataFrame) -> pd.DataFrame:
    """Trois lignes réelles, pour que l'interface MLflow montre un appel valide.

    Les colonnes entières sont converties en flottantes : un entier ne porte pas
    de valeur manquante, et le contrôle de schéma refuserait un appel légitime.
    """
    exemple = val[FEATURES_SAISIES].head(3).copy()
    for colonne in FEATURES_NUMERIQUES:
        exemple[colonne] = exemple[colonne].astype("float64")
    return exemple


# --------------------------------------------------------------------------- #
# 3. Source : le modèle retenu à l'étape 2
# --------------------------------------------------------------------------- #


def resoudre_source(mlflow):
    """Récupère au registre la version que l'alias `champion` désigne.

    La source est le **registre**, pas `models/modele_final.joblib` : c'est la
    version taguée qui porte la traçabilité — run d'origine, empreintes de
    données, métriques. Partir du fichier laisserait une dérive silencieuse
    possible entre les deux ; on vérifie d'ailleurs qu'ils coïncident.
    """
    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME,
                                                    ALIAS_CHAMPION)
    except Exception as erreur:
        raise RuntimeError(
            f"Aucun alias « {ALIAS_CHAMPION} » sur {REGISTERED_MODEL_NAME}. "
            f"Le choix du modèle se fait dans notebooks/model_experimentation."
            f"ipynb, qui appelle promouvoir_champion(). ({erreur})"
        ) from erreur

    uri = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS_CHAMPION}"
    print(f"=== Source : le modèle retenu à l'étape 2 ===")
    print(f"  {REGISTERED_MODEL_NAME} version {version.version} "
          f"({version.tags.get('modele', '?')}) <- alias {ALIAS_CHAMPION}")
    print(f"  run d'origine : {version.run_id}")
    print(f"  données       : {version.tags.get('data_version')} / "
          f"split {version.tags.get('split_version')}")
    return mlflow.sklearn.load_model(uri), version


# --------------------------------------------------------------------------- #
# 4. Assemblage et vérifications
# --------------------------------------------------------------------------- #


def construire_artefact(source, train: pd.DataFrame) -> FrozenEstimator:
    """Greffe le transformateur devant le modèle et gèle l'ensemble.

    Le gel porte sur le Pipeline entier. Geler le seul estimateur laisserait
    `fit()` réajuster la dérivation — la table FAO serait réapprise sur les
    données d'appel, et les prédictions changeraient sans qu'aucune erreur ne
    le signale.
    """
    derivation = DerivationVariables().fit(train)
    print(f"  table FAO ajustée sur {len(train):,} lignes, "
          f"{len(derivation.cultures_connues_)} cultures")
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


def verifier_avant_ecriture(artefact, source, val: pd.DataFrame) -> float:
    """Toutes les vérifications, avant la moindre écriture.

    Renvoie l'écart moyen introduit par l'omission des trois champs
    facultatifs — la seule mesure qui ne soit pas censée valoir zéro, et qui
    est journalisée comme métrique.
    """
    reference = source.predict(val[FEATURES_MODELE])

    print("\n=== Intégrité de la source ===")
    exiger_parite("registre == models/modele_final.joblib",
                  joblib.load(FINAL_MODEL_PATH).predict(val[FEATURES_MODELE]),
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
    artefact.fit(val[FEATURES_SAISIES], val[TARGET])
    if not np.array_equal(avant, interne.coef_):
        raise RuntimeError("Le gel ne protège pas le modèle. Rien n'a été écrit.")
    print(f"  {'fit() accidentel : coefficients inchangés':<44} OK")

    return ecart_sans


# --------------------------------------------------------------------------- #
# 5. Publication
# --------------------------------------------------------------------------- #


def publier(mlflow, artefact, source, version_source, val: pd.DataFrame,
            ecart_sans: float) -> int:
    """Écrit le `.joblib`, publie au registre, puis **revérifie les trois
    chemins de rechargement**.

    Trois chemins, parce qu'ils ne contrôlent pas la même chose :

    - le `.joblib` est ce que l'image Docker embarquera ;
    - `mlflow.sklearn.load_model` rend l'objet brut, **sans** appliquer la
      signature : il passerait même si celle-ci était fausse ;
    - `mlflow.pyfunc.load_model` l'applique. C'est le seul qui valide le
      contrat d'entrée, et c'est aussi ce que chargerait un serveur MLflow.
    """
    reference = source.predict(val[FEATURES_MODELE])
    exigences = dependances_verrouillees(GROUPE_DEPENDANCES)

    print(f"\n=== Dépendances de désérialisation (groupe « "
          f"{GROUPE_DEPENDANCES} » du lock) ===")
    print(f"  {len(exigences)} paquets épinglés : "
          f"{', '.join(exigences[:4])}...")

    print("\n=== Écriture ===")
    API_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, API_MODEL_PATH)
    taille_ko = API_MODEL_PATH.stat().st_size / 1024
    sceau = empreinte(API_MODEL_PATH)
    print(f"  {API_MODEL_PATH.name:<28} {taille_ko:>7.1f} Ko  "
          f"sha256 {sceau[:12]}…")

    empreintes = versions_des_donnees()
    etiquettes = {
        **empreintes,
        "phase": "export",
        "artefact": "service",
        "schema_entree": f"{len(FEATURES_SAISIES)}-saisies",
        "modele": version_source.tags.get("modele", "inconnu"),
        # Traçabilité vers l'étape 2 : de quelle version du registre
        # d'entraînement cet artefact descend, et par quel run elle a été
        # produite.
        "registre_source": REGISTERED_MODEL_NAME,
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
    }

    with mlflow.start_run(run_name=f"export-api-{etiquettes['modele']}"):
        mlflow.set_tags(etiquettes)
        mlflow.log_params({"version_source": version_source.version,
                           "n_val": len(val),
                           "n_saisies": len(FEATURES_SAISIES),
                           "n_colonnes_modele": len(FEATURES_MODELE)})
        mlflow.log_metrics({"parite_ecart_max": 0.0,
                            "ecart_sans_facultatives": ecart_sans,
                            "taille_artefact_ko": taille_ko})
        journaliser_jeux(mlflow, val=val)

        info = mlflow.sklearn.log_model(
            artefact,
            name="modele_api",
            # cloudpickle plutôt que skops : skops refuse `numpy.dtype` comme
            # type non fiable et ne saurait pas sérialiser le transformateur
            # maison greffé devant le modèle.
            serialization_format="cloudpickle",
            signature=construire_signature(),
            input_example=exemple_entree(val),
            pip_requirements=exigences,
        )
        version = mlflow.register_model(f"models:/{info.model_id}",
                                        REGISTERED_MODEL_API, tags=etiquettes)

    print(f"  {REGISTERED_MODEL_API} version {version.version} publiée "
          f"(sans alias)")

    # Provenance écrite à côté de l'artefact. Elle existe parce que l'image
    # n'embarque pas mlflow : sans ce fichier, le conteneur servirait un
    # modèle sans pouvoir dire lequel. `/model/info` le lit quand le modèle
    # vient du `.joblib` plutôt que du registre.
    provenance = {
        "registre": REGISTERED_MODEL_API,
        "version": int(version.version),
        # Volontairement nul : l'alias se pose après les tests d'intégration.
        # Un artefact qui s'annoncerait champion avant validation mentirait.
        "alias": None,
        **{cle: etiquettes[cle] for cle in (
            "modele", "registre_source", "version_source", "run_source",
            "schema_entree", "data_version", "split_version",
            "hypotheses_economiques", "sklearn_version", "pandas_version",
            "numpy_version", "artefact_sha256")},
        "saisies_obligatoires": FEATURES_OBLIGATOIRES,
        "saisies_facultatives": FEATURES_FACULTATIVES,
        "exporte_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    API_METADATA_PATH.write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"  {API_METADATA_PATH.name:<28} provenance -> "
          f"{REGISTERED_MODEL_NAME} v{version_source.version} "
          f"({etiquettes['modele']})")

    print("\n=== Rechargement : les trois chemins ===")
    exiger_parite(".joblib (ce que l'image embarque)",
                  joblib.load(API_MODEL_PATH).predict(val[FEATURES_SAISIES]),
                  reference)

    uri = f"models:/{REGISTERED_MODEL_API}/{version.version}"
    exiger_parite("registre, mlflow.sklearn (objet brut)",
                  mlflow.sklearn.load_model(uri).predict(val[FEATURES_SAISIES]),
                  reference)

    pyfunc = mlflow.pyfunc.load_model(uri)
    exiger_parite("registre, mlflow.pyfunc (signature appliquée)",
                  np.asarray(pyfunc.predict(val[FEATURES_SAISIES])).ravel(),
                  reference)

    # Le contrat doit aussi **refuser** ce qu'il déclare obligatoire : une
    # signature qui laisse tout passer ne protège de rien.
    try:
        pyfunc.predict(val[FEATURES_SAISIES].drop(columns=["Crop"]))
    except Exception:
        print(f"  {'saisie obligatoire absente : rejetée':<44} OK")
    else:
        raise RuntimeError("La signature n'exige pas `Crop`. Contrat invalide.")

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


def _ecrire_provenance(provenance: dict) -> None:
    """Écrit le fichier de provenance par remplacement, non par troncature.

    `write_text` ouvre en écrasant : une interruption en cours d'écriture
    laisserait un JSON tronqué, que l'API refuserait ensuite de charger. Le
    fichier temporaire est créé dans le même dossier — `os.replace` peut
    échouer entre deux systèmes de fichiers.

    La doc Python attribue l'atomicité du remplacement à une **exigence
    POSIX** et n'en donne aucune sous Windows. Ce détour supprime donc le cas
    du fichier à moitié écrit ; il ne rend pas l'opération transactionnelle.
    """
    temporaire = API_METADATA_PATH.with_suffix(".json.tmp")
    temporaire.write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(temporaire, API_METADATA_PATH)


def promouvoir(mlflow) -> None:
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

    if not API_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Provenance absente : {API_METADATA_PATH}. Exécuter l'export "
            "avant toute promotion.")

    provenance = json.loads(API_METADATA_PATH.read_text(encoding="utf-8"))

    # Une provenance écrite par une version antérieure du script n'a pas
    # toutes les clés contrôlées. Le dire vaut mieux qu'un KeyError nu.
    manquantes = [cle for cle in CLES_CONCORDANCE if cle not in provenance]
    if manquantes:
        raise RuntimeError(
            f"{API_METADATA_PATH.name} est antérieur à ces clés de "
            f"traçabilité : {', '.join(manquantes)}.\n"
            "  Réexécuter l'export pour produire une provenance complète.")

    registre, version = provenance["registre"], int(provenance["version"])
    client = MlflowClient()
    enregistree = client.get_model_version(registre, str(version))

    ecarts = {cle: (provenance[cle], enregistree.tags.get(cle))
              for cle in CLES_CONCORDANCE
              if str(provenance[cle]) != str(enregistree.tags.get(cle))}
    if ecarts:
        raise RuntimeError(
            f"La provenance du disque ne décrit pas {registre} v{version}.\n"
            f"  écarts (disque, registre) : {ecarts}\n"
            "  Réexécuter l'export avant de promouvoir.")
    print(f"\n=== Concordance ===")
    print(f"  {len(CLES_CONCORDANCE)} clés identiques entre "
          f"{API_METADATA_PATH.name} et {registre} v{version}")

    # L'empreinte est recalculée sur le fichier **présent**. Comparer la
    # provenance au registre ne dirait rien d'un `.joblib` remplacé depuis
    # l'export : les deux porteraient la même valeur périmée.
    if not API_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Artefact absent : {API_MODEL_PATH}. Promouvoir une version dont "
            "le binaire manque laisserait l'image sans modèle à servir.")
    constatee = empreinte(API_MODEL_PATH)
    if constatee != provenance["artefact_sha256"]:
        raise RuntimeError(
            f"{API_MODEL_PATH.name} n'est pas le fichier qui a été exporté.\n"
            f"  enregistrée : {provenance['artefact_sha256']}\n"
            f"  constatée   : {constatee}\n"
            "  Réexécuter l'export avant de promouvoir.")
    print(f"  {API_MODEL_PATH.name} scellé        sha256 {constatee[:12]}…")

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
        _ecrire_provenance(provenance)
    except OSError as erreur:
        # L'alias est posé, le fichier ne l'est pas : c'est précisément la
        # divergence que cette fonction ne peut pas empêcher. On la nomme.
        raise RuntimeError(
            f"DIVERGENCE — « {ALIAS_CHAMPION} » désigne {registre} v{version} "
            f"au registre, mais {API_METADATA_PATH.name} n'a pas pu être "
            f"écrit ({erreur}).\n"
            "  L'artefact du disque annonce donc un statut faux.\n"
            "  Reprise : corriger la cause, puis relancer la même commande. "
            "Elle est idempotente et résorbe l'écart sans créer de version.\n"
            + (f"  Retour arrière : l'alias désignait v{precedente}.\n"
               if precedente is not None else "")
        ) from erreur

    # Les deux écritures ont rendu la main : on relit plutôt que de supposer.
    relue = json.loads(API_METADATA_PATH.read_text(encoding="utf-8"))
    pose = int(client.get_model_version_by_alias(registre,
                                                 ALIAS_CHAMPION).version)
    if relue["alias"] != ALIAS_CHAMPION or pose != version:
        raise RuntimeError(
            f"Vérification finale : le registre désigne v{pose}, le fichier "
            f"porte alias={relue['alias']!r}. Relancer la commande.")

    print(f"  {registre} v{version}  <-  « {ALIAS_CHAMPION} »")
    print(f"  {API_METADATA_PATH.name:<28} alias = {ALIAS_CHAMPION!r}")
    print(f"  {'relecture des deux côtés':<28} concordent")
    print("\n  L'API peut désormais servir depuis le registre :")
    print(f"    $env:AGRITECH_MODEL_URI = "
          f"\"models:/{registre}@{ALIAS_CHAMPION}\"")
    print("    uv run uvicorn api.main:app")


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def analyser_arguments():
    """Les options de la ligne de commande.

    `--sample` implique `--verifier-seulement` : la table FAO doit être
    ajustée sur l'intégralité de `train`, un artefact construit sur un
    échantillon ne doit donc jamais être publié.
    """
    parseur = argparse.ArgumentParser(
        description="Assemble et publie l'artefact de service de l'API.")
    parseur.add_argument(
        "--verifier-seulement", action="store_true",
        help="exécute toutes les vérifications sans rien écrire")
    parseur.add_argument(
        "--promouvoir", action="store_true",
        help="pose l'alias champion sur la version décrite par la provenance "
             "et met ce fichier à jour — sans rien réexporter")
    parseur.add_argument(
        "--sample", type=int, default=None,
        help="sous-échantillonne train — implique --verifier-seulement, la "
             "table FAO devant être ajustée sur l'intégralité de train")
    return parseur.parse_args()


def main():
    """Vérifie, puis publie — jamais l'inverse.

    Rien n'est écrit tant que la parité avec le modèle d'origine n'est pas
    établie, et la publication ne déplace pas l'alias `champion` : la
    promotion reste un geste explicite, postérieur aux tests de l'API.
    """
    arguments = analyser_arguments()

    # La promotion est un geste à part : elle ne réexporte rien, elle désigne.
    # La combiner à un export ferait promouvoir une version dans le même souffle
    # qu'on la publie, c'est-à-dire avant de l'avoir testée.
    if arguments.promouvoir:
        if arguments.verifier_seulement or arguments.sample:
            raise SystemExit("--promouvoir ne se combine ni avec "
                             "--verifier-seulement ni avec --sample.")
        promouvoir(configurer_mlflow(JEU_PAR_DEFAUT))
        return

    verifier_seulement = arguments.verifier_seulement or bool(arguments.sample)
    if arguments.sample and not arguments.verifier_seulement:
        print("--sample implique --verifier-seulement : rien ne sera publié.\n")

    mlflow = configurer_mlflow(JEU_PAR_DEFAUT)
    source, version_source = resoudre_source(mlflow)

    train, val = charger_partitions(echantillon=arguments.sample)
    print(f"\n=== Assemblage ===")
    artefact = construire_artefact(source, train)

    ecart_sans = verifier_avant_ecriture(artefact, source, val)

    if verifier_seulement:
        print("\nToutes les vérifications passent. Rien n'a été écrit "
              "(--verifier-seulement).")
        return

    version = publier(mlflow, artefact, source, version_source, val, ecart_sans)

    print("\n=== Publié, pas promu ===")
    print(f"  {REGISTERED_MODEL_API} version {version} n'a aucun alias.")
    print(f"  L'alias « {ALIAS_CHAMPION} » se pose après les tests "
          f"d'intégration de l'API :")
    print("    uv run python scripts/export_api_model.py --promouvoir")
    # Une commande de ce script, et non `mlflow` en direct : poser l'alias à la
    # main laisserait `alias` à `null` dans le fichier de provenance, et
    # l'artefact embarqué mentirait sur son propre statut. Les deux doivent
    # bouger ensemble.


if __name__ == "__main__":
    main()

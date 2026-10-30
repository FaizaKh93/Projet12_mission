"""Orchestration de la chaîne, de six CSV à l'artefact servi par l'API.

```
préparation  →  découpage  →  entraînement  →  export
   data/          splits/        mlflow.db      models/modele_api.joblib
```

**Ce que l'orchestrateur garantit, et qu'une suite de commandes ne garantit
pas.** Les quatre étapes ne sont pas indépendantes : chacune consomme ce que la
précédente a écrit, et les empreintes `data_version` et `split_version` lient
le tout. Relancer la préparation sans redécouper, c'est entraîner sur des
partitions qui décrivent un autre dataset — une erreur que rien ne signale
avant que les métriques ne soient devenues incomparables.

Ce module vérifie ces liens avant d'agir, et refuse plutôt que de deviner.

**L'entraînement ne part jamais tout seul.** Il coûte une demi-heure et écrit
dans le registre MLflow. Il faut le demander : `--jusqua entrainement`. Par
défaut, la chaîne s'arrête au découpage et dit ce qu'il reste à faire.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from agritech import decoupage, preparation
from agritech.config import (
    API_METADATA_PATH,
    CONSOLIDATED_PARQUET_PATH,
    CONSOLIDATED_PATH,
    CROP_REFERENCE_PATH,
    DATA_DICTIONARY_PATH,
    FAO_PANEL_PATH,
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SPLIT_METADATA_PATH,
)

#: Les étapes, dans l'ordre où elles doivent se dérouler.
ETAPES = ["preparation", "decoupage", "entrainement", "export"]


@dataclass(frozen=True)
class Resultat:
    """Ce qu'une étape a fait, pour le compte rendu final.

    `bilan` porte les chiffres de l'étape quand elle en produit, pour éviter
    que l'appelant ne refasse le travail juste pour les afficher.
    """

    etape: str
    agi: bool
    detail: str
    bilan: dict | None = None


# --------------------------------------------------------------------------- #
# Écriture du livrable
# --------------------------------------------------------------------------- #


def ecrire_preparation(
    resultat: preparation.Preparation,
    processed: Path | None = None,
    interim: Path | None = None,
) -> dict[str, Path]:
    """Écrit le dataset consolidé et les artefacts reconstructibles.

    Deux formats pour le consolidé : le CSV est le livrable demandé par la
    mission, lisible partout ; le Parquet porte les mêmes données avec leur
    typage, et c'est lui que le reste de la chaîne lit.

    Les dossiers sont paramétrables pour que la vérification puisse écrire
    ailleurs et comparer sans rien écraser. Ils sont résolus **à l'appel** et
    non à la définition : une valeur par défaut figée dans la signature
    pointerait sur le vrai `data/` même en test.
    """
    processed = processed or PROCESSED_DIR
    interim = interim or INTERIM_DIR

    processed.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)

    chemins = {
        "csv": processed / CONSOLIDATED_PATH.name,
        "parquet": processed / CONSOLIDATED_PARQUET_PATH.name,
        "reference": interim / CROP_REFERENCE_PATH.name,
        "panel": interim / FAO_PANEL_PATH.name,
        "dictionnaire": interim / DATA_DICTIONARY_PATH.name,
    }

    resultat.dataset.to_csv(chemins["csv"], index=False)
    resultat.dataset.to_parquet(chemins["parquet"], index=False, compression="snappy")
    resultat.reference.to_csv(chemins["reference"])
    resultat.panel.to_csv(chemins["panel"], index=False)
    resultat.dictionnaire.to_csv(chemins["dictionnaire"])
    return chemins


# --------------------------------------------------------------------------- #
# Vérification de non-régression
# --------------------------------------------------------------------------- #


def verifier_preparation() -> dict:
    """Le dataset régénéré est-il celui qui est sur le disque ?

    C'est le contrôle qui autorise à extraire la préparation d'un notebook sans
    rien casser. Il écrit dans un dossier temporaire, jamais dans `data/`.

    Il répond à **deux questions distinctes**, et leur différence importe :

    - *le contenu* est-il identique — colonnes, types, valeurs ? C'est
      l'invariant réel, celui qui dit que le code extrait fait bien la même
      chose que le notebook ;
    - *les octets* sont-ils identiques ? C'est `data_version`, l'invariant
      opérationnel. Il peut différer alors que le contenu concorde : une
      nouvelle version de pyarrow sérialise le même tableau autrement. Dans ce
      cas les runs MLflow passés restent valides sur le fond, mais leurs tags
      ne correspondront plus — il faut le savoir avant d'écraser quoi que ce soit.
    """
    attendu = pd.read_parquet(CONSOLIDATED_PARQUET_PATH)
    obtenu = preparation.preparer().dataset

    rapport = {
        "colonnes_identiques": list(obtenu.columns) == list(attendu.columns),
        "types_identiques": bool((obtenu.dtypes == attendu.dtypes).all()),
        "lignes": (len(obtenu), len(attendu)),
    }
    rapport["contenu_identique"] = (
        rapport["colonnes_identiques"]
        and rapport["types_identiques"]
        and obtenu.equals(attendu)
    )

    with tempfile.TemporaryDirectory(prefix="agritech-verif-") as dossier:
        temporaire = Path(dossier) / CONSOLIDATED_PARQUET_PATH.name
        obtenu.to_parquet(temporaire, index=False, compression="snappy")
        rapport["data_version_obtenue"] = decoupage.empreinte(temporaire)

    rapport["data_version_attendue"] = decoupage.empreinte(CONSOLIDATED_PARQUET_PATH)
    rapport["octets_identiques"] = (
        rapport["data_version_obtenue"] == rapport["data_version_attendue"]
    )
    return rapport


# --------------------------------------------------------------------------- #
# Étapes
# --------------------------------------------------------------------------- #


def livrables() -> tuple[Path, ...]:
    """Les cinq fichiers que la préparation produit.

    Ils forment un **lot** : les artefacts de `interim/` décrivent le dataset
    de `processed/`, et les séparer n'aurait pas de sens.

    Une fonction, et non une constante : figée à l'import, elle pointerait sur
    le vrai `data/` même là où les chemins ont été redirigés. C'est la raison
    qui interdit déjà une valeur par défaut dans une signature.
    """
    return (
        CONSOLIDATED_PATH,
        CONSOLIDATED_PARQUET_PATH,
        CROP_REFERENCE_PATH,
        FAO_PANEL_PATH,
        DATA_DICTIONARY_PATH,
    )


def etat_preparation() -> tuple[bool, str]:
    """La préparation a-t-elle déjà produit son livrable, en entier ?"""
    attendus = livrables()
    presents = [c for c in attendus if c.exists()]
    if not presents:
        return False, "aucun livrable"
    if len(presents) < len(attendus):
        return False, f"livrable incomplet ({len(presents)}/{len(attendus)} fichiers)"
    return True, f"data_version {decoupage.empreinte(CONSOLIDATED_PARQUET_PATH)}"


def decider_preparation(forcer: bool = False) -> decoupage.Decision:
    """Faut-il construire le livrable, le vérifier, ou refuser ?

    Même discipline que pour les partitions, et pour la même raison : un lot
    **partiel** est le vestige d'une exécution interrompue ou d'une suppression
    manuelle. Les fichiers restants peuvent décrire un autre dataset, et les
    écraser changerait `data_version` sans que rien ne le signale.
    """
    attendus = livrables()
    presents = [c for c in attendus if c.exists()]

    if not presents:
        return decoupage.Decision(True, "aucun livrable")

    if forcer:
        return decoupage.Decision(True, "reconstruction explicitement demandée")

    if len(presents) < len(attendus):
        manquants = [c.name for c in attendus if not c.exists()]
        raise RuntimeError(
            f"LIVRABLE PARTIEL — {len(presents)}/{len(attendus)} fichiers présents.\n"
            f"  manquants : {', '.join(manquants)}\n"
            "  Les fichiers restants peuvent décrire un autre dataset : les écraser\n"
            "  changerait `data_version` et périmerait les runs MLflow existants.\n"
            "  Vider data/processed/ et data/interim/ puis relancer, ou --forcer."
        )

    return decoupage.Decision(False, etat_preparation()[1])


def lancer_preparation(forcer: bool = False) -> Resultat:
    """Construit le dataset consolidé, sans jamais écraser à l'aveugle.

    Si un livrable complet existe déjà, on vérifie que le code reproduit le
    même contenu. Tant que ce n'est pas le cas, écraser ferait perdre la
    correspondance avec tous les runs MLflow existants.
    """
    decision = decider_preparation(forcer=forcer)

    if not decision.ecrire:
        rapport = verifier_preparation()
        if not rapport["contenu_identique"]:
            raise RuntimeError(
                "Le dataset régénéré diffère de celui du disque.\n"
                f"  colonnes identiques : {rapport['colonnes_identiques']}\n"
                f"  types identiques    : {rapport['types_identiques']}\n"
                f"  lignes              : {rapport['lignes'][0]:,} contre "
                f"{rapport['lignes'][1]:,}\n"
                "  Écraser périmerait les runs MLflow. Relancer avec --forcer\n"
                "  si la différence est voulue."
            )
        return Resultat(
            "preparation", False, f"déjà à jour, contenu vérifié — {decision.motif}"
        )

    resultat = preparation.preparer()
    bilan = preparation.controler(resultat)
    ecrire_preparation(resultat)
    return Resultat(
        "preparation",
        True,
        f"{bilan['lignes']:,} lignes × {bilan['colonnes']} colonnes, "
        f"data_version {decoupage.empreinte(CONSOLIDATED_PARQUET_PATH)}",
        bilan=bilan,
    )


def lancer_decoupage(forcer: bool = False) -> Resultat:
    """Découpe le dataset, ou constate que les partitions sont déjà bonnes."""
    dataset, data_version = decoupage.charger_dataset()
    decision = decoupage.decider(data_version, forcer=forcer)

    # Le découpage est refait dans tous les cas, y compris quand on n'écrit
    # pas : il est déterministe, donc le comparer aux fichiers présents
    # détecterait des partitions issues d'un autre tirage.
    partitions = decoupage.decouper(dataset)
    decoupage.controler(partitions, dataset)

    if not decision.ecrire:
        metadonnees = decoupage.metadonnees_du_disque()
        decoupage.relire_et_verifier(metadonnees, partitions)
        return Resultat(
            "decoupage",
            False,
            f"{decision.motif} — split_version {metadonnees['split_version']}",
        )

    metadonnees = decoupage.ecrire(partitions, dataset, data_version)
    decoupage.relire_et_verifier(metadonnees, partitions)
    return Resultat(
        "decoupage",
        True,
        f"{decision.motif} — split_version {metadonnees['split_version']}, "
        + " / ".join(f"{n} {e:,}" for n, e in metadonnees["effectifs"].items()),
    )


def _executer(script: str, *arguments: str) -> None:
    """Lance un script du projet dans un processus neuf.

    En sous-processus plutôt qu'en important son `main()` : `train.py` et
    `export_api_model.py` analysent `sys.argv`, qu'ils interpréteraient comme
    le leur. Un processus séparé rend aussi l'étape interruptible sans laisser
    l'orchestrateur dans un état indéterminé.
    """
    acheve = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *arguments],
        cwd=PROJECT_ROOT,
    )
    if acheve.returncode != 0:
        raise RuntimeError(f"{script} s'est arrêté avec le code {acheve.returncode}")


def lancer_entrainement(forcer: bool = False) -> Resultat:
    """Lance les quatre études Optuna. Une demi-heure, et des écritures MLflow.

    Jamais ignoré au motif qu'un modèle existerait déjà : un entraînement est
    une décision, pas une conséquence. S'il est demandé, il a lieu.
    """
    _executer("train.py")
    return Resultat("entrainement", True, "4 études × 100 essais, runs dans mlflow.db")


def etat_export() -> tuple[bool, str]:
    """L'artefact servi décrit-il les partitions actuelles ?"""
    if not API_METADATA_PATH.exists():
        return False, "aucun artefact exporté"
    if not SPLIT_METADATA_PATH.exists():
        return False, "aucune partition de référence"

    import json

    provenance = json.loads(API_METADATA_PATH.read_text(encoding="utf-8"))
    partitions = decoupage.metadonnees_du_disque()

    if provenance.get("split_version") != partitions["split_version"]:
        return False, (
            f"artefact sur split_version {provenance.get('split_version')}, "
            f"partitions sur {partitions['split_version']}"
        )
    return True, f"split_version {partitions['split_version']}"


def lancer_export(forcer: bool = False) -> Resultat:
    """Exporte l'artefact de l'API depuis le champion du registre."""
    a_jour, detail = etat_export()
    if a_jour and not forcer:
        return Resultat("export", False, f"déjà à jour — {detail}")

    _executer("export_api_model.py")
    return Resultat("export", True, detail)


ACTIONS = {
    "preparation": lancer_preparation,
    "decoupage": lancer_decoupage,
    "entrainement": lancer_entrainement,
    "export": lancer_export,
}


# --------------------------------------------------------------------------- #
# Enchaînement
# --------------------------------------------------------------------------- #


def executer(
    depuis: str = "preparation",
    jusqua: str = "decoupage",
    forcer: bool = False,
) -> list[Resultat]:
    """Déroule les étapes de `depuis` à `jusqua`, incluses."""
    if depuis not in ETAPES or jusqua not in ETAPES:
        raise ValueError(f"Étapes connues : {', '.join(ETAPES)}")
    if ETAPES.index(depuis) > ETAPES.index(jusqua):
        raise ValueError(f"`{depuis}` vient après `{jusqua}` dans la chaîne")

    choisies = ETAPES[ETAPES.index(depuis): ETAPES.index(jusqua) + 1]
    return [ACTIONS[etape](forcer=forcer) for etape in choisies]

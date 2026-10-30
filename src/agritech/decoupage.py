"""Découpage train / validation / test, et les garde-fous qui vont avec.

**Pourquoi ce module existe.** Les partitions sont écrites **une seule fois**,
puis relues par chaque entraînement. Les réécrire sans le vouloir périmerait
tous les runs MLflow qui s'y réfèrent : leurs métriques ne décriraient plus les
mêmes données, sans que rien ne le signale. Le notebook protégeait cela par un
verrou manuel en haut d'une cellule ; ici, c'est le code qui refuse.

Deux empreintes circulent dans tout le projet :

- `data_version` — le contenu du dataset consolidé ;
- `split_version` — les octets des trois partitions, dans un ordre fixe.

Elles répondent à deux questions différentes. La première dit *de quel dataset*
un run provient ; la seconde, *de quels fichiers*. Deux partitions issues du
même dataset mais réécrites par une autre version de pyarrow porteraient le même
`data_version` et un `split_version` différent — et c'est voulu, puisque ce sont
ces octets-là que l'entraînement a lus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from agritech.config import (
    CONSOLIDATED_PARQUET_PATH,
    SPLIT_METADATA_PATH,
    SPLIT_PROPORTIONS,
    SPLIT_RANDOM_STATE,
    SPLIT_STRATIFY_ON,
    SPLITS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)

#: Les trois partitions et leur fichier, dans l'ordre qui définit `split_version`.
CHEMINS = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}

#: Le lot complet : des partitions sans leurs métadonnées ne sont pas
#: exploitables, puisque rien ne dirait de quel dataset elles proviennent.
ARTEFACTS = {**CHEMINS, "metadata": SPLIT_METADATA_PATH}


@dataclass(frozen=True)
class Decision:
    """Ce que le module a décidé de faire, et pourquoi."""

    ecrire: bool
    motif: str


def empreinte(chemin: Path, longueur: int = 12) -> str:
    """SHA-256 par blocs d'un mégaoctet : le consolidé pèse plusieurs dizaines de Mo."""
    digest = hashlib.sha256()
    with chemin.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1 << 20), b""):
            digest.update(bloc)
    return digest.hexdigest()[:longueur]


def charger_dataset() -> tuple[pd.DataFrame, str]:
    """Relit le dataset consolidé et calcule son empreinte."""
    dataset = pd.read_parquet(CONSOLIDATED_PARQUET_PATH)
    if not dataset.index.is_unique:
        raise ValueError("Index non unique : les contrôles de chevauchement seraient faussés")
    return dataset, empreinte(CONSOLIDATED_PARQUET_PATH)


# --------------------------------------------------------------------------- #
# Découpage
# --------------------------------------------------------------------------- #


def decouper(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """70 / 15 / 15, stratifié par culture, en deux temps.

    `train_test_split` ne sait couper qu'en deux. On isole d'abord `test`, puis
    on recoupe le reste — d'où la proportion recalculée, sans quoi `val` serait
    trop petite. La stratification est refaite sur le reste, pas héritée.
    """
    part_test = SPLIT_PROPORTIONS["test"]
    part_val_dans_reste = SPLIT_PROPORTIONS["val"] / (1 - part_test)

    reste, test = train_test_split(
        dataset,
        test_size=part_test,
        random_state=SPLIT_RANDOM_STATE,
        stratify=dataset[SPLIT_STRATIFY_ON],
        shuffle=True,
    )
    train, val = train_test_split(
        reste,
        test_size=part_val_dans_reste,
        random_state=SPLIT_RANDOM_STATE,
        stratify=reste[SPLIT_STRATIFY_ON],
        shuffle=True,
    )
    return {"train": train, "val": val, "test": test}


def controler(partitions: dict[str, pd.DataFrame], dataset: pd.DataFrame) -> dict:
    """Les trois partitions se partagent le dataset, sans perte ni chevauchement.

    Une ligne présente dans deux partitions ferait fuiter l'entraînement vers
    l'évaluation. Une ligne perdue fausserait les effectifs. Les deux se
    détectent sur les index, avant toute écriture.
    """
    index = {nom: set(d.index) for nom, d in partitions.items()}

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        commun = index[a] & index[b]
        if commun:
            raise ValueError(f"{len(commun)} lignes communes entre {a} et {b}")

    reunion = set().union(*index.values())
    if reunion != set(dataset.index):
        manquantes = len(set(dataset.index) - reunion)
        raise ValueError(f"{manquantes} lignes du dataset n'appartiennent à aucune partition")

    return {
        nom: {
            "lignes": len(d),
            "part_%": round(100 * len(d) / len(dataset), 2),
            "part_visee_%": 100 * SPLIT_PROPORTIONS[nom],
        }
        for nom, d in partitions.items()
    }


# --------------------------------------------------------------------------- #
# Écriture, et ce qui l'empêche
# --------------------------------------------------------------------------- #


def decider(data_version: str, forcer: bool = False) -> Decision:
    """Faut-il écrire, ne rien faire, ou refuser ?

    Quatre cas, et un seul autorise l'écriture silencieuse : l'absence totale
    d'artefacts. Un lot incomplet est le vestige d'une exécution interrompue —
    les fichiers présents peuvent provenir de découpages différents, et deviner
    serait pire que s'arrêter.
    """
    presents = [nom for nom, chemin in ARTEFACTS.items() if chemin.exists()]

    if not presents:
        return Decision(True, "aucun artefact existant")

    if forcer:
        return Decision(True, "écrasement explicitement demandé")

    if len(presents) < len(ARTEFACTS):
        manquants = [n for n in ARTEFACTS if n not in presents]
        raise RuntimeError(
            f"État partiel. Présents : {', '.join(presents)} | "
            f"manquants : {', '.join(manquants)}. Supprimer le dossier splits/ "
            f"et relancer, ou passer --forcer."
        )

    precedent = json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))
    if precedent["data_version"] != data_version:
        raise RuntimeError(
            "PARTITIONS PÉRIMÉES — elles décrivent un autre dataset.\n"
            f"  sur disque     : data_version {precedent['data_version']}, "
            f"{len(precedent['colonnes'])} colonnes, généré le {precedent['genere_le']}\n"
            f"  dataset actuel : data_version {data_version}\n"
            "  La préparation des données a changé depuis. Relancer avec --forcer\n"
            "  pour regénérer — en sachant que les runs MLflow existants deviendront\n"
            "  incomparables."
        )

    return Decision(False, "lot complet et à jour")


def calculer_split_version() -> str:
    """Empreinte des octets des trois partitions, dans un ordre fixe."""
    combinee = "".join(empreinte(chemin, 64) for chemin in CHEMINS.values())
    return hashlib.sha256(combinee.encode()).hexdigest()[:12]


def ecrire(
    partitions: dict[str, pd.DataFrame],
    dataset: pd.DataFrame,
    data_version: str,
) -> dict:
    """Écrit les trois partitions puis leurs métadonnées, et rend celles-ci.

    L'ordre compte : `split_version` se calcule sur les fichiers écrits, donc
    après leur écriture et avant celle des métadonnées qui le portent.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for nom, partition in partitions.items():
        partition.to_parquet(CHEMINS[nom], index=False)

    metadonnees = {
        "data_version": data_version,
        "split_version": calculer_split_version(),
        "source": CONSOLIDATED_PARQUET_PATH.name,
        "random_state": SPLIT_RANDOM_STATE,
        "stratify_on": SPLIT_STRATIFY_ON,
        "proportions": SPLIT_PROPORTIONS,
        "effectifs": {nom: len(d) for nom, d in partitions.items()},
        "colonnes": list(dataset.columns),
        "genere_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    SPLIT_METADATA_PATH.write_text(json.dumps(metadonnees, indent=2), encoding="utf-8")
    return metadonnees


def metadonnees_du_disque() -> dict:
    """Les métadonnées présentes, qui font foi quand rien n'a été réécrit."""
    return json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))


def relire_et_verifier(
    metadonnees: dict, partitions: dict[str, pd.DataFrame] | None = None
) -> dict:
    """Relit les partitions écrites et confronte leurs effectifs aux métadonnées.

    C'est le contrôle de sortie : ce qui est sur le disque est-il bien ce que
    les métadonnées annoncent ? Un `split_version` qui ne correspond plus
    signale des fichiers réécrits en dehors de cette chaîne.

    `partitions` ajoute la comparaison des **valeurs** : les fichiers portent-ils
    bien ce qui vient d'être calculé ? Le découpage étant déterministe, le
    contrôle vaut aussi quand rien n'a été réécrit — il détecterait alors des
    fichiers issus d'un autre tirage.
    """
    relues = {nom: pd.read_parquet(chemin) for nom, chemin in CHEMINS.items()}

    for nom, partition in relues.items():
        attendu = metadonnees["effectifs"][nom]
        if len(partition) != attendu:
            raise ValueError(
                f"{nom} : {len(partition):,} lignes sur le disque, "
                f"{attendu:,} annoncées par les métadonnées"
            )
        if list(partition.columns) != metadonnees["colonnes"]:
            raise ValueError(f"{nom} : colonnes différentes de celles annoncées")

        if partitions is not None:
            # `reset_index` : l'écriture a perdu l'index, les valeurs non.
            calculee = partitions[nom][TARGET].reset_index(drop=True)
            ecart = float((partition[TARGET] - calculee).abs().max())
            if not ecart < 1e-12:
                raise ValueError(
                    f"{nom} : la cible du disque diffère de celle calculée "
                    f"(écart maximal {ecart:.3e})"
                )

    constate = calculer_split_version()
    if constate != metadonnees["split_version"]:
        raise ValueError(
            f"split_version {constate} sur le disque, "
            f"{metadonnees['split_version']} dans les métadonnées : "
            "les fichiers ont été réécrits hors de cette chaîne"
        )

    return {nom: len(d) for nom, d in relues.items()}

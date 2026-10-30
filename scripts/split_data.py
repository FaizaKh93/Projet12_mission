"""Point d'entrée du découpage : train / validation / test, une seule fois.

Les partitions sont écrites **une fois**, puis relues par chaque entraînement.
Les réécrire sans le vouloir périmerait tous les runs MLflow qui s'y réfèrent :
leurs métriques ne décriraient plus les mêmes données, sans que rien ne le
signale. Ce script refuse donc d'écraser, sauf demande explicite.

Il refuse aussi dans deux autres cas :

- **lot incomplet** — vestige d'une exécution interrompue. Les fichiers
  présents peuvent provenir de découpages différents, et deviner serait pire
  que s'arrêter ;
- **partitions périmées** — elles portent un `data_version` qui n'est plus
  celui du dataset. C'est le cas courant après une modification de la
  préparation.

Exemples :

    # Découper, ou constater que les partitions sont déjà bonnes
    uv run python scripts/split_data.py

    # Regénérer après une modification voulue de la préparation
    uv run python scripts/split_data.py --forcer
"""

from __future__ import annotations

import argparse
import sys

from agritech import decoupage, pipeline
from agritech.config import SPLIT_PROPORTIONS


def analyser_arguments():
    parseur = argparse.ArgumentParser(
        description="Découpe le dataset consolidé en train / validation / test.")
    parseur.add_argument(
        "--forcer", action="store_true",
        help="réécrit les partitions existantes — périme les runs MLflow qui s'y réfèrent")
    return parseur.parse_args()


def main() -> int:
    arguments = analyser_arguments()

    try:
        resultat = pipeline.lancer_decoupage(forcer=arguments.forcer)
    except RuntimeError as refus:
        # Un refus est un diagnostic, pas un plantage : il porte déjà
        # l'explication et la marche à suivre.
        print(refus, file=sys.stderr)
        return 1

    metadonnees = decoupage.metadonnees_du_disque()

    print("Partitions")
    for nom, effectif in metadonnees["effectifs"].items():
        part = 100 * effectif / sum(metadonnees["effectifs"].values())
        print(f"  {nom:<8} {effectif:>10,} lignes   {part:>5.2f} %"
              f"   (visé {100 * SPLIT_PROPORTIONS[nom]:.0f} %)")
    print()
    print(f"  data_version  {metadonnees['data_version']}")
    print(f"  split_version {metadonnees['split_version']}")
    print(f"  stratifié par {metadonnees['stratify_on']}, "
          f"random_state {metadonnees['random_state']}")
    print()
    print(f"Découpage : {resultat.detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

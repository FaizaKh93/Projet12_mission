"""Point d'entrée de la chaîne complète : des six CSV à l'artefact servi.

```
préparation  →  découpage  →  entraînement  →  export
   data/          splits/        mlflow.db      models/modele_api.joblib
```

Chaque étape consomme ce que la précédente a écrit, et deux empreintes lient le
tout : `data_version` pour le contenu du dataset, `split_version` pour les
octets des partitions. Lancer une étape sur des entrées périmées est l'erreur
que cette chaîne rend impossible — elle refuse plutôt que de deviner.

**L'entraînement ne part pas tout seul.** Il coûte une demi-heure et écrit dans
le registre MLflow. Par défaut la chaîne s'arrête au découpage, et dit ce qu'il
reste à faire. Il faut le demander : `--jusqua entrainement`.

Exemples :

    # Préparer et découper — ce qu'on lance après un clone neuf
    uv run python scripts/run_pipeline.py

    # La chaîne entière, entraînement compris (~30 min)
    uv run python scripts/run_pipeline.py --jusqua export

    # Reprendre après coup, sans refaire la préparation
    uv run python scripts/run_pipeline.py --depuis decoupage --jusqua export
"""

from __future__ import annotations

import argparse
import sys
import time

from agritech import pipeline

#: Ce que chaque étape produit, pour le compte rendu.
PRODUIT = {
    "preparation": "data/processed/ et data/interim/",
    "decoupage": "data/processed/splits/",
    "entrainement": "runs et versions dans mlflow.db",
    "export": "models/modele_api.joblib et sa provenance",
}


def analyser_arguments():
    parseur = argparse.ArgumentParser(
        description="Déroule la chaîne MLOps, d'une étape à une autre.")
    parseur.add_argument(
        "--depuis", choices=pipeline.ETAPES, default="preparation",
        help="première étape à exécuter (défaut : preparation)")
    parseur.add_argument(
        "--jusqua", choices=pipeline.ETAPES, default="decoupage",
        help="dernière étape à exécuter (défaut : decoupage, pour ne pas "
             "déclencher un entraînement sans le vouloir)")
    parseur.add_argument(
        "--forcer", action="store_true",
        help="réécrit les artefacts existants — périme les runs MLflow")
    return parseur.parse_args()


def main() -> int:
    arguments = analyser_arguments()
    choisies = pipeline.ETAPES[
        pipeline.ETAPES.index(arguments.depuis): pipeline.ETAPES.index(arguments.jusqua) + 1
    ]

    print(f"Chaîne : {' → '.join(choisies)}")
    if "entrainement" in choisies:
        print("L'entraînement est inclus : comptez une trentaine de minutes.")
    print()

    resultats = []
    debut = time.perf_counter()

    for etape in choisies:
        print(f"── {etape} ".ljust(78, "─"))
        try:
            resultat = pipeline.ACTIONS[etape](forcer=arguments.forcer)
        except RuntimeError as refus:
            # Un refus porte son diagnostic : on l'affiche tel quel et on
            # s'arrête, sans entamer les étapes suivantes sur des entrées
            # dont on vient d'établir qu'elles ne conviennent pas.
            print()
            print(refus, file=sys.stderr)
            print()
            print(f"Arrêt à l'étape « {etape} ». "
                  f"{len(choisies) - len(resultats) - 1} étape(s) non exécutée(s).")
            return 1

        resultats.append(resultat)
        marque = "exécutée" if resultat.agi else "ignorée "
        print(f"   {marque}  {resultat.detail}")
        print()

    print("─" * 78)
    for resultat in resultats:
        marque = "✓" if resultat.agi else "·"
        print(f"  {marque} {resultat.etape:<14} {PRODUIT[resultat.etape]}")
    print(f"\nTerminé en {time.perf_counter() - debut:.1f} s.")

    # Ce qui reste, pour que l'arrêt par défaut ne passe pas pour la fin.
    restantes = pipeline.ETAPES[pipeline.ETAPES.index(arguments.jusqua) + 1:]
    if restantes:
        print(f"Étapes suivantes, non lancées : {', '.join(restantes)}")
        print("  uv run python scripts/run_pipeline.py "
              f"--depuis {restantes[0]} --jusqua {pipeline.ETAPES[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Point d'entrée de la préparation : des six CSV sources au dataset consolidé.

Différence avec `src/agritech/preparation.py` : ce module-ci est une
**bibliothèque** de fonctions pures, que le notebook importe pour commenter ses
sorties. Ce fichier est un **script** — il lit la ligne de commande, appelle la
bibliothèque, écrit sur le disque et rend compte.

Ce que produit une exécution :

- `data/processed/crop_yield_consolidated.csv` — le livrable de la mission ;
- `data/processed/crop_yield_consolidated.parquet` — mêmes données, typage
  préservé, et c'est ce fichier que le reste de la chaîne lit ;
- `data/interim/` — panel FAO, référentiel par culture, dictionnaire des
  variables. Reconstructibles, donc non versionnés.

**Rien n'est écrasé à l'aveugle.** Si un livrable existe déjà, le script
régénère le dataset et vérifie qu'il produit le même contenu avant de conclure.
Écraser un dataset différent changerait `data_version` et rendrait incomparables
tous les runs MLflow qui s'y réfèrent.

Exemples :

    # Construire le livrable, ou vérifier qu'il est à jour
    uv run python scripts/prepare_data.py

    # Comparer sans rien écrire : le code reproduit-il le disque ?
    uv run python scripts/prepare_data.py --verifier

    # Regénérer malgré une différence constatée
    uv run python scripts/prepare_data.py --forcer
"""

from __future__ import annotations

import argparse
import sys

from agritech import pipeline
from agritech.config import CONSOLIDATED_PARQUET_PATH, CONSOLIDATED_PATH


def analyser_arguments():
    parseur = argparse.ArgumentParser(
        description="Construit le dataset consolidé à partir des six fichiers sources.")
    parseur.add_argument(
        "--verifier", action="store_true",
        help="compare le dataset régénéré à celui du disque, sans rien écrire")
    parseur.add_argument(
        "--forcer", action="store_true",
        help="réécrit le livrable même si le contenu diffère de celui du disque")
    return parseur.parse_args()


def rendre_compte_verification(rapport: dict) -> int:
    """Affiche le résultat de la comparaison et rend le code de sortie."""
    obtenues, attendues = rapport["lignes"]
    print("Contenu")
    print(f"  colonnes identiques : {rapport['colonnes_identiques']}")
    print(f"  types identiques    : {rapport['types_identiques']}")
    print(f"  lignes              : {obtenues:,} régénérées / {attendues:,} sur le disque")
    print(f"  valeurs identiques  : {rapport['contenu_identique']}")
    print()
    print("Octets")
    print(f"  data_version du disque   : {rapport['data_version_attendue']}")
    print(f"  data_version régénérée   : {rapport['data_version_obtenue']}")
    print()

    if rapport["contenu_identique"] and rapport["octets_identiques"]:
        print("Le code reproduit le livrable à l'octet près.")
        return 0

    if rapport["contenu_identique"]:
        # Cas bénin mais à connaître : pyarrow sérialise différemment d'une
        # version à l'autre. Les données sont les mêmes, l'empreinte non.
        print("Le contenu est identique, les octets non.")
        print("La sérialisation a changé — probablement une version de pyarrow")
        print("différente de celle qui a produit le fichier. Les runs MLflow")
        print("existants restent valides sur le fond, mais leurs tags")
        print("`data_version` ne correspondraient plus après réécriture.")
        return 0

    print("LE CONTENU DIFFÈRE. Ne pas écraser sans comprendre pourquoi.")
    return 1


def main() -> int:
    arguments = analyser_arguments()

    if arguments.verifier:
        print("Vérification — rien ne sera écrit\n")
        return rendre_compte_verification(pipeline.verifier_preparation())

    try:
        resultat = pipeline.lancer_preparation(forcer=arguments.forcer)
    except RuntimeError as refus:
        # Un refus est un diagnostic, pas un plantage : il porte déjà
        # l'explication et la marche à suivre.
        print(refus, file=sys.stderr)
        return 1

    if not resultat.agi:
        print(f"Préparation : {resultat.detail}")
        return 0

    print("Dataset consolidé")
    for libelle, valeur in resultat.bilan.items():
        print(f"  {libelle.replace('_', ' '):<26} {valeur:>12,}")
    print()
    for chemin in (CONSOLIDATED_PATH, CONSOLIDATED_PARQUET_PATH):
        print(f"  {chemin.name:<40} {chemin.stat().st_size / 1024 ** 2:>8,.1f} Mo")
    print()
    print(f"Préparation : {resultat.detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Point d'entrée de l'entraînement : c'est CE fichier qu'on exécute.

Différence avec `src/agritech/training.py` :

- `training.py` est une **bibliothèque**. Elle définit *comment* construire un
  Pipeline, quels hyperparamètres explorer, comment journaliser dans MLflow.
  On ne la lance jamais : on l'importe. L'API l'importera aussi, pour pouvoir
  désérialiser le modèle.
- `scripts/train.py`, ce fichier, est un **script**. Il ne définit rien : il lit
  les arguments de la ligne de commande, décide *quoi* lancer, et appelle la
  bibliothèque. On ne l'importe jamais.

C'est cette séparation qui rend l'entraînement automatisable en CI/CD : une
chaîne d'intégration appelle un script, elle n'ouvre pas un notebook.

Ce que produit une exécution complète :

- des **runs MLflow** — un parent par étude, un enfant par essai Optuna, plus
  le modèle du meilleur essai enregistré comme artefact et les partitions
  `train` / `val` déclarées en entrées du run ;
- une **version au registre** `agritech-rendement` par modèle entraîné, taguée
  avec l'algorithme et les empreintes de données. Aucune n'est promue ici ;
- un **`.joblib` par étude** dans `models/`, pour que le notebook de
  comparaison recharge les modèles sans les réajuster. Seul
  `models/modele_final.joblib` est versionné : les autres vont jusqu'à 1,3 Mo
  et se régénèrent à l'identique.

Ni `modele_final.joblib` ni l'alias `champion` du registre ne sont écrits
automatiquement : le choix du modèle à déployer est multicritère — erreur,
stabilité par culture, latence, interprétabilité — et se fait dans
`notebooks/model_experimentation.ipynb`. L'option `--export-best` existe pour
la CI, qui a besoin d'un critère unique.

Exemples :

    # Parcelles simulées : 4 modèles, les 13 variables, 100 essais chacun
    uv run python scripts/train.py

    # Panel FAO : ses 8 variables, les deux références comprises
    uv run python scripts/train.py --source real

    # Les deux chaînes d'affilée
    uv run python scripts/train.py --source simulated real

    # Test d'intégration rapide, sans rien écrire dans MLflow ni sur disque
    uv run python scripts/train.py --sample 5000 --trials 2 --no-mlflow --no-export

    # Un seul modèle
    uv run python scripts/train.py --models ridge

    # Désigner en plus un modèle final sur le seul critère du RMSE
    uv run python scripts/train.py --export-best
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from agritech.config import (
    FINAL_MODEL_PATH,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    PANEL_FINAL_MODEL_PATH,
)
from agritech.training import (
    ESTIMATEURS,
    JEUX_DE_VARIABLES,
    SOURCE_PAR_DEFAUT,
    charger_partitions,
    chemin_modele,
    exporter_joblib,
    optimiser,
)


def analyser_arguments():
    """Décrit les options acceptées en ligne de commande.

    `choices` fait valider les valeurs par argparse : une faute de frappe
    s'arrête ici avec un message clair, pas trois minutes plus tard dans une
    boucle d'entraînement.
    """
    parseur = argparse.ArgumentParser(
        description="Entraîne et optimise les modèles de prédiction de rendement.")

    parseur.add_argument(
        "--models", nargs="+", default=None, choices=list(ESTIMATEURS),
        help="modèles à étudier (défaut : les quatre, la référence `dummy` "
             "comprise)")
    parseur.add_argument(
        "--source", nargs="+", default=[SOURCE_PAR_DEFAUT],
        choices=list(JEUX_DE_VARIABLES),
        help=f"chaîne de données à entraîner (défaut : {SOURCE_PAR_DEFAUT}, les "
             "13 variables des parcelles simulées). `real` entraîne les 8 "
             "variables du panel FAO. Le même mot qu'attendent "
             "`prepare_data.py` et `split_data.py`")
    parseur.add_argument(
        "--trials", type=int, default=OPTUNA_N_TRIALS,
        help=f"essais Optuna par étude (défaut : {OPTUNA_N_TRIALS})")
    parseur.add_argument(
        "--sample", type=int, default=None,
        help="sous-échantillonne train — test d'intégration seulement, "
             "les métriques obtenues ne sont pas comparables à une étude complète")
    parseur.add_argument(
        "--no-mlflow", action="store_true",
        help="n'enregistre aucun run : sert à vérifier que le code tourne")
    parseur.add_argument(
        "--no-export", action="store_true",
        help=f"n'écrit aucun .joblib dans {MODELS_DIR.name}/ (export actif par défaut)")
    parseur.add_argument(
        "--export-best", action="store_true",
        help="désigne en plus le meilleur RMSE comme modèle final de sa "
             f"chaîne ({FINAL_MODEL_PATH.name} pour les parcelles simulées, "
             f"{PANEL_FINAL_MODEL_PATH.name} pour le panel)")

    return parseur.parse_args()


def main():
    arguments = analyser_arguments()
    suivi_mlflow = not arguments.no_mlflow
    exporter = not arguments.no_export

    print(f"MLflow : {'actif' if suivi_mlflow else 'DÉSACTIVÉ'} | "
          f"export : {'actif' if exporter else 'DÉSACTIVÉ'} | "
          f"{arguments.trials} essais par étude\n")

    # Une étude par couple (chaîne, modèle). Les partitions ne sont lues qu'une
    # fois par chaîne : relire 700 000 lignes à chaque modèle serait du
    # gaspillage.
    resultats = []
    for source in arguments.source:
        train, val = charger_partitions(echantillon=arguments.sample,
                                        source=source)
        print(f"chaîne {source} : train {len(train):,} lignes | "
              f"val {len(val):,} lignes | "
              f"{len(JEUX_DE_VARIABLES[source].features)} variables")

        for modele in arguments.models or list(ESTIMATEURS):
            debut = time.perf_counter()
            # `optimiser` renvoie aussi le Pipeline du meilleur essai, déjà
            # réajusté : on n'a donc jamais à le refaire pour l'exporter.
            etude, pipeline = optimiser(modele, source, train, val,
                                        n_essais=arguments.trials,
                                        suivi_mlflow=suivi_mlflow)
            duree = time.perf_counter() - debut

            resultats.append({
                "source": source,
                "modele": modele,
                "val_rmse": etude.best_value,
                "essais": len(etude.trials),
                "duree_s": round(duree, 1),
                "pipeline": pipeline,   # gardé pour l'export, retiré de l'affichage
            })
            print(f"{source:<10} {modele:<11} RMSE {etude.best_value:.4f}  "
                  f"({len(etude.trials)} essais, {duree:.0f} s)")

    print("\n=== Récapitulatif, trié par RMSE sur val ===")
    tableau = pd.DataFrame(resultats).drop(columns="pipeline").sort_values("val_rmse")
    print(tableau.to_string(index=False))

    if exporter:
        exporter_tous(resultats)
    if arguments.export_best:
        designer_modele_final(resultats)


def exporter_tous(resultats: list[dict]):
    """Écrit un `.joblib` par étude : `ridge_modele.joblib` pour les parcelles
    simulées, `ridge_panel_modele.joblib` pour le panel FAO.

    C'est ce qui permet au notebook de comparaison de recharger les modèles
    optimisés instantanément, plutôt que de les réajuster à chaque exécution —
    soit plusieurs minutes économisées par figure.

    Le nom du fichier vient de `artefact`, pas de la source : les deux chaînes
    cohabitent dans `models/` sans qu'un fichier puisse en écraser un autre.
    """
    print(f"\n=== Modèles exportés dans {MODELS_DIR.name}/ ===")
    for resultat in resultats:
        artefact = JEUX_DE_VARIABLES[resultat["source"]].artefact
        exporter_joblib(resultat["pipeline"],
                        chemin_modele(resultat["modele"], artefact))


def designer_modele_final(resultats: list[dict]):
    """Recopie le modèle au plus petit RMSE sous le nom `modele_final.joblib`.

    Attention : « meilleur » signifie ici **plus petit RMSE sur val**, rien de
    plus. Le vrai choix est multicritère et appartient au notebook. Cette
    option existe pour la CI, qui doit trancher sans intervention humaine.

    La référence est écartée : `dummy` ne prédit que la moyenne, il n'a rien
    appris.
    """
    candidats = [r for r in resultats if r["modele"] != "dummy"]
    if not candidats:
        print("\nAucun modèle à désigner (seule la référence a tourné).")
        return

    # Un RMSE de parcelle et un RMSE national ne se comparent pas : l'un porte
    # sur des hectares simulés autour de 4,65 t/ha, l'autre sur des moyennes
    # nationales de 1,5 à 18 t/ha. Le minimum des deux ne désignerait rien.
    chaines = {r["source"] for r in candidats}
    if len(chaines) > 1:
        print(f"\nAucun modèle désigné : l'exécution couvre {len(chaines)} "
              f"chaînes de données ({', '.join(sorted(chaines))}), dont les "
              "RMSE ne sont pas comparables. Relancer `--export-best` sur une "
              "seule chaîne.")
        return

    meilleur = min(candidats, key=lambda r: r["val_rmse"])
    destination = JEUX_DE_VARIABLES[meilleur["source"]].modele_final
    print(f"\n=== Modèle final (RMSE seul) : {meilleur['modele']} / "
          f"{meilleur['source']} — RMSE {meilleur['val_rmse']:.4f} ===")
    exporter_joblib(meilleur["pipeline"], destination)


if __name__ == "__main__":
    main()

# Agritech Answers — prédiction de rendement et recommandation de culture

Prototype MLOps : une API de prédiction du rendement agricole et de recommandation
de culture, interrogée par une interface Streamlit.

## Structure

```
src/agritech/config.py          chemins, graines, vocabulaire, hypothèses (valeurs seules)
src/agritech/training.py        prétraitement, modèles, optimisation Optuna, runs MLflow
src/agritech/evaluation.py      métriques de prédiction, économiques et de recommandation
scripts/train.py                point d'entrée de l'entraînement
models/                         modèles optimisés en .joblib (un seul versionné)
data/OC_InputData/              jeux de données sources (non versionnés)
data/interim/                   artefacts intermédiaires reconstructibles
data/processed/                 dataset consolidé (CSV livrable + Parquet de travail)
data/processed/splits/          train / val / test et leurs métadonnées
notebooks/                      exploration, préparation, découpage, expérimentation
reports/                        résumés des choix et figures
```

Le contenu de `data/` n'est pas versionné : il se reconstruit en exécutant les
notebooks dans l'ordre `exploratory_data_analysis` → `data_preparation` → `data_split`.

## Démarrage

```bash
uv sync                # crée .venv, installe les dépendances et le paquet agritech
```

Le projet est installé en mode éditable : `from agritech.config import ...`
fonctionne depuis les notebooks, les scripts et les tests, sans manipulation
de `sys.path`.

Ouvrir ensuite les notebooks dans VS Code en sélectionnant le noyau
`.venv\Scripts\python.exe`, ou lancer `uv run jupyter lab` pour travailler
dans le navigateur. Préfixer une commande par `uv run` l'exécute dans
l'environnement du projet sans avoir à l'activer.

## Suivi des expériences

MLflow 3 a placé le backend fichier (`./mlruns`) en mode maintenance : le suivi
utilise donc une base **SQLite**, `mlflow.db`, et les artefacts de modèle vont
dans `mlartifacts/`. Ni l'un ni l'autre n'est versionné.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Une seule expérience, `agritech-rendement` : les quatre modèles sur les **13
variables explicatives**. Chaque étude d'optimisation est un run parent, chaque
essai Optuna un run enfant. Seul le meilleur essai de chaque étude sérialise un
modèle — un artefact par essai ferait gonfler le stockage de plusieurs
gigaoctets sans utilité.

Les essais enfants étant cent fois plus nombreux que les parents, la liste des
runs est vite saturée. Pour n'afficher que les quatre études :

```
tags.phase = 'optimisation'
```

et `tags.phase != ''` y ajoute le run d'évaluation finale.

### Jeux de données attachés aux runs

Chaque run déclare les partitions qui l'ont produit : `train` et `val` pour une
étude d'optimisation, `val` et `test` pour l'évaluation finale. MLflow n'en
stocke **pas les données**, seulement le schéma typé, le nombre de lignes, une
empreinte du contenu et le chemin du fichier.

Cette empreinte porte sur le contenu ; les tags `data_version` et
`split_version` portent sur les fichiers. Les deux se complètent — la première
détecte qu'un contenu a changé, les seconds disent de quelle génération de
dataset le run provient.

### Registre des modèles

`train.py` publie chaque modèle entraîné comme une **version** du modèle
enregistré `agritech-rendement`, taguée avec l'algorithme et les empreintes de
données. Aucune n'est privilégiée : le choix est multicritère et appartient au
notebook de comparaison, qui pose l'alias en appelant
`training.promouvoir_champion`.

L'API chargera ensuite le modèle sans rien savoir de son identité :

```python
mlflow.sklearn.load_model("models:/agritech-rendement@champion")
```

Changer de modèle en production revient donc à déplacer cet alias, sans toucher
au code de l'API ni la redéployer. MLflow 3 ayant supprimé les *stages*
(`Staging`, `Production`) au profit des alias, c'est `champion` qui tient ce
rôle.

### Modèles exportés

`uv run python scripts/train.py` écrit aussi un `.joblib` par étude dans
`models/`, nommé `{modèle}_{jeu_de_variables}.joblib`, pour que le notebook de
comparaison recharge les modèles sans les réajuster.

Seul `models/modele_final.joblib` est versionné — les autres vont jusqu'à 1,3 Mo
et se régénèrent à l'identique. Il n'est pas désigné automatiquement : le choix
appartient à `notebooks/model_experimentation.ipynb`. L'option `--export-best`
tranche sur le seul RMSE, pour les besoins de la CI.

## Rejouer les résultats à l'identique

`data/`, `mlflow.db` et `mlartifacts/` ne sont pas versionnés : un clone neuf
repart des CSV sources et reconstruit tout.

```bash
uv sync                                   # versions exactes figées par uv.lock
# 1. notebooks/exploratory_data_analysis.ipynb
# 2. notebooks/data_preparation.ipynb      -> data/processed/
# 3. notebooks/data_split.ipynb            -> data/processed/splits/
uv run python scripts/train.py            # ~30 min, 4 études × 100 essais
# 4. notebooks/model_experimentation.ipynb -> figures, modèle final, alias
```

Ce qui garantit l'identité des résultats : `uv.lock` fige les versions, les
graines sont fixées à 42 partout (`SPLIT_RANDOM_STATE` pour le découpage,
`RANDOM_STATE` pour l'échantillonneur TPE et les estimateurs), et
`training.ESPACES` est la **source unique** des bornes de recherche — consommée
à l'entraînement, relue par le notebook pour reconstituer les études Optuna
depuis MLflow.

Les empreintes `data_version` et `split_version`, portées par chaque run, sont
le contrôle : si elles ne correspondent plus à `split_metadata.json`, le run ne
décrit pas les données courantes.

## Jeux de données

- **Agriculture Crop Yield** — 1 000 000 parcelles simulées, cible `Yield_tons_per_hectare`.
- **Crop Yield Prediction** — données FAO pays × année × culture, plus ses quatre fichiers sources.

## Jeux d'entraînement, de validation et de test

`data/processed/splits/` contient `train.parquet`, `val.parquet` et `test.parquet`,
issus du dataset consolidé produit par `notebooks/data_preparation.ipynb`.
Découpage 70 / 15 / 15, `random_state=42`, stratifié par culture.

### Gestion des partitions

- Les trois fichiers sont générés **une seule fois** par `notebooks/data_split.ipynb`,
  puis conservés pour toutes les expérimentations. Le notebook d'entraînement les lit
  sans jamais les régénérer.
- Toute modification du dataset consolidé impose de **régénérer les partitions** et de
  **relancer l'entraînement et l'évaluation**.
- Chaque run MLflow porte deux tags : `data_version` (empreinte du dataset consolidé)
  et `split_version` (empreinte des partitions). Des métriques issues de versions
  différentes ne sont pas directement comparables sans vérifier que le protocole
  d'évaluation est identique.

### Prévention des fuites

- Les transformations qui s'ajustent sur les données — imputation, encodage, mise à
  l'échelle, sélection de variables — sont apprises **sur `train` seul**.
- `val` sert à comparer les modèles et à optimiser les hyperparamètres.
- `test` est réservé à l'**évaluation finale du modèle retenu**. Il peut être relu
  autant que nécessaire pour reproduire un résultat, mais il ne doit intervenir dans
  **aucune décision** de modélisation : modifier le modèle après avoir vu ses
  performances sur `test` ferait perdre à cette mesure son indépendance.

# Agritech Answers — prédiction de rendement et recommandation de culture

[![CI/CD](https://github.com/FaizaKh93/Projet12_mission/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/FaizaKh93/Projet12_mission/actions/workflows/ci_cd.yml)

Prototype MLOps : une API de prédiction du rendement agricole et de recommandation
de culture, interrogée par une interface Streamlit.

## Structure

```
src/agritech/config.py          chemins, graines, vocabulaire, hypothèses (valeurs seules)
src/agritech/training.py        prétraitement, modèles, optimisation Optuna, runs MLflow
src/agritech/evaluation.py      métriques de prédiction, économiques et de recommandation
src/agritech/preprocessing.py   reconstruction des 13 variables à partir des 8 saisies
src/agritech/preparation.py     construction du dataset consolidé, en fonctions pures
src/agritech/decoupage.py       découpage train/val/test et ses garde-fous
src/agritech/pipeline.py        orchestration des quatre étapes et leurs empreintes
api/                            service FastAPI, schémas Pydantic, journal des appels
app/app.py                      interface Streamlit, qui interroge l'API en HTTP
Dockerfile                      image de service : ni MLflow, ni données, ni uv
scripts/prepare_data.py         point d'entrée — préparation
scripts/split_data.py           point d'entrée — découpage
scripts/run_pipeline.py         point d'entrée — la chaîne entière
scripts/train.py                point d'entrée de l'entraînement
scripts/smoke_docker.py         épreuve d'un conteneur réel (bibliothèque standard seule)
tests/                          suite pytest : API, interface, parité du prétraitement
.github/workflows/ci_cd.yml     chaîne d'intégration et de livraison continues
models/                         modèles optimisés en .joblib (un seul versionné)
data/OC_InputData/              jeux de données sources (non versionnés)
data/interim/                   artefacts intermédiaires reconstructibles
data/processed/                 dataset consolidé (CSV livrable + Parquet de travail)
data/processed/splits/          train / val / test et leurs métadonnées
notebooks/                      exploration, préparation, découpage, expérimentation
reports/                        rapports destinés aux agriculteurs, et leurs figures
```

Deux publics, deux endroits : `reports/` s'adresse aux **agriculteurs** —
tableaux, figures, aucun jargon. Tout ce qui relève du **développement**,
chaîne CI/CD comprise, se documente dans ce README.

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

## Chaîne MLOps

```
préparation  →  découpage  →  entraînement  →  export
   data/          splits/       mlflow.db      models/modele_api.joblib
```

```bash
uv run python scripts/run_pipeline.py                      # préparation + découpage
uv run python scripts/run_pipeline.py --jusqua export      # tout, entraînement compris
```

### Où vit le code, et pourquoi

La transformation des données était écrite dans les notebooks. Un notebook ne se
rejoue pas : il s'ouvre, se fait dérouler cellule après cellule, et rien ne
garantit l'ordre ni l'état des variables. Le livrable n'était donc reproductible
que par un humain attentif.

La logique vit maintenant dans trois modules de `src/agritech/`, en fonctions
**pures** — elles prennent des tables, en rendent d'autres, n'écrivent rien et
n'affichent rien. Les notebooks les appellent pour commenter leurs sorties ; les
scripts les appellent pour produire le livrable. Les deux chemins traversent le
même code, donc ne peuvent pas diverger.

Les notebooks gardent tout le reste : contrôles qualité, validation hors
échantillon, analyses de redondance, figures. C'est leur rôle — expliquer et
justifier, pas exécuter.

**Un seul producteur.** Les notebooks n'écrivent **aucun jeu de données** — ni
CSV, ni Parquet, ni artefact intermédiaire, pas même temporaire. Ils calculent
en mémoire, commentent, puis confrontent leur résultat aux fichiers que les
scripts ont produits. Aucune ambiguïté n'est donc possible sur le fichier qui
fait foi, et un « Run All » ne peut pas réécrire `data/` par surprise. Les
seules écritures d'un notebook vont dans `reports/figures/` : ce sont les
figures du rapport métier, pas des données.

### Les deux empreintes

| | Porte sur | Répond à |
|---|---|---|
| `data_version` | le contenu du dataset consolidé | de quel **dataset** ce run provient-il ? |
| `split_version` | les octets des trois partitions | de quels **fichiers** l'entraînement a-t-il lu ? |

Chaque run MLflow et la provenance de l'artefact déployé les portent toutes les
deux. Deux runs d'empreintes différentes ne sont pas comparables — et c'est ce
que la chaîne vérifie avant d'agir.

### Ce que la chaîne refuse de faire

**Écraser un dataset différent.** `prepare_data.py` régénère puis compare avant
d'écrire. Si le contenu diffère de celui du disque, il s'arrête : réécrire
changerait `data_version` et périmerait tous les runs existants.

**Réécrire des partitions.** Elles sont produites une fois, puis relues par
chaque entraînement. Le script refuse dans trois cas — lot incomplet, vestige
d'une exécution interrompue dont les fichiers peuvent provenir de découpages
différents ; partitions périmées, générées pour un autre dataset ; et lot
complet déjà à jour, où il vérifie sans toucher.

**Entraîner sans qu'on le lui demande.** Une demi-heure de calcul et des
écritures dans le registre ne doivent pas être la conséquence d'une commande
tapée distraitement. Par défaut la chaîne s'arrête au découpage.

`--forcer` lève ces refus, un par un, pour les régénérations voulues.

### Vérifier sans rien écrire

```bash
uv run python scripts/prepare_data.py --verifier
```

Régénère le dataset dans un dossier temporaire et le compare à celui du disque,
sur deux plans distincts : le **contenu** — colonnes, types, valeurs — et les
**octets**. Les deux peuvent diverger : une autre version de pyarrow sérialise
le même tableau différemment, auquel cas les données sont identiques mais
`data_version` change. Il vaut mieux le savoir avant d'écraser quoi que ce soit.

C'est ce contrôle qui a autorisé l'extraction : la chaîne extraite reproduit le
dataset livré à l'octet près, `data_version c9547cbd218b`, celui-là même que
cite la provenance de l'artefact servi par l'API.

### Ce qui est éprouvé en intégration continue

`tests/test_pipeline.py` fait tourner la chaîne sur six tables minuscules
construites dans le test : rendement négatif retiré, cible manquante conservée,
pluie illisible transformée en absence, cardinalité validée avant jointure,
`NaN` préservés pour les cultures sans référence FAO, ordre des colonnes du
livrable, et les refus du découpage. Aucune donnée n'est nécessaire.

Ce que la CI ne peut pas porter, c'est « le pipeline reproduit le dataset
livré » — il demande le million de lignes. C'est `--verifier` qui s'en charge,
sur un poste qui a les données.

## Intégration et livraison continues

[`.github/workflows/ci_cd.yml`](.github/workflows/ci_cd.yml) — une chaîne, deux
jobs, déclenchée sur **toute poussée** quelle que soit la branche, et à
l'**ouverture** d'une proposition de fusion.

```
CI · Contrôles  ──→  CI · Image et conteneur → CD · Publication
```

### Ce que chaque job prouve

**`CI · Contrôles`** installe l'environnement figé par `uv.lock`, lance la suite
de tests **sous Linux** — alors que le développement se fait sous Windows —
avec un plancher de couverture à 90 %, puis vérifie que les deux
`requirements.txt` livrés décrivent encore le lock. Ces fichiers sont exigés par
le cahier des charges, mais ils sont **dérivés** : `uv.lock` reste la seule
source de vérité, et toute divergence fait échouer la chaîne.

**`CI · Image et conteneur → CD · Publication`** construit l'image, la charge
dans le démon du runner et l'éprouve avec `scripts/smoke_docker.py` : un
conteneur réel démarre, ses routes sont interrogées en HTTP, l'absence de MLflow
et de PyArrow est vérifiée, et un artefact falsifié doit **empêcher** le
démarrage.

La livraison ne s'exécute qu'après une poussée sur `main` :

```yaml
if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

Une proposition de fusion construit donc son image et l'éprouve, mais ne touche
ni aux identifiants ni au registre. L'image publiée n'est **jamais
reconstruite** : `docker tag` renomme celle qui vient de passer les tests, de
sorte que les octets publiés sont exactement les octets éprouvés. Deux
étiquettes partent : `latest`, qui suit la branche, et le SHA du commit, qui
rattache chaque image à son code.

### Ce qui n'est pas vérifié en intégration continue

Dix tests sont ignorés sur un runner. Le résumé de chaque exécution les nomme,
avec leur motif.

| Tests | Pourquoi |
|---|---|
| 8 tests de parité (`test_preprocessing.py`) | ils comparent la reconstruction aux colonnes que le dataset porte lui-même ; `data/` n'est pas versionné |
| 1 contrôle des bornes du domaine (`test_api.py`) | il confronte `config.py` aux minima et maxima réellement observés dans `train` |
| 1 parité avec le registre (`test_api.py`) | `mlflow.db` n'existe pas sur un runner |

Ce ne sont **pas** des tests désactivés. Ils s'exécutent intégralement sur un
poste qui possède les données, et c'est là qu'ils doivent passer avant toute
fusion. Fabriquer un jeu de données pour les faire tourner en CI les rendrait
tautologiques : inventer la référence, puis vérifier qu'on la retrouve.

Tout le reste du module de prétraitement s'éprouve bien en CI, sur l'**artefact
déployé** — `models/modele_api.joblib` embarque la table FAO ajustée à
l'entraînement, donc les règles de dérivation se vérifient sans aucune donnée,
et sur la pièce effectivement livrée.

### Reproduire les contrôles localement

```bash
# ce que lance la CI, à l'identique
uv run pytest -q -rfEs --cov --cov-report=term-missing --cov-fail-under=90

# régénérer les requirements après un changement de dépendance
uv export --no-default-groups --group api --no-emit-project --no-hashes --format requirements.txt -o requirements.txt
uv export --no-default-groups --group ui  --no-emit-project --no-hashes --format requirements.txt -o app/requirements.txt

# l'image et son épreuve
docker build -t agritech-api:ci .
uv run python scripts/smoke_docker.py --image agritech-api:ci
```

### Ce que le dépôt doit fournir

| Nom | Où | Valeur |
|---|---|---|
| `DOCKERHUB_USERNAME` | Settings → Secrets and variables → Actions → **Variables** | le nom de compte Docker Hub |
| `DOCKERHUB_TOKEN` | idem → **Secrets** | un jeton dédié à la CI, niveau **Repo Read & Write** |

Un jeton Docker Hub personnel **ne peut pas être restreint à un seul dépôt** :
sa permission vaut pour tout le compte. D'où l'intérêt d'un jeton dédié,
révocable sans toucher à l'usage quotidien, et du niveau minimal — jamais
*Repo Read, Write, Delete*.

### Versions des actions

Toutes les actions sont épinglées sur un **SHA de commit**, leur numéro de
version en commentaire. Une étiquette, même `@v7`, peut être déplacée par son
mainteneur ; un SHA, non. Cela compte surtout pour `docker/login-action`, le
seul pas qui manipule le jeton.

[`.github/dependabot.yml`](.github/dependabot.yml) les fait vieillir sous
surveillance, chaque semaine, pour le **seul** écosystème `github-actions`. Y
déclarer Python ouvrirait des propositions qui modifient `uv.lock` sans toucher
aux `requirements.txt` exportés : chacune échouerait sur le contrôle de dérive.
Les mises à jour de *sécurité* Python, elles, arrivent sans configuration.

### État de la chaîne au 29 octobre 2026

Les deux moitiés sont **éprouvées en conditions réelles**. L'intégration
continue — tests, couverture, dépendances, construction de l'image et épreuve du
conteneur — a tourné sur un runner. La livraison continue a publié son image sur
Docker Hub à la fusion de la chaîne dans `main`, sous deux étiquettes : `latest`
et le SHA du commit, de tailles identiques à l'octet, preuve qu'aucune
reconstruction n'a eu lieu entre l'épreuve et la publication.

Les pas de publication se sont par ailleurs correctement abstenus sur chaque
branche de travail, où seuls la construction et le test de fumée s'exécutent.

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

MLflow 3 ayant supprimé les *stages* (`Staging`, `Production`) au profit des
alias, c'est `champion` qui tient ce rôle.

**Déplacer l'alias ne déploie rien.** L'API sert `models/modele_api.joblib`,
embarqué dans son image et scellé par une empreinte SHA-256 vérifiée au
démarrage du conteneur. Le registre n'est une source de chargement que si
`AGRITECH_MODEL_URI` est posée, et il est alors résolu **une seule fois au
démarrage**, sur un *second* modèle enregistré — `agritech-rendement-api`,
publié par `scripts/export_api_model.py`.

Changer le modèle servi demande donc quatre temps :

1. promouvoir l'alias depuis le notebook de comparaison ;
2. `uv run python scripts/export_api_model.py`, qui réexporte l'artefact, sa
   provenance et son empreinte ;
3. committer ce nouvel artefact ;
4. laisser la chaîne CI/CD construire et publier l'image.

La chaîne GitHub Actions **n'interroge jamais MLflow** : elle emballe l'artefact
versionné, elle ne l'exporte pas. Son test de fumée vérifie d'ailleurs l'absence
de MLflow dans l'image. C'est ce qui rend chaque image traçable à un commit.

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
uv sync                                                 # versions figées par uv.lock
uv run python scripts/run_pipeline.py --jusqua export   # ~30 min, 4 études × 100 essais
# puis notebooks/model_experimentation.ipynb            -> figures, modèle final, alias
```

Les notebooks d'exploration et de préparation restent utiles pour *comprendre*
ce que la chaîne fait, mais ils ne sont plus nécessaires pour la rejouer.

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

# Agritech Answers — prédiction de rendement et recommandation de culture

[![CI/CD](https://github.com/FaizaKh93/Projet12_mission/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/FaizaKh93/Projet12_mission/actions/workflows/ci_cd.yml)

Agritech Answers répond à deux questions qu'un agriculteur se pose avant de semer : **quel rendement attendre de cette parcelle**, et **comment les six cultures se comparent une fois les charges déduites**. Huit informations suffisent — pluviométrie, température, fertilisation, irrigation, culture envisagée, plus trois caractéristiques facultatives de la parcelle.

La seconde réponse est une **marge brute simulée** — rendement estimé × prix de référence − charges saisies par l'utilisateur. C'est un ordre de grandeur pour comparer, pas une recommandation agronomique validée.

Le modèle est servi par une API FastAPI, interrogée par une interface Streamlit. Le tout est reproductible de bout en bout : un clone neuf rejoue la préparation des données, l'entraînement et l'export par une suite de commandes, et une chaîne d'intégration continue éprouve l'ensemble à chaque poussée.

**Dépôt** — <https://github.com/FaizaKh93/Projet12_mission>

## Sommaire

1. [Structure du projet](#1-structure-du-projet)
2. [Installation et démarrage](#2-installation-et-démarrage)
3. [Le pipeline MLOps](#3-le-pipeline-mlops)
4. [Le pipeline CI/CD](#4-le-pipeline-cicd)
5. [Suivi des expériences et registre MLflow](#5-suivi-des-expériences-et-registre-mlflow)
6. [Fonctionnement de l'application](#6-fonctionnement-de-lapplication)
7. [Limites et perspectives](#7-limites-et-perspectives)

---

## 1. Structure du projet

```
src/agritech/      la bibliothèque du projet — importée, jamais exécutée
api/               service FastAPI
app/               interface Streamlit
scripts/           points d'entrée exécutables
tests/             suite pytest
notebooks/         exploration et analyse
data/              jeux de données — non versionnés
models/            artefacts de modèle
reports/           rapports d'étape et figures
.github/           chaîne CI/CD et configuration Dependabot
Dockerfile         image de service
pyproject.toml     dépendances, groupes, couverture
uv.lock            versions exactes — la seule source de vérité
```

### Les composants et leurs responsabilités

| Chemin | Rôle |
|---|---|
| **Bibliothèque** | |
| `src/agritech/config.py` | chemins, graines, vocabulaire, hypothèses — des valeurs, aucune fonction |
| `src/agritech/preparation.py` | construction du dataset consolidé, en fonctions pures |
| `src/agritech/decoupage.py` | découpage train/val/test et ses garde-fous |
| `src/agritech/pipeline.py` | orchestration des quatre étapes et de leurs empreintes |
| `src/agritech/preprocessing.py` | reconstruction des 13 variables du modèle à partir des 8 saisies |
| `src/agritech/training.py` | modèles, optimisation Optuna, journalisation MLflow |
| `src/agritech/evaluation.py` | métriques de prédiction, économiques et de recommandation |
| **Service et interface** | |
| `api/main.py` · `api/schemas.py` | les six routes et leurs contrats Pydantic |
| `api/service.py` | chargement du modèle, vérification de son empreinte, prédiction |
| `api/journal.py` | journal des appels, servi par `/logs` |
| `app/app.py` | interface Streamlit, qui interroge l'API en HTTP |
| `Dockerfile` | image de service : ni MLflow, ni données, ni uv |
| **Points d'entrée** | |
| `scripts/run_pipeline.py` | la chaîne entière, d'une étape à une autre |
| `scripts/prepare_data.py` | préparation — produit `data/processed/` |
| `scripts/split_data.py` | découpage — produit `data/processed/splits/` |
| `scripts/train.py` | entraînement — 4 études × 100 essais Optuna |
| `scripts/export_api_model.py` | export de l'artefact servi et de sa provenance |
| `scripts/smoke_docker.py` | épreuve d'un conteneur réel, bibliothèque standard seule |
| `scripts/build_report_figures.py` | figures des rapports |
| **Données** | |
| `data/OC_InputData/` | les six fichiers sources |
| `data/interim/` | panel FAO, référentiel par culture, dictionnaire des variables |
| `data/processed/` | dataset consolidé — CSV livrable et Parquet de travail |
| `data/processed/splits/` | train / val / test et leurs métadonnées |
| **Modèles** | |
| `models/modele_final.joblib` | le modèle retenu à l'issue de la comparaison |
| `models/modele_api.joblib` | l'artefact servi par l'API, prétraitement compris |
| `models/modele_api.json` | sa provenance : registre, version, empreintes, versions des bibliothèques |
| **Documentation** | |
| [`reports/data_preparation_summary.md`](reports/data_preparation_summary.md) | construction du dataset consolidé |
| [`reports/model_experimentation_summary.md`](reports/model_experimentation_summary.md) | comparaison des modèles et choix du Ridge |
| [`reports/prediction_api_summary.md`](reports/prediction_api_summary.md) | conception de l'API |
| [`reports/streamlit_app_summary.md`](reports/streamlit_app_summary.md) | conception de l'interface |

**Deux endroits, deux usages.** `reports/` s'adresse d'abord au lecteur métier — tableaux, figures, aucun jargon. Ce README porte ce qui relève du développement. Et `data/` n'est pas versionné : il se reconstruit par le pipeline, jamais à la main.

---

## 2. Installation et démarrage

### Prérequis

Python 3.12 et [uv](https://docs.astral.sh/uv/). Docker Desktop si vous voulez construire ou lancer l'image. Les jeux de données sources ne sont pas dans le dépôt : il faut les placer dans `data/OC_InputData/` avant de lancer la préparation.

### Cloner et installer

```bash
git clone https://github.com/FaizaKh93/Projet12_mission.git
cd Projet12_mission
uv sync          # crée .venv, installe les dépendances et le paquet agritech
```

Le projet est installé en mode éditable : `from agritech.config import ...` fonctionne depuis les notebooks, les scripts et les tests, sans manipulation de `sys.path`. Préfixer une commande par `uv run` l'exécute dans l'environnement du projet sans avoir à l'activer.

Les deux pièces se lancent séparément, dans deux terminaux, et dans cet ordre : l'interface ne sait rien faire sans le service.

### Lancer l'API — FastAPI

Le service porte le modèle. Il expose six routes HTTP, vérifie l'empreinte de l'artefact puis le charge au démarrage, et n'a **aucune interface** : il répond en JSON. C'est lui qui calcule les rendements, les intervalles et les marges.

```bash
uv run uvicorn api.main:app --port 8000
```

Il répond alors sur `http://localhost:8000`. Deux adresses valent le détour :

- `http://localhost:8000/docs` — la documentation interactive, générée depuis les schémas Pydantic, qui permet d'essayer chaque route depuis le navigateur ;
- `http://localhost:8000/health` — l'état du service et la provenance du modèle chargé, pour vérifier d'un coup d'œil qu'il a bien démarré.

L'option `--reload` recharge le service à chaque modification du code, pratique en développement.

### Lancer l'interface — Streamlit

L'interface est ce que voit l'agriculteur. Elle n'importe ni FastAPI, ni scikit-learn, ni le modèle : elle interroge le service **en HTTP** et met en forme ses réponses. Deux modes — estimer le rendement d'une parcelle, ou comparer les six cultures sur leur rendement et leur marge brute.

Dans un **second terminal**, le service devant déjà tourner :

```bash
uv run streamlit run app/app.py
```

Elle s'ouvre sur `http://localhost:8501`. Si le service écoute ailleurs — autre port, autre machine, conteneur —, il faut le lui indiquer :

```powershell
$env:AGRITECH_API_URL = "http://localhost:8080"   # PowerShell
uv run streamlit run app/app.py
```

Sans service joignable, l'interface affiche l'adresse qu'elle a tentée et la commande pour démarrer le conteneur, plutôt qu'une page vide.

### Partir de l'image publiée

La chaîne de livraison publie l'image sur Docker Hub à chaque fusion dans `main`. Pour s'en servir sans cloner le dépôt :

```bash
docker pull faiza7/agritech-api:latest
docker run -d --name agritech-api -p 8000:8000 faiza7/agritech-api:latest
```

L'image pèse environ 146 Mo et embarque son modèle : ni données, ni MLflow, ni uv. Deux étiquettes sont publiées — `latest`, qui suit `main`, et le SHA du commit, qui rattache chaque image à son code.

### Disponibilité et hébergement

**Aucune instance publique n'est hébergée.** Un agriculteur ne peut donc pas, aujourd'hui, utiliser l'outil sans installer quelque chose. Les deux parcours ci-dessus supposent une machine avec Python ou Docker. L'hébergement du service et de l'interface reste à faire.

### Variables d'environnement et secrets

Le projet lit exactement **deux variables**, et aucune n'est un secret.

| Variable | Lue par | Défaut | Rôle |
|---|---|---|---|
| `AGRITECH_API_URL` | `app/app.py` | `http://localhost:8000` | adresse du service, indispensable si l'interface et l'API ne sont pas sur la même machine |
| `AGRITECH_MODEL_URI` | `api/service.py` | absente | bascule du joblib embarqué vers le registre MLflow — `models:/agritech-rendement-api@champion` |

Deux réglages sont en revanche nécessaires **côté GitHub**, pour que la livraison puisse publier :

| Nom | Où | Valeur |
|---|---|---|
| `DOCKERHUB_USERNAME` | Settings → Secrets and variables → Actions → **Variables** | le nom de compte Docker Hub |
| `DOCKERHUB_TOKEN` | idem → **Secrets** | un jeton dédié à la CI, niveau **Repo Read & Write** |

Un jeton Docker Hub personnel ne peut pas être restreint à un seul dépôt : sa permission vaut pour tout le compte. D'où l'intérêt d'un jeton dédié, révocable sans toucher à l'usage quotidien, et du niveau minimal — jamais *Repo Read, Write, Delete*.

---

## 3. Le pipeline MLOps

```
préparation  →  découpage  →  entraînement  →  export
   data/          splits/       mlflow.db      models/modele_api.joblib
```

```bash
uv run python scripts/run_pipeline.py                        # préparation + découpage
uv run python scripts/run_pipeline.py --jusqua entrainement  # + les 4 études Optuna
uv run python scripts/run_pipeline.py --depuis export        # après le choix du champion
```

La chaîne ne va pas d'un trait jusqu'au bout, et c'est voulu : entre l'entraînement et l'export s'intercale une **décision humaine**, celle du modèle à déployer.

### Les deux jeux de données

- **Agriculture Crop Yield** — 1 000 000 de parcelles simulées, cible `Yield_tons_per_hectare`.
- **Crop Yield Prediction** — données FAO par pays × année × culture, et ses quatre fichiers sources.

Les deux ne se joignent pas : le premier décrit des parcelles sans localisation, le second des moyennes nationales annuelles. Le second sert donc de **référentiel par culture**, pas de source de jointure. Le raisonnement est détaillé dans [`reports/data_preparation_summary.md`](reports/data_preparation_summary.md).

### Préparation et enrichissement

`scripts/prepare_data.py` nettoie les six fichiers, construit le panel FAO, en dérive un référentiel par culture, puis enrichit chaque parcelle. Il produit **999 769 lignes × 14 colonnes** — 13 variables explicatives et la cible.

L'orge et le coton n'ont pas d'équivalent dans le référentiel FAO. Leurs indicateurs restent `NaN` plutôt que de recevoir une valeur fabriquée, et la variable `crop_in_fao_reference` rend cette absence lisible par le modèle.

### Les 13 variables du modèle

| Famille | Variables |
|---|---|
| Quantitatives | `Rainfall_mm`, `Temperature_Celsius` |
| Booléennes | `Fertilizer_Used`, `Irrigation_Used`, `low_rain_no_irrigation` |
| Catégorielles | `Region`, `Soil_Type`, `Crop`, `Weather_Condition` |
| Référentiel FAO | `crop_productivity_index`, `crop_yield_geo_cv`, `crop_yield_trend_decade`, `crop_in_fao_reference` |

L'agriculteur n'en saisit que **huit** : pluviométrie, température, fertilisation, irrigation et culture envisagée, plus trois caractéristiques facultatives — région, type de sol et condition météo. Les cinq autres — `low_rain_no_irrigation` et les quatre indicateurs FAO — sont *calculées* par `agritech.preprocessing`, qui part avec le modèle dans l'artefact. Les exposer permettrait de décrire une parcelle impossible — 800 mm de pluie, irriguée, et pourtant en situation de faible pluviométrie sans irrigation.

### Découpage et prévention des fuites

70 / 15 / 15, stratifié par culture, graine fixée à 42 : **699 837** lignes d'entraînement, **149 966** de validation, **149 966** de test.

- Les transformations qui s'ajustent sur les données — imputation, encodage, mise à l'échelle — sont apprises **sur `train` seul**.
- `val` sert à comparer les modèles et à optimiser les hyperparamètres.
- `test` est réservé à l'évaluation finale. Il peut être relu pour reproduire un résultat, mais ne doit intervenir dans **aucune décision** de modélisation : modifier le modèle après avoir vu ses performances sur `test` ferait perdre à cette mesure son indépendance.

### Entraînement, évaluation et sélection

`scripts/train.py` lance quatre études — `dummy`, `ridge`, `histgbm`, `lightgbm` — de 100 essais Optuna chacune. Chaque étude est un run parent, chaque essai un run enfant, et seul le meilleur essai sérialise un modèle.

Aucune version n'est privilégiée automatiquement. Le choix est **multicritère et humain** : erreur, stabilité par culture, latence, interprétabilité. Il se fait dans `notebooks/model_experimentation.ipynb`, dont les opérations d'écriture sont protégées par un verrou `PUBLIER = False` — un simple *Run All* de consultation ne peut ni journaliser, ni exporter, ni promouvoir.

Le modèle retenu est un **Ridge**. Sur le jeu de test, qui n'a servi à aucune décision :

| | `val` | `test` |
|---|---|---|
| RMSE (t/ha) | 0,5004 | **0,5013** |
| MAE (t/ha) | 0,3993 | 0,3998 |
| R² | 0,9132 | **0,9121** |
| MAE (€/ha, à 200 €/t) | 79,87 | 79,96 |

Le run d'évaluation finale enregistre un écart val/test sur le RMSE de **0,0008 t/ha** — calculé sur les valeurs non arrondies, d'où le léger décalage avec les deux chiffres du tableau. Cette proximité ne révèle pas de dégradation de la généralisation sur le jeu tenu à l'écart. Elle ne démontre pas pour autant l'absence de tout surapprentissage à la validation, qui a servi à optimiser les hyperparamètres. Le détail de la comparaison est dans [`reports/model_experimentation_summary.md`](reports/model_experimentation_summary.md).

### Séparation des responsabilités : scripts et notebooks

La logique de transformation vit dans `src/agritech/`, en fonctions **pures** : elles prennent des tables, en rendent d'autres, n'écrivent rien et n'affichent rien. Les scripts les appellent pour produire le livrable, les notebooks pour commenter leurs sorties. Les deux chemins traversent le même code, donc ne peuvent pas diverger.

**Un seul producteur.** Les notebooks n'écrivent **aucun jeu de données** — ni CSV, ni Parquet, ni artefact intermédiaire, pas même temporaire. Ils calculent en mémoire, puis confrontent leur résultat aux fichiers que les scripts ont produits. En usage courant, leurs seules écritures vont dans `reports/figures/` : ce sont des figures, pas des données.

Une exception, et elle est explicite. `model_experimentation.ipynb` porte la décision du champion : journaliser l'évaluation finale, exporter `modele_final.joblib`, déplacer l'alias. Ces trois opérations sont protégées par `PUBLIER = False` et ne s'exécutent que si l'on bascule ce verrou — une décision, jamais un effet de bord.

### `data_version` et `split_version`

| | Porte sur | Répond à |
|---|---|---|
| `data_version` | le contenu du dataset consolidé | de quel **dataset** ce run provient-il ? |
| `split_version` | les octets des trois partitions | de quels **fichiers** l'entraînement a-t-il lu ? |

Chaque run MLflow et la provenance de l'artefact déployé portent les deux. Deux runs d'empreintes différentes ne sont pas comparables — et c'est ce que la chaîne vérifie avant d'agir.

### Ce que la chaîne refuse de faire

**Écraser un dataset différent.** `prepare_data.py` régénère puis compare avant d'écrire. Si le contenu diffère de celui du disque, il s'arrête : réécrire changerait `data_version` et périmerait tous les runs existants.

**Reconstruire un lot partiel.** Les cinq fichiers de la préparation forment un ensemble. S'il en manque, les autres peuvent décrire un autre dataset.

**Réécrire des partitions.** Elles sont produites une fois, puis relues par chaque entraînement. Le script refuse devant un lot incomplet — vestige d'une exécution interrompue — comme devant des partitions périmées, générées pour un autre dataset.

**Entraîner sans qu'on le lui demande.** Une demi-heure de calcul et des écritures au registre ne doivent pas être la conséquence d'une commande tapée distraitement. Par défaut la chaîne s'arrête au découpage.

`--forcer` lève ces refus, un par un, pour les régénérations voulues.

### Vérifier sans rien écrire

```bash
uv run python scripts/prepare_data.py --verifier
```

Régénère le dataset dans un dossier temporaire et le compare à celui du disque, sur deux plans distincts : le **contenu** — colonnes, types, valeurs — et les **octets**. Les deux peuvent diverger : une autre version de pyarrow sérialise le même tableau différemment, auquel cas les données sont identiques mais `data_version` change. Mieux vaut le savoir avant d'écraser quoi que ce soit.

### Rejouer le projet depuis un clone neuf

`export_api_model.py` lit l'alias `champion` au registre et **s'arrête s'il n'en trouve pas**, en renvoyant vers le notebook de comparaison. Sur un clone neuf, où l'entraînement publie des versions sans en privilégier aucune, l'ordre est donc le suivant :

```bash
uv sync                                                      # versions figées par uv.lock
uv run python scripts/run_pipeline.py --jusqua entrainement  # ~30 min, 4 études × 100 essais
# notebooks/model_experimentation.ipynb : comparer, choisir, puis PUBLIER = True
uv run python scripts/run_pipeline.py --depuis export        # artefact servi et provenance
```

Ce qui garantit l'identité des résultats : `uv.lock` fige les versions, les graines sont fixées à 42 partout, et `training.ESPACES` est la **source unique** des bornes de recherche — consommée à l'entraînement, relue par le notebook pour reconstituer les études Optuna depuis MLflow.

---

## 4. Le pipeline CI/CD

[`.github/workflows/ci_cd.yml`](.github/workflows/ci_cd.yml) — une chaîne, deux jobs.

```
CI · Contrôles  ──→  CI · Image et conteneur → CD · Publication
```

### Déclencheurs

Toute poussée, sur n'importe quelle branche, et les événements d'ouverture d'une proposition de fusion. Les deux se complètent : `push` éprouve la branche, `pull_request` éprouve `refs/pull/N/merge`, c'est-à-dire le **résultat de la fusion** dans `main`. Deux objets distincts, deux vérifications.

### Les deux jobs

**`CI · Contrôles`** installe l'environnement figé par `uv.lock`, lance la suite de tests **sous Linux** — alors que le développement se fait sous Windows — avec un plancher de couverture à 90 %, puis vérifie que les deux `requirements.txt` livrés décrivent encore le lock. Ces fichiers sont exigés par le cahier des charges, mais ils sont **dérivés** : `uv.lock` reste la seule source de vérité, et toute divergence fait échouer la chaîne.

**`CI · Image et conteneur → CD · Publication`** construit l'image, la charge dans le démon du runner et l'éprouve avec `scripts/smoke_docker.py` : un conteneur réel démarre, ses routes sont interrogées en HTTP, l'absence de MLflow et de PyArrow est vérifiée, et un artefact falsifié doit **empêcher** le démarrage.

La livraison ne s'exécute qu'après une poussée sur `main` :

```yaml
if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

Une proposition de fusion construit donc son image et l'éprouve, mais ne touche ni aux identifiants ni au registre. L'image publiée n'est **jamais reconstruite** : `docker tag` renomme celle qui vient de passer les tests, de sorte que les octets publiés sont exactement les octets éprouvés.

### Ce qui n'est pas vérifié en intégration continue

Dix tests sont ignorés sur un runner. Le résumé de chaque exécution les nomme, avec leur motif.

| Tests | Pourquoi |
|---|---|
| 8 tests de parité (`test_preprocessing.py`) | ils comparent la reconstruction aux colonnes que le dataset porte lui-même ; `data/` n'est pas versionné |
| 1 contrôle des bornes du domaine (`test_api.py`) | il confronte `config.py` aux minima et maxima réellement observés dans `train` |
| 1 parité avec le registre (`test_api.py`) | `mlflow.db` n'existe pas sur un runner |

Ce ne sont **pas** des tests désactivés. Ils s'exécutent intégralement sur un poste qui possède les données, et c'est là qu'ils doivent passer avant toute fusion. Fabriquer un jeu de données pour les faire tourner en CI les rendrait tautologiques : inventer la référence, puis vérifier qu'on la retrouve.

Tout le reste du module de prétraitement s'éprouve bien en CI, sur l'**artefact déployé** : `models/modele_api.joblib` embarque la table FAO ajustée à l'entraînement, donc les règles de dérivation se vérifient sans aucune donnée, et sur la pièce effectivement livrée.

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

### De la promotion d'un modèle à l'image publiée

La chaîne GitHub Actions **n'interroge jamais MLflow** : elle emballe l'artefact versionné, elle ne l'exporte pas. Son test de fumée vérifie d'ailleurs l'absence de MLflow dans l'image. Changer le modèle servi demande donc quatre temps :

1. promouvoir l'alias depuis le notebook de comparaison, avec `PUBLIER = True` ;
2. `uv run python scripts/export_api_model.py`, qui réexporte l'artefact, sa provenance et son empreinte ;
3. committer ce nouvel artefact ;
4. laisser la chaîne construire et publier l'image.

Déplacer un alias ne met à jour ni une image déjà publiée, ni un conteneur en fonctionnement. C'est ce qui rend chaque image traçable à un commit.

### Protection de `main`, épinglage et Dependabot

`main` est protégée par un *ruleset* actif : suppression interdite, réécriture d'historique interdite, proposition de fusion obligatoire, et les deux contrôles exigés verts avant toute fusion. La liste de dérogation est vide — la règle s'applique aussi à la propriétaire du dépôt.

Toutes les actions sont épinglées sur un **SHA de commit**, leur numéro de version en commentaire. Une étiquette, même `@v7`, peut être déplacée par son mainteneur ; un SHA, non. Cela compte surtout pour `docker/login-action`, le seul pas qui manipule le jeton.

[`.github/dependabot.yml`](.github/dependabot.yml) les fait vieillir sous surveillance, chaque semaine, pour le **seul** écosystème `github-actions`. Y déclarer Python ouvrirait des propositions qui modifient `uv.lock` sans toucher aux `requirements.txt` exportés : chacune échouerait sur le contrôle de dérive. Les mises à jour de *sécurité* Python, elles, arrivent sans configuration.

---

## 5. Suivi des expériences et registre MLflow

### Ouvrir l'interface

MLflow 3 a placé le backend fichier (`./mlruns`) en mode maintenance : le suivi utilise donc une base **SQLite**, `mlflow.db`, et les artefacts de modèle vont dans `mlartifacts/`. Ni l'un ni l'autre n'est versionné.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

L'interface s'ouvre sur `http://localhost:5000`.

### Organisation des runs

Une seule expérience, `agritech-rendement` : les quatre modèles sur les 13 variables. Chaque étude d'optimisation est un run parent, chaque essai Optuna un run enfant. Seul le meilleur essai de chaque étude sérialise un modèle — un artefact par essai ferait gonfler le stockage de plusieurs gigaoctets sans utilité.

Les essais enfants étant cent fois plus nombreux que les parents, la liste est vite saturée. Pour n'afficher que les quatre études :

```
tags.phase = 'optimisation'
```

et `tags.phase != ''` y ajoute le run d'évaluation finale, marqué `phase=test`. Celui-ci est séparé à dessein : mélangées dans un même tri par RMSE, les métriques de `val` et de `test` ne voudraient plus rien dire.

### Jeux de données attachés aux runs

Chaque run déclare les partitions qui l'ont produit : `train` et `val` pour une étude d'optimisation, `val` et `test` pour l'évaluation finale. MLflow n'en stocke **pas les données**, seulement le schéma typé, le nombre de lignes, une empreinte du contenu et le chemin du fichier.

Cette empreinte porte sur le contenu ; les tags `data_version` et `split_version` portent sur les fichiers. Les deux se complètent — la première détecte qu'un contenu a changé, les seconds disent de quelle génération de dataset le run provient.

### Registre, versions et alias

`train.py` publie chaque modèle entraîné comme une **version** du modèle enregistré `agritech-rendement`, taguée avec l'algorithme et les empreintes de données. Aucune n'est privilégiée : le choix appartient au notebook de comparaison, qui pose l'alias `champion` en appelant `training.promouvoir_champion`.

MLflow 3 ayant supprimé les *stages* (`Staging`, `Production`) au profit des alias, c'est `champion` qui tient ce rôle.

### Trois artefacts à ne pas confondre

| Artefact | Produit par | Rôle |
|---|---|---|
| `models/modele_final.joblib` | le notebook de comparaison | le modèle retenu, livrable de la mission |
| `models/modele_api.joblib` | `scripts/export_api_model.py` | ce que sert l'API : prétraitement **et** modèle, gelés ensemble |
| `agritech-rendement-api` au registre | le même script | la même pièce, publiée et aliasée dans MLflow |

Le registre `agritech-rendement` porte les modèles issus de l'entraînement ; `agritech-rendement-api` porte l'artefact destiné au service. Les confondre revient à croire qu'une promotion déploie.

---

## 6. Fonctionnement de l'application

### L'API — FastAPI

Le service est **sans état** : il ne stocke rien entre deux appels, hormis le journal des requêtes. Chaque réponse porte, outre son résultat, la provenance du modèle qui l'a produite — registre, version, empreinte — de sorte qu'une prédiction reste rattachable à l'artefact exact qui l'a calculée.

Les contrats d'entrée et de sortie sont décrits par des schémas Pydantic : une saisie hors vocabulaire est rejetée en HTTP 422 avec le détail du champ fautif, jamais silencieusement convertie.

| Route | Ce qu'elle rend |
|---|---|
| `POST /predict` | le rendement attendu d'une parcelle et son intervalle à 95 % |
| `POST /recommend` | les six cultures classées par rendement, et par marge brute si des charges sont fournies |
| `GET /health` | l'état du service et la provenance du modèle chargé |
| `GET /model/info` | registre, version, alias, empreintes, versions des bibliothèques |
| `GET /reference/prices` | les prix de référence par culture, source FAOSTAT 2024, avec leur dispersion |
| `GET /logs` | le journal des appels, refus compris |

La documentation interactive est servie sur `/docs`.

### L'artefact embarqué et son empreinte

L'image ne contient ni MLflow, ni PyArrow, ni uv : elle charge un `.joblib` par pickle standard. Avant de le désérialiser, le service **vérifie son empreinte SHA-256** contre celle inscrite dans `models/modele_api.json`. Un artefact corrompu, tronqué ou remplacé seul empêche alors le démarrage, au lieu de servir des prédictions fausses — comportement éprouvé à chaque exécution de la chaîne par `scripts/smoke_docker.py`.

C'est un **contrôle d'intégrité, pas une signature** : il atteste que le `.joblib` est bien celui que décrit son fichier de provenance. Remplacer les deux ensemble passerait le contrôle. Ce qui s'y oppose est ailleurs — les deux fichiers sont versionnés, et toute modification passe par une proposition de fusion sur une branche protégée.

Le registre MLflow n'est une source de chargement que si `AGRITECH_MODEL_URI` est posée, et il est alors résolu **une seule fois au démarrage**.

### L'interface — Streamlit

L'interface n'importe ni FastAPI, ni le modèle : elle interroge le service en HTTP, et les deux pourraient être déployés séparément — sur deux machines, ou l'un en conteneur et l'autre non. C'est `AGRITECH_API_URL` qui les relie.

Elle propose deux modes — décrire une parcelle pour en estimer le rendement, ou comparer les six cultures.

Le second mode dépasse le rendement. Les six cultures étant statistiquement proches, l'interface permet de saisir les charges opérationnelles par culture et calcule une **marge brute simulée** — rendement × prix − charges —, qui les sépare bien mieux. Le détail de la conception est dans [`reports/streamlit_app_summary.md`](reports/streamlit_app_summary.md).

---

## 7. Limites et perspectives

### Un jeu de données simulé

Le jeu de parcelles est **simulé**, pas observé. Ses modalités sont équilibrées, ses relations régulières, et sa variable `Region` ne correspond à aucune géographie réelle. Un R² de 0,91 décrit donc la capacité du modèle à retrouver une règle de génération, pas sa valeur agronomique sur des parcelles réelles.

Toute conclusion tirée ici devrait être remesurée sur des données de terrain avant d'orienter une décision.

### Ce que la recommandation n'établit pas

Les six cultures rendent des prédictions **très proches** — l'écart entre la meilleure et la pire se compte en dixièmes de tonne par hectare, souvent sous le seuil de 0,1 t/ha que l'API juge discriminant. L'API le signale explicitement plutôt que de laisser croire à un classement net.

Les coefficients du modèle décrivent des **associations dans ce jeu de données**, pas des effets causaux. Qu'une parcelle fertilisée y rende davantage ne garantit pas que fertiliser une parcelle donnée augmentera son rendement d'autant.

Enfin, le modèle **extrapole linéairement** hors du domaine observé — 100 à 1000 mm de pluie, 15 à 40 °C. L'API avertit au-delà, mais ne refuse pas : une saison plus humide que toutes celles observées reste une question légitime, dont la réponse est simplement moins fiable.

### Hypothèses économiques

La marge brute repose sur un prix de référence de 200 €/t, avec une sensibilité calculée à 150, 200 et 300 €/t. Les prix par culture viennent de FAOSTAT 2024 et portent leur dispersion entre pays déclarants. Les charges, elles, sont saisies par l'utilisateur : l'outil ne les présume pas.

### Perspectives

Héberger le service et l'interface, pour qu'un agriculteur n'ait rien à installer. Mesurer la dérive du modèle en production. Et remplacer le jeu simulé par des observations réelles — c'est de loin ce qui changerait le plus la valeur de l'ensemble.

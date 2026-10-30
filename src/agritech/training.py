"""Construction, optimisation et journalisation des modèles de rendement.

**Ce module est une bibliothèque : il ne s'exécute pas tout seul.**
Il définit *comment* on construit et optimise un modèle. C'est
`scripts/train.py` qui décide *quoi* lancer et avec quels réglages.

Trois principes de conception :

- **un seul prétraitement**, partagé par les quatre modèles, pour que la
  comparaison porte sur les modèles et non sur leur préparation ;
- **l'optimisation vise `val`**, jamais `test`, qui reste réservé à
  l'évaluation finale du seul modèle retenu ;
- **aucun modèle n'est sérialisé pendant la recherche.** Un artefact par essai
  Optuna produirait des centaines de fichiers ; seul le meilleur essai de
  chaque étude est enregistré.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agritech.config import (
    ALIAS_CHAMPION,
    EXPERIMENT_NOM,
    FEATURES_BOOLEENNES,
    FEATURES_CATEGORIELLES,
    FEATURES_MODELE,
    FEATURES_NUMERIQUES,
    HYPOTHESES_ECONOMIQUES_VERSION,
    MLARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    SPLIT_METADATA_PATH,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from agritech.evaluation import evaluer, metriques_prediction

# Les 3 variables FAO quantitatives. Ce sont les seules colonnes du dataset
# porteuses de valeurs manquantes (33 % : Barley et Cotton sont absents du
# référentiel FAO), d'où l'imputation par la médiane plus bas.
FAO_QUANTITATIVES = [
    "crop_productivity_index",
    "crop_yield_geo_cv",
    "crop_yield_trend_decade",
]

#: Plus aucune variable ordinale depuis le retrait de `inputs_level` (§8.7)
FEATURES_ORDINALES: list[str] = []

# Un seul jeu de variables : les treize du dataset consolidé, toutes utilisées.
JEUX_DE_VARIABLES = {
    "modele": (FEATURES_MODELE, EXPERIMENT_NOM),
}

#: Jeu entraîné quand la ligne de commande n'en précise aucun
JEU_PAR_DEFAUT = "modele"


# --------------------------------------------------------------------------- #
# 1. Chargement des données
# --------------------------------------------------------------------------- #


def charger_partitions(echantillon: int | None = None, avec_test: bool = False):
    """Lit les partitions figées par `notebooks/data_split.ipynb`.

    `echantillon` ne tronque que `train`, pour les tests d'intégration rapides.
    `val` et `test` ne sont jamais réduits : leurs effectifs doivent rester
    constants pour que les métriques restent comparables d'un run à l'autre.

    Renvoie `(train, val)`, ou `(train, val, test)` si `avec_test=True` — à
    n'activer qu'au moment de l'évaluation finale du modèle retenu.
    """
    train = pd.read_parquet(TRAIN_PATH)
    if echantillon:
        train = train.sample(echantillon, random_state=RANDOM_STATE)

    partitions = [train, pd.read_parquet(VAL_PATH)]
    if avec_test:
        partitions.append(pd.read_parquet(TEST_PATH))
    return tuple(partitions)


def versions_des_donnees() -> dict[str, str]:
    """Empreintes à attacher en tags à chaque run MLflow.

    Deux runs portant des empreintes différentes ne sont **pas comparables** :
    ils ne parlent ni du même dataset, ni du même découpage, ni des mêmes
    hypothèses économiques.
    """
    metadonnees = json.loads(SPLIT_METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "data_version": metadonnees["data_version"],      # dataset consolidé
        "split_version": metadonnees["split_version"],    # les 3 partitions
        "hypotheses_economiques": HYPOTHESES_ECONOMIQUES_VERSION,  # prix, coûts
    }


# --------------------------------------------------------------------------- #
# 2. Prétraitement : identique pour les quatre modèles
# --------------------------------------------------------------------------- #


def construire_pretraitement(features: list[str]) -> ColumnTransformer:
    """Assemble les transformations à appliquer aux colonnes d'entrée.

    Un `ColumnTransformer` applique un traitement différent selon la colonne.
    Celui-ci est **unique pour les quatre modèles** : un écart de performance
    est alors attribuable au modèle, et non à sa préparation. C'est un choix de
    protocole, pas une obligation méthodologique — comparer des pipelines
    optimisés pour chaque algorithme serait tout aussi légitime, et
    sélectionnerait la meilleure solution de production plutôt que le meilleur
    algorithme.

    **Pourquoi pas le support catégoriel natif des modèles à base d'arbres ?**
    Mesuré sur `val` : LightGBM avec ses catégories natives contre le one-hot
    donne z = −0,96, soit un écart indiscernable. Le one-hot évite en prime un
    second chemin de prétraitement, donc une seconde source de divergence avec
    l'API.

    **Pourquoi pas un encodage par la cible ?** Pour la simplicité, pas pour un
    risque de fuite : le `TargetEncoder` de scikit-learn procède par
    cross-fitting interne et ne fuit pas. Avec 3 à 6 modalités par variable, il
    n'apporterait rien que le one-hot ne fasse déjà.
    """
    # Répartition des colonnes demandées en trois familles de traitement.
    # Les deux premières listes sont explicites ; TOUT le reste est traité comme
    # quantitatif. Sans ce repli, une colonne absente des listes de `config`
    # serait silencieusement ignorée par le ColumnTransformer — le modèle
    # s'entraînerait sans elle, sans le moindre avertissement.
    categorielles = [c for c in features if c in FEATURES_CATEGORIELLES]
    booleennes = [c for c in features
                  if c in FEATURES_BOOLEENNES or c == "crop_in_fao_reference"]
    quantitatives = [c for c in features if c not in categorielles + booleennes]

    assert sorted(quantitatives + categorielles + booleennes) == sorted(features),         "Des colonnes ne seraient pas transmises au modèle"

    return ColumnTransformer(
        [
            # Quantitatives : on remplace les valeurs manquantes par la médiane,
            # puis on centre et réduit. La médiane est apprise sur `train` seul,
            # au moment du .fit() — c'est ce qui évite la fuite de données.
            # La standardisation sert à Ridge, dont la pénalité dépend de
            # l'échelle. Elle est sans effet sur les arbres, et ce n'est pas un
            # constat empirique : une transformation affine est monotone, donc
            # elle ne change ni l'ordre des valeurs, ni les coupures possibles,
            # ni le découpage en histogrammes. Mesuré : 6,5e-05 d'écart de
            # RMSE sur LightGBM, du bruit d'arrondi.
            ("quantitatives",
             make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
             quantitatives),

            # Catégorielles : une colonne binaire par modalité, TOUTES les
            # modalités conservées (drop=None). Avec la pénalité L2 de Ridge,
            # les coefficients d'un même bloc somment à zéro : chacun se lit
            # comme un écart à la moyenne des EFFETS des modalités — non
            # pondérée par leur fréquence, donc pas la moyenne empirique des
            # rendements.
            #
            # `handle_unknown='ignore'` est un FILET TECHNIQUE, pas une
            # fonctionnalité : une modalité inconnue produit une ligne de zéros,
            # donc une prédiction sans fiabilité. C'est à l'API de valider les
            # modalités autorisées et de refuser une saisie invalide.
            ("categorielles",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorielles),

            # Booléennes : déjà 0/1, rien à faire. `crop_in_fao_reference` sert
            # au passage d'indicateur de valeur manquante pour les 3 colonnes
            # FAO — pas besoin d'en ajouter un avec `add_indicator`.
            ("booleennes", "passthrough", booleennes),
        ],
        # Garde les noms de colonnes lisibles en sortie (`Crop_Maize` plutôt
        # que `categorielles__Crop_Maize`), utile pour lire les coefficients.
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# 3. Les quatre modèles
# --------------------------------------------------------------------------- #

# Pour chaque modèle : sa classe, et les arguments qui ne sont PAS optimisés.
# Tout ce qui n'est pas ici est exploré par Optuna dans `espace_recherche`.
#
# LightGBM vaut None : sa classe est importée à la demande par
# `_classe_lightgbm`, pour ne pas charger la bibliothèque quand on n'entraîne
# qu'un Ridge.
ESTIMATEURS = {
    # Référence : prédit toujours la moyenne de train. Sert de plancher —
    # un modèle qui ne le bat pas n'a rien appris.
    "dummy": (DummyRegressor, {"strategy": "mean"}),

    # Régression linéaire pénalisée.
    "ridge": (Ridge, {}),

    # Gradient boosting de scikit-learn.
    "histgbm": (HistGradientBoostingRegressor, {"random_state": RANDOM_STATE}),

    # Gradient boosting de LightGBM. `subsample_freq=1` est indispensable ici :
    # sans lui, le paramètre `subsample` exploré par Optuna serait ignoré.
    "lightgbm": (None, {"random_state": RANDOM_STATE, "verbose": -1,
                        "n_jobs": -1, "subsample_freq": 1}),
}


def _classe_lightgbm():
    """Import différé de LightGBM, inutile pour les trois autres modèles."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor


def construire_pipeline(nom_modele: str, parametres: dict,
                        features: list[str]) -> Pipeline:
    """Assemble prétraitement et estimateur en **un seul objet**.

    Parce que le prétraitement est *à l'intérieur* du Pipeline, l'objet
    sérialisé contient les médianes d'imputation et les catégories apprises sur
    `train` : l'API les rejoue à l'identique sans pouvoir en diverger.

    **Ce que le Pipeline ne contient pas encore.** Il attend les 13 variables
    du dataset, alors que l'agriculteur n'en saisira que 8. Les 5 autres —
    `low_rain_no_irrigation` et les 4 indicateurs FAO — sont dérivées, et rien
    ici ne les reconstruit : une prédiction à partir des seuls champs saisis
    échoue sur `columns are missing`. Tant qu'un transformateur de tête ne les
    calcule pas à partir des 8 champs, avec la table FAO embarquée, le modèle
    n'est pas déployable tel quel.
    """
    if nom_modele not in ESTIMATEURS:
        raise ValueError(f"Modèle inconnu : {nom_modele!r}. "
                         f"Attendu : {', '.join(ESTIMATEURS)}")

    classe, arguments_fixes = ESTIMATEURS[nom_modele]
    classe = classe or _classe_lightgbm()   # None = LightGBM, importé ici

    return Pipeline([
        ("pretraitement", construire_pretraitement(features)),
        ("modele", classe(**arguments_fixes, **parametres)),
    ])


#: Espaces de recherche, sous forme declarative.
#:
#: Source UNIQUE : `espace_recherche` les consomme a l'entrainement, et le
#: notebook de comparaison les reutilise pour reconstituer les etudes Optuna a
#: partir des runs MLflow. Ecrits a un seul endroit, ils ne peuvent pas diverger.
#:
#: `log=True` echantillonne sur une echelle logarithmique, adaptee aux
#: parametres couvrant plusieurs ordres de grandeur : un alpha de 0,001 et de
#: 0,01 different autant que 10 et 100.
ESPACES: dict[str, dict] = {
    # Le modele de reference n'a rien a regler : il predit la moyenne.
    "dummy": {},

    # alpha = force de la penalite. Plus il est grand, plus les coefficients
    # sont tires vers zero.
    "ridge": {"alpha": ("float", 1e-4, 1e4, True)},

    "histgbm": {
        # Pas d'apprentissage : petit = lent mais plus fin
        "learning_rate": ("float", 0.01, 0.3, True),
        # Nombre d'arbres construits
        "max_iter": ("int", 100, 500, False),
        # Complexite d'un arbre et taille minimale d'une feuille : les deux
        # leviers principaux contre le surapprentissage
        "max_leaf_nodes": ("int", 15, 255, False),
        "min_samples_leaf": ("int", 10, 200, False),
        "l2_regularization": ("float", 1e-8, 1.0, True),
    },

    "lightgbm": {
        "learning_rate": ("float", 0.01, 0.3, True),
        # Le nombre d'arbres est optimise plutot qu'arrete precocement sur
        # `val` : cela evite d'utiliser val a l'interieur de l'ajustement
        # lui-meme, en plus de l'utiliser pour comparer les modeles.
        "n_estimators": ("int", 100, 800, False),
        "num_leaves": ("int", 15, 255, False),
        "min_child_samples": ("int", 10, 200, False),
        # Echantillonnage aleatoire des lignes et des colonnes a chaque arbre :
        # decorrele les arbres entre eux
        "subsample": ("float", 0.6, 1.0, False),
        "colsample_bytree": ("float", 0.6, 1.0, False),
        "reg_lambda": ("float", 1e-8, 10.0, True),
    },
}


def espace_recherche(nom_modele: str, trial) -> dict:
    """Demande a Optuna une valeur pour chaque hyperparametre de `ESPACES`.

    `trial.suggest_*` ne tire pas au hasard : Optuna choisit en tenant compte
    des essais precedents.

    **Tout ce qui est renvoye ici vient d'un `suggest_*`.** Une constante
    glissee dans ce dictionnaire serait utilisee pendant la recherche mais
    absente de `etude.best_params`, donc perdue au reajustement final : le
    modele enregistre differerait du meilleur essai. Les constantes vont dans
    `ESTIMATEURS`.
    """
    if nom_modele not in ESPACES:
        raise ValueError(f"Espace de recherche non défini pour {nom_modele!r}")

    valeurs = {}
    for nom, (genre, bas, haut, log) in ESPACES[nom_modele].items():
        if genre == "float":
            valeurs[nom] = trial.suggest_float(nom, bas, haut, log=log)
        elif genre == "int":
            valeurs[nom] = trial.suggest_int(nom, bas, haut, log=log)
        else:
            raise ValueError(f"Type d'espace inconnu : {genre!r}")
    return valeurs


def distributions_optuna(nom_modele: str) -> dict:
    """Traduit `ESPACES` en distributions Optuna.

    Sert a reconstituer une etude a partir de ses essais enregistres dans
    MLflow, sans avoir a re-declarer les bornes.
    """
    from optuna.distributions import FloatDistribution, IntDistribution

    classes = {"float": FloatDistribution, "int": IntDistribution}
    return {nom: classes[genre](bas, haut, log=log)
            for nom, (genre, bas, haut, log) in ESPACES[nom_modele].items()}


# --------------------------------------------------------------------------- #
# 4. Suivi MLflow
# --------------------------------------------------------------------------- #


#: Pour chaque partition : le contexte MLflow et le fichier dont elle vient.
#: Le contexte est ce qui s'affiche dans la colonne « Dataset » de l'interface.
CONTEXTES_JEUX = {
    "train": ("training", TRAIN_PATH),
    "val": ("validation", VAL_PATH),
    "test": ("test", TEST_PATH),
}


def journaliser_jeux(mlflow, **jeux: pd.DataFrame) -> None:
    """Attache à un run les partitions qui l'ont produit.

    MLflow n'enregistre **pas les données** : seulement le schéma typé, le
    nombre de lignes, une empreinte du contenu et le chemin du fichier source.
    Le coût est négligeable — moins d'une seconde pour 700 000 lignes — et la
    base n'en grossit pas.

    Cette empreinte porte sur le **contenu**, là où les tags `data_version` et
    `split_version` portent sur les **fichiers**. Les deux se complètent : la
    première détecte qu'un contenu a changé, les seconds disent de quelle
    génération de dataset le run provient.

    Un sous-échantillonnage (`--sample`) produit une empreinte différente de
    celle du fichier complet : c'est voulu, le run doit décrire ce qu'il a
    réellement vu.
    """
    for nom, donnees in jeux.items():
        contexte, chemin = CONTEXTES_JEUX[nom]
        mlflow.log_input(
            mlflow.data.from_pandas(donnees, source=chemin.as_uri(),
                                    targets=TARGET, name=nom),
            context=contexte,
        )


def configurer_mlflow(jeu_de_variables: str):
    """Branche MLflow sur la base locale et sélectionne la bonne expérience.

    MLflow 3 a placé le backend fichier (`./mlruns`) en mode maintenance : le
    suivi passe donc par une base **SQLite**. Les artefacts de modèle, trop
    volumineux pour une base, vont sur disque dans `mlartifacts/`, avec un
    sous-dossier par expérience.

    Renvoie le module `mlflow` configuré, pour que l'appelant n'ait pas à
    l'importer de son côté.
    """
    import mlflow

    _, nom_experience = JEUX_DE_VARIABLES[jeu_de_variables]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # À la première exécution, l'expérience n'existe pas : on la crée en
    # fixant nous-mêmes où iront ses artefacts.
    if mlflow.get_experiment_by_name(nom_experience) is None:
        dossier = MLARTIFACTS_DIR / nom_experience
        dossier.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(nom_experience, artifact_location=dossier.as_uri())

    mlflow.set_experiment(nom_experience)
    return mlflow


# --------------------------------------------------------------------------- #
# 5. Optimisation
# --------------------------------------------------------------------------- #


def optimiser(nom_modele: str, jeu_de_variables: str, train: pd.DataFrame,
              val: pd.DataFrame, n_essais: int, suivi_mlflow: bool = True):
    """Cherche les hyperparamètres qui minimisent le RMSE sur `val`.

    Déroulé : Optuna propose un jeu d'hyperparamètres, on construit le Pipeline
    correspondant, on l'ajuste sur `train`, on mesure sur `val`, et on rend le
    RMSE à Optuna — qui s'en sert pour proposer le jeu suivant. Cent fois.

    Dans MLflow : **un run parent par étude, un run enfant par essai.** Aucun
    modèle n'est sérialisé dans les enfants ; seul le meilleur essai l'est, par
    `journaliser_meilleur`.

    Limite assumée : `val` sert à la fois au réglage des hyperparamètres et à la
    comparaison des modèles, ce qui rend son estimation légèrement optimiste.
    C'est exactement pour cela que `test` n'est ouvert qu'après le choix final.

    Renvoie `(etude, pipeline)` : l'étude pour ses statistiques, et le Pipeline
    du meilleur essai, déjà réajusté sur `train`. Le réajustement a lieu une
    seule fois et sert aussi bien à MLflow qu'à l'export `.joblib`.
    """
    import optuna

    features, _ = JEUX_DE_VARIABLES[jeu_de_variables]
    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = val[features], val[TARGET]

    mlflow = configurer_mlflow(jeu_de_variables) if suivi_mlflow else None

    # Le modèle de référence n'a aucun hyperparamètre : 100 essais donneraient
    # 100 fois le même résultat.
    essais_utiles = 1 if nom_modele == "dummy" else n_essais

    def objectif(trial):
        """Un essai : construire, ajuster, mesurer. Renvoie ce qu'Optuna minimise."""
        parametres = espace_recherche(nom_modele, trial)
        pipeline = construire_pipeline(nom_modele, parametres, features)
        pipeline.fit(X_train, y_train)
        mesures = metriques_prediction(y_val, pipeline.predict(X_val))

        if mlflow:
            # `nested=True` rattache ce run à celui ouvert plus bas
            with mlflow.start_run(run_name=f"essai-{trial.number:03d}", nested=True):
                mlflow.log_params(parametres)
                mlflow.log_metrics({f"val_{nom}": v for nom, v in mesures.items()})

        return mesures["rmse"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)   # sans quoi : un log par essai

    etude = optuna.create_study(
        study_name=f"{nom_modele}-{jeu_de_variables}",
        direction="minimize",                              # le RMSE, on le baisse
        # TPE : à partir des essais passés, modélise les zones prometteuses de
        # l'espace. Plus efficace qu'une grille ou qu'un tirage aléatoire.
        # La graine rend la recherche reproductible à l'identique.
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )

    if mlflow:
        with mlflow.start_run(run_name=f"{nom_modele}-{jeu_de_variables}"):
            mlflow.set_tags({**versions_des_donnees(),
                             "modele": nom_modele,
                             "jeu_de_variables": jeu_de_variables,
                             "phase": "optimisation",
                             "methode_recherche": "optuna-tpe"})
            mlflow.log_param("n_essais", essais_utiles)
            journaliser_jeux(mlflow, train=train, val=val)
            etude.optimize(objectif, n_trials=essais_utiles)
            pipeline = reajuster_meilleur(etude, nom_modele, features, train)
            journaliser_meilleur(mlflow, etude, pipeline, nom_modele,
                                 features, val)
    else:
        etude.optimize(objectif, n_trials=essais_utiles)
        pipeline = reajuster_meilleur(etude, nom_modele, features, train)

    return etude, pipeline


def reajuster_meilleur(etude, nom_modele: str, features: list[str],
                       train: pd.DataFrame) -> Pipeline:
    """Reconstruit et ajuste le Pipeline du meilleur essai.

    Les essais ont bien ajusté un Pipeline chacun, mais aucun n'a été conservé —
    les garder tous en mémoire serait absurde. On réajuste donc ici le seul jeu
    d'hyperparamètres qui nous intéresse.
    """
    pipeline = construire_pipeline(nom_modele, etude.best_params, features)
    pipeline.fit(train[features], train[TARGET])
    return pipeline


def journaliser_meilleur(mlflow, etude, pipeline: Pipeline, nom_modele: str,
                         features: list[str], val: pd.DataFrame):
    """Enregistre dans MLflow le modèle du meilleur essai — le seul du lot.

    Les runs enfants n'ont reçu que des métriques : sérialiser un modèle par
    essai produirait des centaines de fichiers pour des dizaines de Mo chacun.
    """
    from mlflow.models import infer_signature

    # Préfixe `best_` pour distinguer ces paramètres de ceux des runs enfants
    mlflow.log_params({f"best_{nom}": v for nom, v in etude.best_params.items()})
    mlflow.log_metrics(evaluer(pipeline, val, features, prefixe="val"))

    # Exemple d'entrée, qui sert à MLflow à déduire le schéma attendu.
    # Les colonnes entières sont converties en flottantes : un entier ne peut
    # pas porter de valeur manquante, or l'API acceptera un `Days_to_Harvest`
    # absent. Sans cette conversion, la validation du schéma rejetterait
    # l'appel au lieu de laisser l'imputation faire son travail.
    exemple = val[features].head(5).copy()
    colonnes_entieres = exemple.select_dtypes(include="integer").columns
    exemple[colonnes_entieres] = exemple[colonnes_entieres].astype("float64")

    info = mlflow.sklearn.log_model(
        pipeline,
        name="modele",
        # cloudpickle plutôt que le format skops par défaut : skops refuse
        # `numpy.dtype` comme type non fiable, et ne saurait de toute façon pas
        # sérialiser le transformateur maison qui viendra se greffer devant.
        serialization_format="cloudpickle",
        signature=infer_signature(exemple, pipeline.predict(exemple)),
        input_example=exemple,
    )

    # Publication au **registre**, sous un nom unique : chaque modèle entraîné
    # y entre comme une version candidate. Lequel part en production n'est pas
    # décidé ici — c'est l'affaire de `promouvoir_champion`, appelée depuis le
    # notebook de comparaison une fois le choix multicritère arrêté.
    #
    # Les tags portés par la version permettent de la retrouver ensuite :
    # `modele` dit de quel algorithme il s'agit, les empreintes de données
    # disent sur quelle génération de dataset il a été entraîné.
    mlflow.register_model(
        f"models:/{info.model_id}",
        REGISTERED_MODEL_NAME,
        tags={"modele": nom_modele, **versions_des_donnees()},
    )


def promouvoir_champion(nom_modele: str,
                        jeu_de_variables: str = JEU_PAR_DEFAUT) -> int:
    """Désigne au registre la version qui part en production.

    `journaliser_meilleur` publie une version par modèle entraîné, sans en
    privilégier aucune : le choix est **multicritère et humain**, il appartient
    au notebook de comparaison. Cette fonction ne fait qu'enregistrer sa
    décision.

    Une fois l'alias posé, la décision est enregistrée — mais rien n'est
    déployé. L'API sert `models/modele_api.joblib`, embarqué dans son image et
    scellé par une empreinte vérifiée au démarrage. Il faut donc réexporter cet
    artefact (`scripts/export_api_model.py`), le committer, puis laisser la
    chaîne CI/CD construire et publier la nouvelle image.

    La version retenue est la **plus récente** parmi celles qui portent ce nom
    de modèle *et* la génération de données courante : un réentraînement crée
    une nouvelle version, et c'est elle qui doit être promue.
    """
    from mlflow import MlflowClient

    configurer_mlflow(jeu_de_variables)
    empreintes = versions_des_donnees()
    client = MlflowClient()

    candidates = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}' "
        f"and tags.modele = '{nom_modele}' "
        f"and tags.data_version = '{empreintes['data_version']}'"
    )
    if not candidates:
        raise RuntimeError(
            f"Aucune version de {nom_modele!r} au registre pour le dataset "
            f"{empreintes['data_version']}. Lancer `scripts/train.py` d'abord."
        )

    version = max(candidates, key=lambda v: int(v.version))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION,
                                      version.version)
    print(f"  {REGISTERED_MODEL_NAME} version {version.version} "
          f"({nom_modele}) -> alias « {ALIAS_CHAMPION} »")
    print(f"  l'API chargera : "
          f"models:/{REGISTERED_MODEL_NAME}@{ALIAS_CHAMPION}")
    return int(version.version)


def journaliser_evaluation_test(nom_modele: str, pipeline: Pipeline,
                                features: list[str], val: pd.DataFrame,
                                test: pd.DataFrame,
                                jeu_de_variables: str = JEU_PAR_DEFAUT) -> str:
    """Enregistre dans MLflow l'évaluation finale du modèle retenu sur `test`.

    Un run **séparé** de ceux de l'optimisation, marqué `phase=test`. Les
    métriques de test ne doivent jamais se mêler à celles qui ont servi au
    choix du modèle : dans l'interface MLflow, un tri par RMSE qui mélangerait
    les deux ne voudrait plus rien dire.

    `val` et `test` sont journalisés **côte à côte**, avec leur écart. C'est
    cet écart qui démontre l'absence de surapprentissage — une valeur de test
    seule ne prouve rien, faute de point de comparaison.

    **Idempotent.** Relancer le notebook ne crée pas de doublon : un run
    existant pour le même modèle et les mêmes versions de données est repris
    et mis à jour, plutôt que dupliqué.
    """
    mlflow = configurer_mlflow(jeu_de_variables)
    _, nom_experience = JEUX_DE_VARIABLES[jeu_de_variables]
    experience = mlflow.get_experiment_by_name(nom_experience)
    versions = versions_des_donnees()

    # Un seul run par (modèle, version du dataset, version du découpage) :
    # au-delà, deux runs ne décriraient pas la même chose.
    filtre = " and ".join([
        "tags.phase = 'test'",
        f"tags.modele = '{nom_modele}'",
        f"tags.data_version = '{versions['data_version']}'",
        f"tags.split_version = '{versions['split_version']}'",
    ])
    deja = mlflow.search_runs([experience.experiment_id], filter_string=filtre,
                              output_format="list")
    reprise = deja[0].info.run_id if deja else None

    mesures = {**evaluer(pipeline, val, features, prefixe="val"),
               **evaluer(pipeline, test, features, prefixe="test")}
    # L'indicateur qui se lit directement dans l'interface : un écart proche de
    # zéro signifie que la performance tient hors du jeu ayant servi au choix.
    mesures["ecart_val_test_rmse"] = mesures["test_rmse"] - mesures["val_rmse"]

    with mlflow.start_run(run_id=reprise,
                          run_name=f"evaluation-finale-{nom_modele}") as run:
        mlflow.set_tags({**versions,
                         "modele": nom_modele,
                         "jeu_de_variables": jeu_de_variables,
                         "phase": "test",
                         "modele_retenu": "oui"})
        mlflow.log_params({"n_val": len(val), "n_test": len(test),
                           "n_variables": len(features)})
        mlflow.log_metrics(mesures)
        journaliser_jeux(mlflow, val=val, test=test)

    print(f"  run « evaluation-finale-{nom_modele} » "
          f"{'mis a jour' if reprise else 'cree'}")
    print(f"  val_rmse {mesures['val_rmse']:.4f}  ->  "
          f"test_rmse {mesures['test_rmse']:.4f}   "
          f"(ecart {mesures['ecart_val_test_rmse']:+.4f})")
    return run.info.run_id


def exporter_joblib(pipeline, chemin) -> None:
    """Écrit le modèle dans un fichier `.joblib` autonome.

    En double de l'artefact MLflow, pour deux raisons : la mission demande
    explicitement un modèle sauvegardé en `.pkl` ou `.joblib`, et le notebook
    de comparaison recharge ainsi les modèles instantanément au lieu de les
    réajuster.
    """
    import joblib

    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, chemin)
    print(f"  {chemin.name:<28} {chemin.stat().st_size / 1024**2:>7.2f} Mo")


def chemin_modele(nom_modele: str, jeu_de_variables: str):
    """Nom de fichier d'un modèle optimisé : `ridge_parcelle.joblib`."""
    return MODELS_DIR / f"{nom_modele}_{jeu_de_variables}.joblib"

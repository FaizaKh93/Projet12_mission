"""Configuration du projet Agritech Answers.

Ce module ne contient que des **valeurs** : chemins, graines, vocabulaire du
domaine, hypothèses économiques et palette graphique. Aucune fonction, aucun
effet de bord à l'import — l'API doit pouvoir le charger sans embarquer la
pile graphique ni toucher au disque.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

# Ce fichier est dans src/agritech/ : la racine est deux niveaux au-dessus.
# L'installation étant éditable, ce chemin pointe bien vers le dépôt.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "OC_InputData"
PREDICTION_DIR = DATA_DIR / "Crop Yield Prediction Dataset"

# Jeux exploitables
CROP_YIELD_PATH = DATA_DIR / "Agriculture Crop Yield" / "crop_yield.csv"
YIELD_DF_PATH = PREDICTION_DIR / "yield_df.csv"

# Fichiers bruts, sources de yield_df.csv
YIELD_RAW_PATH = PREDICTION_DIR / "yield.csv"
PESTICIDES_PATH = PREDICTION_DIR / "pesticides.csv"
RAINFALL_PATH = PREDICTION_DIR / "rainfall.csv"
TEMP_PATH = PREDICTION_DIR / "temp.csv"

ALL_SOURCES = {
    "crop_yield.csv  (jeu A)": CROP_YIELD_PATH,
    "yield_df.csv  (jeu B, fusionné)": YIELD_DF_PATH,
    "yield.csv  (brut FAO)": YIELD_RAW_PATH,
    "pesticides.csv  (brut)": PESTICIDES_PATH,
    "rainfall.csv  (brut)": RAINFALL_PATH,
    "temp.csv  (brut)": TEMP_PATH,
}

# Artefacts intermédiaires : reconstructibles, utiles aux étapes suivantes
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FAO_PANEL_PATH = INTERIM_DIR / "fao_panel.csv"
CROP_REFERENCE_PATH = INTERIM_DIR / "crop_reference.csv"
DATA_DICTIONARY_PATH = INTERIM_DIR / "data_dictionary.csv"

# Livrable de l'étape 1. Le CSV est le format demandé par la mission ;
# le Parquet porte exactement les mêmes données, dans un format typé et compact,
# destiné aux étapes suivantes où il sera relu à chaque entraînement.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONSOLIDATED_PATH = PROCESSED_DIR / "crop_yield_consolidated.csv"
CONSOLIDATED_PARQUET_PATH = PROCESSED_DIR / "crop_yield_consolidated.parquet"

# --------------------------------------------------------------------------- #
# Partitions d'apprentissage
#
# Générées une seule fois par `notebooks/data_split.ipynb`, puis conservées.
# Le notebook d'entraînement les lit sans jamais les régénérer.
# --------------------------------------------------------------------------- #

SPLITS_DIR = PROCESSED_DIR / "splits"
TRAIN_PATH = SPLITS_DIR / "train.parquet"
VAL_PATH = SPLITS_DIR / "val.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"
SPLIT_METADATA_PATH = SPLITS_DIR / "split_metadata.json"

SPLIT_RANDOM_STATE = 42
SPLIT_PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_STRATIFY_ON = "Crop"

# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #

# Graine partagée par l'exploration, le split train/test et l'entraînement
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Vocabulaire du domaine (jeu A)
# --------------------------------------------------------------------------- #

TARGET = "Yield_tons_per_hectare"

NUMERIC_FEATURES = ["Rainfall_mm", "Temperature_Celsius", "Days_to_Harvest"]
CATEGORICAL_FEATURES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
BOOLEAN_FEATURES = ["Fertilizer_Used", "Irrigation_Used"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# --------------------------------------------------------------------------- #
# Variables du modèle
#
# Le dataset consolidé compte 14 colonnes : ces 13 variables explicatives et la
# cible. `inputs_level` et `Days_to_Harvest` en ont été retirées à la
# préparation (§8.7) — la première parce qu'elle est la somme exacte de deux
# colonnes présentes, la seconde parce que sa disponibilité au moment de la
# prédiction n'est pas acquise.
#
# **Les treize sont toutes utilisées à l'entraînement.** Leur contribution
# respective est établie par le modèle et l'analyse SHAP, pas présumée ici.
# --------------------------------------------------------------------------- #

FEATURES_NUMERIQUES = ["Rainfall_mm", "Temperature_Celsius"]
FEATURES_BOOLEENNES = ["Fertilizer_Used", "Irrigation_Used", "low_rain_no_irrigation"]
FEATURES_CATEGORIELLES = ["Region", "Soil_Type", "Crop", "Weather_Condition"]

# Variables de référence issues du jeu FAO, constantes par culture.
# À garder en tête pour SHAP : `Crop` les détermine entièrement.
FEATURES_FAO = [
    "crop_productivity_index",
    "crop_yield_geo_cv",
    "crop_yield_trend_decade",
    "crop_in_fao_reference",
]

#: Les 13 variables explicatives du dataset consolidé
FEATURES_MODELE = (
    FEATURES_NUMERIQUES + FEATURES_BOOLEENNES + FEATURES_CATEGORIELLES + FEATURES_FAO
)

NUMERIC_COLUMNS = [*NUMERIC_FEATURES, TARGET]  # quantitatives + cible, pour l'EDA

# Correspondance des vocabulaires de cultures entre les deux jeux
CROP_MAPPING = {
    "Maize": "Maize",
    "Wheat": "Wheat",
    "Rice": "Rice, paddy",
    "Soybean": "Soybeans",
}

# Cultures du jeu A sans équivalent dans le jeu B
CROPS_WITHOUT_REFERENCE = ["Barley", "Cotton"]

# Seuil de faible pluviométrie : tiers inférieur de la plage observée
# (100-1000 mm), ordre de grandeur du seuil semi-aride usuel.
# Partagé par `data_preparation.ipynb` et le transformateur de l'API : les deux
# doivent lire la même valeur, sinon `low_rain_no_irrigation` diverge.
SEUIL_FAIBLE_PLUIE_MM = 400

# Conversion des rendements FAO : hectogrammes/ha -> tonnes/ha
HG_PER_HA_TO_T_PER_HA = 10_000

# --------------------------------------------------------------------------- #
# Suivi d'expériences et artefacts de modèle
# --------------------------------------------------------------------------- #

# MLflow 3 a placé le backend fichier (`./mlruns`) en mode maintenance et refuse
# de l'utiliser sans dérogation : le backend recommandé est une base SQLite.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

# Les artefacts (modèles, signatures) ne tiennent pas dans la base : ils vont
# sur disque, dans un dossier non versionné.
MLARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"

# Une seule expérience : les quatre modèles sur les treize variables.
EXPERIMENT_NOM = "agritech-rendement"

# Nom sous lequel les modèles sont publiés au **registre** MLflow. Un seul nom,
# une version par modèle entraîné : le registre représente l'emplacement unique
# « le modèle de rendement en service », et les versions en sont les candidats.
REGISTERED_MODEL_NAME = "agritech-rendement"

# MLflow 3 a supprimé les *stages* (`Staging`, `Production`) au profit des
# **alias**. L'API chargera `models:/agritech-rendement@champion` : changer de
# modèle en production reviendra à déplacer cet alias, sans toucher à son code
# ni redéployer quoi que ce soit.
ALIAS_CHAMPION = "champion"

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "modele_final.joblib"

# Budget par étude Optuna ; ~2 s par ajustement sur 700 000 lignes
OPTUNA_N_TRIALS = 100

# --------------------------------------------------------------------------- #
# Hypothèses économiques
#
# Aucun des deux jeux de données ne contient de prix ni de coût : ces valeurs
# sont des **hypothèses explicites**, pas des prix de marché. Elles servent à
# traduire une erreur de rendement en euros par hectare. La conclusion du
# rapport est vérifiée sur toute la plage `PRIX_SENSIBILITE`, de sorte qu'elle
# ne dépende pas du prix retenu.
# --------------------------------------------------------------------------- #

PRIX_TONNE_EUR = 200
PRIX_SENSIBILITE = (150, 200, 300)

# Surestimer pousse à sur-investir en intrants et en stockage : perte sèche.
# Sous-estimer fait manquer une opportunité. Pondérations du coût asymétrique.
COUT_SURESTIMATION = 1.5
COUT_SOUSESTIMATION = 1.0

# Identifiant des hypothèses ci-dessus, journalisé dans MLflow : deux runs
# calculés sous des hypothèses différentes ne sont pas comparables.
HYPOTHESES_ECONOMIQUES_VERSION = "v1-2026-10"

# --------------------------------------------------------------------------- #
# Palette graphique
# --------------------------------------------------------------------------- #

BLUE = "#2b6cb0"    # jeu A
GREEN = "#2f855a"   # jeu B et sources brutes
ORANGE = "#c05621"  # mises en avant

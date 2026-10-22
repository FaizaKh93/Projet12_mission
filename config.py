"""Configuration du projet Agritech Answers.

Ce fichier ne contient que des **valeurs** : chemins, graine aléatoire,
vocabulaire du domaine et palette graphique. Aucune fonction — les fonctions
d'analyse sont définies dans les notebooks qui les utilisent.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

# Ce fichier est à la racine du projet : son dossier parent EST la racine
PROJECT_ROOT = Path(__file__).resolve().parent

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

# Conversion des rendements FAO : hectogrammes/ha -> tonnes/ha
HG_PER_HA_TO_T_PER_HA = 10_000

# --------------------------------------------------------------------------- #
# Palette graphique
# --------------------------------------------------------------------------- #

BLUE = "#2b6cb0"    # jeu A
GREEN = "#2f855a"   # jeu B et sources brutes
ORANGE = "#c05621"  # mises en avant

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

# --------------------------------------------------------------------------- #
# Contrat d'entrée de l'API
#
# Ces valeurs sont **observées sur `train`**, pas décrétées. Un test de la
# suite vérifie qu'elles coïncident avec les modalités de l'encodeur ajusté :
# une mise à jour du modèle qui changerait le vocabulaire ferait échouer les
# tests au lieu de laisser le contrat diverger en silence.
# --------------------------------------------------------------------------- #

# L'API doit **rejeter** une modalité inconnue, pas l'accepter. Le
# transformateur substitue une sentinelle aux champs facultatifs *non
# renseignés* ; si une valeur invalide lui parvenait, elle serait traitée comme
# une information absente — une faute de frappe deviendrait une omission
# silencieuse.
MODALITES_CATEGORIELLES = {
    "Crop": ("Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"),
    "Region": ("East", "North", "South", "West"),
    "Soil_Type": ("Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"),
    "Weather_Condition": ("Cloudy", "Rainy", "Sunny"),
}

# Domaine de validité du modèle. Au-delà, Ridge **extrapole linéairement**
# sans aucun garde-fou : rien ne plafonne la prédiction, et l'erreur n'est plus
# celle qui a été mesurée. L'API avertit donc, mais ne refuse pas — une saison
# plus humide que toutes celles observées reste une question légitime.
DOMAINE_ENTRAINEMENT = {
    "Rainfall_mm": (100.0, 1000.0),
    "Temperature_Celsius": (15.0, 40.0),
}

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

# Nom sous lequel les modèles **entraînés** sont publiés au registre MLflow :
# une version par modèle étudié à l'étape 2, et l'alias `champion` sur celui
# que la comparaison multicritère a retenu. Ces versions attendent toutes les
# 13 colonnes du dataset consolidé.
REGISTERED_MODEL_NAME = "agritech-rendement"

# L'artefact **de service** est publié sous un second nom, parce qu'il n'a pas
# le même contrat d'entrée : il n'attend que les 8 saisies de l'agriculteur, le
# transformateur de dérivation reconstituant les 5 autres. Les réunir sous un
# même nom donnerait un registre dont les versions ne sont pas interchangeables.
# Séparés, chaque nom garde un contrat stable et son alias `champion` désigne
# une décision distincte : le modèle retenu par la comparaison d'un côté,
# l'artefact servi par l'API de l'autre.
REGISTERED_MODEL_API = "agritech-rendement-api"

# MLflow 3 a supprimé les *stages* (`Staging`, `Production`) au profit des
# **alias**. Déplacer un alias suffit à changer ce qu'il désigne, sans toucher
# au code qui le charge ni redéployer quoi que ce soit.
ALIAS_CHAMPION = "champion"

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "modele_final.joblib"

# Artefact déployé : le Pipeline de service, gelé, dans un fichier autonome.
# C'est lui que l'image Docker embarque. Le conteneur ne consulte pas le
# registre à son démarrage — il sert un fichier figé au moment de la
# construction de l'image, ce qui le rend sans état et démarrable sans réseau.
API_MODEL_PATH = MODELS_DIR / "modele_api.joblib"

# Provenance de l'artefact ci-dessus, écrite à côté de lui par le script
# d'export. Elle existe parce que l'image n'embarque pas mlflow : sans ce
# fichier, le conteneur servirait un modèle sans pouvoir dire d'où il vient.
# C'est la source de `/model/info` quand le modèle ne vient pas du registre.
API_METADATA_PATH = MODELS_DIR / "modele_api.json"

# Demi-largeur de l'**intervalle de prédiction** à 95 % — l'intervalle qui doit
# couvrir le rendement d'une parcelle nouvelle, et non un intervalle de
# confiance autour d'une moyenne : les deux ne mesurent pas la même chose.
#
# Obtenue par les **quantiles empiriques des résidus sur `val`** (−0,9792 et
# +0,9785), pas par une formule gaussienne. Couverture mesurée sur `test`, qui
# n'entre dans aucun calcul de calibration : **94,92 %** pour 95 % nominal.
# Elle tient par culture (94,73 % à 95,01 %, étendue 0,28 point), aux déciles
# extrêmes de pluie et de température, et sur les parcelles sèches sans
# intrants (95,27 %).
#
# Une seule largeur suffit parce que la variance des résidus est constante :
# l'écart-type par décile de prédiction varie de 1,5 %. La borne basse est
# tronquée à zéro, un rendement ne pouvant être négatif.
#
# **Réserve.** Les données sont simulées avec un bruit gaussien ; cet
# intervalle en mesure fidèlement la dispersion. Sur des parcelles réelles, il
# devrait être recalibré, et la normalité parfaite des résidus ne s'y
# observerait pas.
DEMI_INTERVALLE_PREDICTION_95_T_HA = 0.979

# Seuil au-dessous duquel un écart de rendement entre cultures n'est pas
# exploitable pour décider. C'est une **convention métier annoncée**, pas un
# seuil de significativité : elle ne se déduit ni du RMSE, ni de l'intervalle
# de prédiction.
#
# Pourquoi 0,1 t/ha, soit 100 kg/ha :
#   - c'est l'ordre de grandeur de ce qu'une pesée de récolte réelle peut
#     distinguer, une fois comptées les variations d'humidité, les pertes au
#     battage et l'hétérogénéité intraparcellaire ;
#   - valorisé aux prix à la production observés (213 à 508 €/t pour nos six
#     cultures), cela représente 21 à 51 €/ha — en dessous de la précision de
#     n'importe quel budget prévisionnel.
#
# Attention à ce qu'elle ne dit PAS : deux prédictions séparées par moins que
# ce seuil ne sont pas « statistiquement indiscernables ». La significativité
# des écarts entre cultures **n'a pas été évaluée formellement** dans ce
# projet, et ce seuil n'en tient pas lieu : il borne la portée *pratique*
# d'un écart, pas sa détectabilité.
#
# L'évaluer demanderait la covariance des coefficients sous pénalisation
# Ridge, dans un plan où les 4 indicateurs FAO sont des fonctions exactes de
# `Crop`, donc colinéaires au bloc one-hot. Une erreur-type OLS sur deux
# groupes n'y répond pas, et le seuil ci-dessus ne prétend pas y répondre.
SEUIL_ECART_RENDEMENT_PERTINENT_T_HA = 0.1

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
# Prix de référence à la production
#
# Les deux jeux de données du projet ne contiennent aucun prix. Ces valeurs
# viennent donc de l'extérieur : FAOSTAT, domaine *Producer Prices*, élément
# 5532 « Producer Price (USD/tonne) », année 2024, converti au taux annuel
# moyen BCE.
#
# **Ce qu'elles sont** : une médiane entre pays déclarants, destinée à
# pré-remplir le champ `price_eur_t` des budgets de `/recommend` plutôt que de
# laisser l'utilisateur devant une case vide.
#
# **Ce qu'elles ne sont pas** : le prix de la parcelle de l'utilisateur. Les
# quartiles sont conservés précisément pour que cela se voie — l'écart
# interquartile du maïs va de 186 à 432 €/t, un facteur 2,3, bien plus large
# que la variation de ±20 % testée par `/recommend`.
#
# **Elles ne suffisent pas à calculer une marge** : les charges opérationnelles
# manquent, et c'est délibéré. Prix mondiaux FAO combinés à des charges
# européennes donnent des marges négatives sur trois cultures sur quatre.
#
# `scripts/build_reference_prices.py` les re-dérive depuis un export FAOSTAT et
# vérifie qu'elles n'ont pas dérivé.
# --------------------------------------------------------------------------- #

PRIX_REFERENCE_SOURCE = {
    "source": "FAOSTAT",
    "jeu_de_donnees": "Producer Prices (PP), All Data (Normalized)",
    "element": "Producer Price (USD/tonne) — code 5532",
    "annee": 2024,
    "methode": "médiane des pays déclarants, agrégats exclus",
    "conversion": "USD vers EUR au taux annuel moyen BCE 2024",
    "taux_usd_par_eur": 1.08238,
    "extrait_le": "2026-10-28",
}

#: Prix médian et quartiles, en euros par tonne. `item_faostat` est orthographié
#: **comme dans les données**, qui diffèrent du fichier de codes : celui-ci
#: écrit « Seed cotton; unginned » avec un point-virgule là où les données
#: mettent une virgule. Un filtre construit sur le fichier de codes ne renvoie
#: aucune ligne — sans lever d'erreur.
PRIX_REFERENCE_FAOSTAT = {
    "Barley": {"prix_eur_t": 213, "q1_eur_t": 171, "q3_eur_t": 284,
               "pays": 60, "item_faostat": "Barley"},
    "Cotton": {"prix_eur_t": 508, "q1_eur_t": 449, "q3_eur_t": 655,
               "pays": 11, "item_faostat": "Seed cotton, unginned"},
    "Maize": {"prix_eur_t": 267, "q1_eur_t": 186, "q3_eur_t": 432,
              "pays": 75, "item_faostat": "Maize (corn)"},
    "Rice": {"prix_eur_t": 423, "q1_eur_t": 353, "q3_eur_t": 695,
             "pays": 45, "item_faostat": "Rice"},
    "Soybean": {"prix_eur_t": 450, "q1_eur_t": 409, "q3_eur_t": 690,
                "pays": 36, "item_faostat": "Soya beans"},
    "Wheat": {"prix_eur_t": 225, "q1_eur_t": 190, "q3_eur_t": 328,
              "pays": 67, "item_faostat": "Wheat"},
}

# --------------------------------------------------------------------------- #
# Palette graphique
# --------------------------------------------------------------------------- #

BLUE = "#2b6cb0"    # jeu A
GREEN = "#2f855a"   # jeu B et sources brutes
ORANGE = "#c05621"  # mises en avant

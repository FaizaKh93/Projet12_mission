"""Reconstruction des variables dérivées, entre les saisies et le modèle.

**Le problème que ce module résout.** Le Pipeline entraîné à l'étape 2 attend
les 13 variables du dataset consolidé. Un agriculteur n'en connaît que 8 : les
5 autres sont *calculées* à partir des premières. Sans ce transformateur, un
appel d'API échoue sur `columns are missing`.

Les 5 variables dérivées ne sont donc **pas** des champs de saisie. Les exposer
à l'utilisateur permettrait de déclarer une parcelle impossible — 800 mm de
pluie, irriguée, et pourtant en situation de faible pluie sans irrigation. Ce
n'est pas une crainte théorique : l'étape 2 a produit exactement ce bug en
changeant `Crop` sans mettre à jour les indicateurs FAO, ce qui faussait de
24 % la mesure centrale de la recommandation.

**Pourquoi un transformateur plutôt qu'une fonction.** Ajusté sur `train`, il
mémorise la table FAO dans son état et part avec le modèle dans le `.joblib`.
L'API ne réimplémente rien, et la règle de dérivation ne peut pas diverger de
celle qui a servi à l'entraînement.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from agritech.config import (
    FEATURES_BOOLEENNES,
    FEATURES_CATEGORIELLES,
    FEATURES_FAO,
    FEATURES_MODELE,
    FEATURES_NUMERIQUES,
    SEUIL_FAIBLE_PLUIE_MM,
)

#: Valeur substituée à une catégorielle facultative non renseignée.
#:
#: `OneHotEncoder(handle_unknown="ignore")` ne connaît pas cette modalité et met
#: donc tout le bloc à zéro : la variable ne contribue simplement pas. Mesuré
#: sur 5 000 parcelles, l'écart vaut au plus 0,0017 t/ha avec les trois champs
#: absents — 0,34 % du RMSE — et le biais moyen est indiscernable de zéro.
#:
#: Cette équivalence avec la marginalisation tient parce que les modalités sont
#: équilibrées dans ce jeu simulé. Sur des données réelles déséquilibrées, elle
#: serait à remesurer.
SENTINELLE = "__INCONNU__"

#: Les 3 catégorielles dont l'ablation ne montre aucun apport (z = −0,6, −0,5,
#: +0,8). `Region` pose en plus un problème métier : North/South/East/West ne
#: correspondent à aucune géographie, un agriculteur ne peut pas y répondre.
FEATURES_FACULTATIVES = ["Region", "Soil_Type", "Weather_Condition"]

#: Ce que l'appelant doit fournir. `Crop` en fait partie pour `/predict` ; pour
#: `/recommend`, l'API la fait varier elle-même sur les six cultures.
FEATURES_OBLIGATOIRES = [
    *FEATURES_NUMERIQUES,                    # Rainfall_mm, Temperature_Celsius
    "Fertilizer_Used", "Irrigation_Used",    # les deux leviers actionnables
    "Crop",
]

#: Les 8 saisies, obligatoires et facultatives confondues.
FEATURES_SAISIES = FEATURES_OBLIGATOIRES + FEATURES_FACULTATIVES


class DerivationVariables(BaseEstimator, TransformerMixin):
    """Complète les 8 saisies en les 13 colonnes attendues par le modèle.

    **Asymétrie assumée entre `fit` et `transform`.** `fit` reçoit les données
    d'entraînement complètes, dont il extrait la table FAO ; `transform` reçoit
    les seules saisies. C'est inhabituel pour un transformateur scikit-learn,
    mais c'est la nature du problème : on apprend une correspondance sur des
    données qui la contiennent, pour la rejouer sur des données qui ne l'ont
    pas.

    Les cinq variables reconstruites :

    - `low_rain_no_irrigation` — pluie sous le seuil **et** parcelle non
      irriguée. La formule a été vérifiée exacte sur les 999 769 lignes du
      dataset consolidé ;
    - les 4 indicateurs FAO — déterminés par la culture seule, donc une simple
      table de correspondance.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Mémorise la table FAO, une ligne par culture.

        L'orge et le coton sont absents du référentiel FAO : leurs trois
        indicateurs numériques valent `NaN`, que l'imputeur du Pipeline
        remplacera par la médiane apprise sur `train`. Y substituer un zéro
        fabriquerait des prédictions silencieusement fausses pour un tiers des
        cultures.
        """
        manquantes = [c for c in ["Crop", *FEATURES_FAO] if c not in X.columns]
        if manquantes:
            raise ValueError(
                f"`fit` attend les données d'entraînement complètes. "
                f"Colonnes absentes : {manquantes}"
            )

        table = (X[["Crop", *FEATURES_FAO]]
                 .drop_duplicates("Crop")
                 .set_index("Crop")
                 .sort_index())

        # Contrôle du présupposé : si une culture portait deux jeux
        # d'indicateurs, la table serait arbitraire et le modèle incohérent.
        varie = X.groupby("Crop", observed=True)[FEATURES_FAO].nunique(dropna=False)
        if (varie > 1).any().any():
            coupables = varie.columns[(varie > 1).any()].tolist()
            raise ValueError(
                f"Les indicateurs FAO ne sont pas constants par culture : {coupables}"
            )

        # pandas 3 adosse ses chaînes à pyarrow dès que pyarrow est installé.
        # Les deux axes de cette table — l'index des cultures et les noms de
        # colonnes — seraient alors des tableaux pyarrow, et le pickle en
        # porterait la référence : l'artefact exigerait à la relecture un
        # paquet que l'image de l'API n'installe pas. Le défaut ne se voit pas
        # sur le poste de développement, seulement dans le conteneur.
        #
        # En `object`, le fichier devient portable quel que soit
        # l'environnement qui l'a produit — ce qui compte autant que la
        # légèreté de l'image : la dépendance, sinon, serait implicite.
        table.index = table.index.astype(object)
        table.columns = table.columns.astype(object)

        self.table_fao_ = table
        self.cultures_connues_ = list(table.index)
        self.feature_names_in_ = list(FEATURES_SAISIES)
        self.n_features_in_ = len(FEATURES_SAISIES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Renvoie les 13 colonnes, dans l'ordre et les types de l'entraînement."""
        manquantes = [c for c in FEATURES_OBLIGATOIRES if c not in X.columns]
        if manquantes:
            raise ValueError(f"Saisies obligatoires absentes : {manquantes}")

        inconnues = set(X["Crop"].dropna().unique()) - set(self.cultures_connues_)
        if inconnues:
            raise ValueError(
                f"Culture inconnue : {sorted(inconnues)}. "
                f"Attendu : {self.cultures_connues_}"
            )

        sortie = X.copy()

        # Champ non fourni -> sentinelle. Une valeur fournie mais invalide doit
        # être rejetée en amont par la validation de l'API : l'accepter ici la
        # transformerait silencieusement en « information absente ».
        for colonne in FEATURES_FACULTATIVES:
            if colonne not in sortie.columns:
                sortie[colonne] = SENTINELLE
            else:
                sortie[colonne] = sortie[colonne].fillna(SENTINELLE)

        sortie["Irrigation_Used"] = sortie["Irrigation_Used"].astype(bool)
        sortie["Fertilizer_Used"] = sortie["Fertilizer_Used"].astype(bool)

        # Situation de stress hydrique : peu de pluie et aucun apport d'eau.
        sortie["low_rain_no_irrigation"] = (
            (sortie["Rainfall_mm"].astype(float) < SEUIL_FAIBLE_PLUIE_MM)
            & ~sortie["Irrigation_Used"]
        )

        # `reindex` sur la culture préserve les NaN de l'orge et du coton.
        indicateurs = self.table_fao_.reindex(sortie["Crop"].to_numpy())
        for colonne in FEATURES_FAO:
            sortie[colonne] = indicateurs[colonne].to_numpy()

        return self._typer(sortie)[FEATURES_MODELE]

    @staticmethod
    def _typer(X: pd.DataFrame) -> pd.DataFrame:
        """Aligne les types sur ceux vus à l'entraînement.

        Le `ColumnTransformer` route les colonnes par nom, pas par type, mais
        un booléen arrivant en `object` ou un flottant en `int` change le
        résultat de la standardisation et de l'encodage.
        """
        quantitatives = FEATURES_NUMERIQUES + [c for c in FEATURES_FAO
                                               if c != "crop_in_fao_reference"]
        for colonne in quantitatives:
            X[colonne] = X[colonne].astype("float64")
        for colonne in FEATURES_BOOLEENNES + ["crop_in_fao_reference"]:
            X[colonne] = X[colonne].astype(bool)
        for colonne in FEATURES_CATEGORIELLES:
            X[colonne] = X[colonne].astype(str)
        return X

    def get_feature_names_out(self, input_features=None):
        """Les 13 colonnes produites, dans l'ordre de l'entraînement.

        Exigée par l'API scikit-learn : c'est elle qui permet au
        `ColumnTransformer` en aval de router les colonnes par leur nom.
        """
        return list(FEATURES_MODELE)

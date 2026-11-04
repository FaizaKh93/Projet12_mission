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

**Deux chaînes, donc deux transformateurs.** `DerivationVariables` complète les
8 saisies de parcelle en les 13 colonnes du dataset consolidé.
`DerivationPanel`, plus bas, complète le couple (pays, culture) en les 8
variables du panel FAO. Le problème est le même, la méthode non : les retards
d'une série nationale ne se déduisent d'aucune saisie, ils se lisent dans
l'historique embarqué.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from agritech.config import (
    FEATURES_BOOLEENNES,
    FEATURES_CATEGORIELLES,
    FEATURES_FAO,
    FEATURES_MODELE,
    FEATURES_NUMERIQUES,
    PANEL_COVARIATES,
    PANEL_FEATURES_MODELE,
    PANEL_FEATURES_PREVISION,
    PANEL_TARGET,
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


# --------------------------------------------------------------------------- #
# Chaîne `real` : du couple (pays, culture) aux 8 variables du panel
# --------------------------------------------------------------------------- #

#: Ce que l'appelant du service national fournit, et rien de plus. Il ne
#: saisit **aucun historique** : les retards se cherchent, ils ne se déclarent
#: pas.
PANEL_SAISIES = ["Area", "Item"]


class DerivationPanel(BaseEstimator, TransformerMixin):
    """Complète le couple (pays, culture) en les 8 variables du panel.

    Même problème que `DerivationVariables` — le modèle attend plus de colonnes
    que l'appelant n'en connaît — mais résolu par une **recherche** plutôt que
    par un calcul. Les huit variables se répartissent en trois familles :

    - `Area` et `Item` sont les deux saisies, reprises telles quelles ;
    - les trois retards sont lus dans l'historique du panel ;
    - les trois covariables portent sur l'année à prédire, donc inconnues avant
      la récolte : elles partent à `NaN`, et l'imputeur du Pipeline y substitue
      la médiane de `train`. Mesuré sur `test`, cela déplace la prédiction de
      0,037 t/ha en moyenne et de 0,368 t/ha au pire — pour une contribution
      SHAP de 1,1 %. Les présenter comme des conditions observées serait faux :
      aucune ne l'est.

    **Le décalage d'un cran.** Pour prévoir l'année suivant l'année de
    référence, `lag1` est le *rendement observé* de l'année de référence, et
    `lag2`, `lag3` sont les deux premiers retards de cette même ligne. Reprendre
    la ligne telle quelle prédirait l'année de référence elle-même. Un contrôle
    de l'export vérifie l'égalité exacte des retards reconstitués avec ceux que
    le dataset porte pour l'année suivante.

    **Pourquoi `fit` reçoit le panel complet, et non `train`.** Les retards sont
    une **entrée de service**, pas une donnée d'apprentissage : servir les
    derniers relevés connus est exactement ce que ferait une application. Le
    modèle, lui, n'est pas réajusté et n'a jamais vu au-delà de 2002. Ajuster
    cette table sur `train` ferait servir des retards vieux de quatorze ans.

    **Les séries interrompues ne sont pas servies.** Un couple dont le dernier
    relevé est antérieur à l'année de référence est écarté : son `lag1` ne
    serait pas le rendement de l'an dernier. Mesuré sur le panel, 72 couples
    sur 1 178 sont dans ce cas, et 8 pays y perdent toutes leurs cultures.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Mémorise les derniers retards connus de chaque série servable."""
        attendues = [*PANEL_SAISIES, "Year", "lag1", "lag2", PANEL_TARGET]
        manquantes = [colonne for colonne in attendues
                      if colonne not in X.columns]
        if manquantes:
            raise ValueError(
                f"`fit` attend le panel complet, retards compris. "
                f"Colonnes absentes : {manquantes}"
            )

        annee = int(X["Year"].max())
        derniere = (X.sort_values("Year")
                     .groupby(PANEL_SAISIES, sort=False).tail(1))
        courante = derniere[derniere["Year"] == annee]

        # Le décalage d'un cran : le relevé de l'année de référence devient
        # `lag1`, et les deux premiers retards de cette ligne glissent d'un
        # rang. Les noms viennent de `config`, les sources sont en regard.
        sources = [PANEL_TARGET, "lag1", "lag2"]
        table = pd.DataFrame(
            {retard: courante[source].to_numpy(dtype="float64")
             for retard, source in zip(PANEL_FEATURES_PREVISION, sources)},
            index=pd.MultiIndex.from_arrays(
                [courante[colonne].to_numpy(dtype=object)
                 for colonne in PANEL_SAISIES],
                names=PANEL_SAISIES),
        )
        # Un retard manquant signale une série trop courte ou trouée : le
        # couple sort de la table plutôt que d'être servi avec une imputation
        # qui ferait passer une médiane mondiale pour son propre passé.
        table = table.dropna().sort_index()

        # Même précaution que la table FAO de `DerivationVariables`, et elle
        # demande ici un geste de plus : `from_arrays` **ré-infère** les
        # niveaux en chaînes adossées à pyarrow, même nourri de tableaux
        # `object`. Le pickle en porterait la référence et l'artefact
        # exigerait, à la relecture, un paquet que le groupe `api` du lock
        # n'installe pas — vérifié : sans ce `set_levels`, `joblib.load`
        # échoue sur `ImportError: pyarrow` dans l'environnement de l'image.
        table.index = table.index.set_levels(
            [niveau.astype(object) for niveau in table.index.levels])
        table.columns = table.columns.astype(object)

        self.table_retards_ = table
        self.annee_reference_ = annee
        self.annee_predite_ = annee + 1
        self.couples_servables_ = [tuple(cle) for cle in table.index]
        self.feature_names_in_ = list(PANEL_SAISIES)
        self.n_features_in_ = len(PANEL_SAISIES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Renvoie les 8 colonnes, dans l'ordre et les types de l'entraînement."""
        manquantes = [colonne for colonne in PANEL_SAISIES
                      if colonne not in X.columns]
        if manquantes:
            raise ValueError(f"Saisies absentes : {manquantes}")

        cles = pd.MultiIndex.from_arrays(
            [X[colonne].to_numpy(dtype=object) for colonne in PANEL_SAISIES],
            names=PANEL_SAISIES)

        inconnus = [tuple(cle)
                    for cle in cles.difference(self.table_retards_.index)]
        if inconnus:
            raise ValueError(
                f"Couple (pays, culture) non servable : {inconnus[:5]}"
                f"{' …' if len(inconnus) > 5 else ''}. La table ne porte que "
                f"les {len(self.couples_servables_)} couples dont le dernier "
                f"relevé du panel est {self.annee_reference_} : un historique "
                f"interrompu avant cette année ne permet pas de prévoir "
                f"{self.annee_predite_}."
            )

        retards = self.table_retards_.reindex(cles)
        sortie = pd.DataFrame({colonne: X[colonne].astype(str).to_numpy()
                               for colonne in PANEL_SAISIES})
        for colonne in PANEL_FEATURES_PREVISION:
            sortie[colonne] = retards[colonne].to_numpy(dtype="float64")
        for colonne in PANEL_COVARIATES:
            sortie[colonne] = np.full(len(sortie), np.nan, dtype="float64")

        return sortie[PANEL_FEATURES_MODELE]

    def retards(self, area: str, item: str) -> dict[str, float]:
        """Les trois retards servis pour ce couple.

        L'API les rend dans sa réponse, et ce n'est pas de la décoration : sur
        une série erratique, la prévision s'écarte beaucoup du dernier relevé —
        le soja de Zambie passe de 0,19 à 0,81 t/ha — parce que le modèle
        revient vers la moyenne des trois années. L'historique affiché est ce
        qui rend ce saut lisible.

        L'année de chaque retard se déduit de `annee_reference_` : `lag1` est
        cette année-là, `lag2` la précédente, `lag3` celle d'avant.
        """
        if (area, item) not in self.table_retards_.index:
            raise KeyError(f"Couple non servable : {(area, item)}")
        ligne = self.table_retards_.loc[(area, item)]
        return {retard: float(ligne[retard])
                for retard in PANEL_FEATURES_PREVISION}

    def cultures_par_pays(self) -> dict[str, list[str]]:
        """Les cultures servables de chaque pays, par ordre alphabétique.

        Source du vocabulaire de l'interface : elle ne propose que des couples
        que le service sait prédire, au lieu de recopier une liste de cultures
        qui finirait par diverger de la table embarquée.
        """
        par_pays: dict[str, list[str]] = {}
        for area, item in self.couples_servables_:
            par_pays.setdefault(area, []).append(item)
        return {area: sorted(cultures)
                for area, cultures in sorted(par_pays.items())}

    def get_feature_names_out(self, input_features=None):
        """Les 8 colonnes produites, dans l'ordre de l'entraînement."""
        return list(PANEL_FEATURES_MODELE)

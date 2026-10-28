"""Contrat d'entrée et de sortie de l'API.

Ce module ne connaît ni le modèle, ni MLflow, ni le disque : il décrit ce qu'un
appelant envoie et ce qu'il reçoit. Les règles **métier** — reconstruction des
variables dérivées, rejet d'une culture inconnue, calcul des marges — vivent
ailleurs ; les règles de **forme** vivent ici.

**Deux vocabulaires, une seule correspondance.** Les champs JSON sont en
anglais, minuscules et tirets bas ; les variables du modèle portent les noms du
dataset (`Rainfall_mm`). Choisir l'anglais ne supprime pas la correspondance,
cela la rend triviale : elle est dérivée par `.lower()`, centralisée dans
`CHAMP_VERS_VARIABLE`, et un test vérifie qu'elle couvre exactement les saisies
attendues par le transformateur.

**Ce que le contrat refuse, et pourquoi il doit le refuser.** Une modalité
invalide et un champ inconnu sont des erreurs, pas des silences. C'est une
exigence du transformateur : il substitue une sentinelle aux facultatives *non
renseignées*. Un `soil_type` mal orthographié qui passerait ici deviendrait
une « information absente » sans que personne en soit averti.

**Ce que le contrat rend impossible de taire.** Quatre états d'information
insuffisante sont des valeurs de première classe — `not_calculable`,
`not_assessed`, `undetermined`, `out_of_domain`. Une réponse ne peut pas
présenter un classement partiel comme exhaustif, ni une absence d'évaluation
comme un avis favorable : la structure l'interdit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agritech.config import DOMAINE_ENTRAINEMENT, MODALITES_CATEGORIELLES
from agritech.preprocessing import FEATURES_SAISIES

#: Du champ JSON vers la variable du modèle. **Dérivée, pas écrite** : une
#: table recopiée à la main serait un endroit de plus où une divergence peut
#: s'installer.
CHAMP_VERS_VARIABLE = {variable.lower(): variable for variable in FEATURES_SAISIES}

# `Literal[tuple]` équivaut à `Literal["a", "b"]` : l'énumération se retrouve
# dans l'OpenAPI, et le message d'erreur cite les valeurs acceptées.
Culture = Literal[MODALITES_CATEGORIELLES["Crop"]]
Region = Literal[MODALITES_CATEGORIELLES["Region"]]
TypeDeSol = Literal[MODALITES_CATEGORIELLES["Soil_Type"]]
Meteo = Literal[MODALITES_CATEGORIELLES["Weather_Condition"]]

# Bornes de **recevabilité**, à ne pas confondre avec le domaine
# d'entraînement : elles écartent l'absurde (pluviométrie négative, 300 °C),
# là où le domaine ne déclenche qu'un avertissement. Larges à dessein —
# refuser 1 200 mm parce que `train` s'arrête à 1 000 serait refuser une
# question légitime.
PLUIE_RECEVABLE = (0.0, 5_000.0)
TEMPERATURE_RECEVABLE = (-20.0, 60.0)

#: Champs facultatifs du profil de parcelle.
CHAMPS_FACULTATIFS = ("region", "soil_type", "weather_condition")


def _domaine(variable: str) -> str:
    """Rappelle le domaine observé dans la description du champ."""
    bas, haut = DOMAINE_ENTRAINEMENT[variable]
    return f"Domaine observé à l'entraînement : {bas:g} à {haut:g}."


# --------------------------------------------------------------------------- #
# Entrées
# --------------------------------------------------------------------------- #


class ContexteParcelle(BaseModel):
    """Les conditions de culture, sans la culture.

    Partie commune aux deux endpoints : `/recommend` fait varier la culture
    lui-même, l'appelant ne la fournit donc pas.
    """

    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(
        ..., ge=PLUIE_RECEVABLE[0], le=PLUIE_RECEVABLE[1], examples=[500.0],
        description="Pluviométrie de la saison, en millimètres. "
                    + _domaine("Rainfall_mm"))
    temperature_celsius: float = Field(
        ..., ge=TEMPERATURE_RECEVABLE[0], le=TEMPERATURE_RECEVABLE[1],
        examples=[25.0],
        description="Température moyenne de la saison, en degrés Celsius. "
                    + _domaine("Temperature_Celsius"))
    fertilizer_used: bool = Field(
        ..., examples=[True],
        description="Parcelle fertilisée. Levier actionnable : +1,50 t/ha.")
    irrigation_used: bool = Field(
        ..., examples=[False],
        description="Parcelle irriguée. Levier actionnable : +1,20 t/ha.")

    # Les trois facultatives. Leur ablation n'a montré aucun apport mesurable
    # (z = −0,6, −0,5, +0,8) : les omettre déplace la prédiction de 0,0017 t/ha
    # en moyenne, soit 0,34 % du RMSE. `Region` pose en plus un problème
    # métier — North/South/East/West ne correspond à aucune géographie.
    region: Region | None = Field(
        None, description="Facultatif. Sans effet mesurable sur la prédiction.")
    soil_type: TypeDeSol | None = Field(
        None, description="Facultatif. Sans effet mesurable sur la prédiction.")
    weather_condition: Meteo | None = Field(
        None, description="Facultatif. Sans effet mesurable sur la prédiction.")

    def vers_variables(self) -> dict[str, float | bool | str]:
        """Traduit les champs fournis en variables du modèle.

        Les facultatives laissées vides sont **omises**, pas transmises à
        `None` : c'est ainsi que le transformateur les reconnaît comme non
        renseignées et leur applique sa sentinelle.
        """
        fournis = self.model_dump(exclude_none=True, exclude={"budgets"})
        return {CHAMP_VERS_VARIABLE[champ]: valeur
                for champ, valeur in fournis.items()}

    def facultatives_omises(self) -> list[str]:
        """Les champs facultatifs que l'appelant n'a pas renseignés."""
        return [champ for champ in CHAMPS_FACULTATIFS
                if getattr(self, champ) is None]


class PredictionRequete(ContexteParcelle):
    """`/predict` — une culture précise, dans un contexte donné."""

    crop: Culture = Field(
        ..., examples=["Wheat"],
        description="Culture envisagée. La liste faisant foi est celle que "
                    "`/model/info` renvoie : c'est l'artefact qui la porte.")


class BudgetProduction(BaseModel):
    """Budget technico-économique d'une culture, dans le contexte de la requête.

    **Trois nombres, pas un de plus.** Le système de production n'est pas un
    champ du budget : `fertilizer_used` et `irrigation_used` le fixent déjà
    pour toute la requête. Le redire par culture permettrait de décrire une
    parcelle à la fois irriguée et pluviale — une incohérence qu'aucune
    validation ne rattraperait. Le produit commercialisé n'en est pas un non
    plus : la culture le désigne.

    Cette contrainte n'appauvrit pas la comparaison, elle la **fonde**. Une
    marge brute ignore les charges de structure, ce qui n'est légitime que si
    les cultures comparées partagent le même système. Ici c'est vrai par
    construction : les six sont évaluées dans le contexte unique de la requête.

    **Rendement et prix doivent se rapporter au même produit commercialisé.**
    Riz paddy ou riz blanchi, coton-graine ou coton-fibre : ces produits n'ont
    ni le même rendement ni le même prix, et les mélanger fausserait la marge
    sans qu'aucune validation puisse le détecter. C'est la seule cohérence qui
    reste à la charge de l'appelant.

    **Complet ou absent.** Un budget partiel est refusé plutôt que complété
    par des moyennes : une charge manquante remplacée par un ordre de grandeur
    produirait une marge fausse, présentée comme calculée.
    """

    model_config = ConfigDict(extra="forbid")

    expected_yield_t_ha: float = Field(
        ..., gt=0, le=50, examples=[6.0],
        description="Rendement attendu dans le contexte de la requête, en "
                    "tonnes par hectare. Celui de l'agriculteur, pas celui "
                    "du modèle.")
    variable_costs_eur_ha: float = Field(
        ..., ge=0, le=20_000, examples=[800.0],
        description="Charges **opérationnelles**, en euros "
                    "par hectare : semences, fertilisation, traitements, "
                    "irrigation, autres charges spécifiques à la culture. "
                    "Critère pour trancher : une charge qui disparaîtrait si "
                    "la culture était abandonnée. Les charges de structure — "
                    "matériel, foncier, main-d'œuvre permanente — en sont "
                    "exclues, et deux budgets doivent les compter pareil.")
    price_eur_t: float = Field(
        ..., gt=0, le=10_000, examples=[210.0],
        description="Prix de vente en euros par tonne, pour le **même produit "
                    "commercialisé** que le rendement ci-dessus.")


#: Corps d'exemple attaché à la requête de recommandation.
#:
#: Le schéma contraint déjà les clés de `budgets` par `propertyNames.enum`, et
#: la validation rejette bien une culture inconnue. Mais Swagger UI ignore
#: `propertyNames` quand il fabrique son exemple : il proposerait
#: `additionalProp1`, `additionalProp2`, `additionalProp3` — des noms qui
#: n'existent nulle part dans ce contrat. L'exemple ci-dessous remplace ces
#: bouche-trous par **les six cultures**, pour que « Try it out » montre le
#: classement complet sans qu'on ait à deviner la forme des clés.
#:
#: **Les prix sont réels** : médianes des prix à la production FAOSTAT 2024,
#: converties au taux de change annuel BCE 2024 (1,08238 USD/EUR). Les
#: rendements et les charges, eux, sont des **ordres de grandeur plausibles**
#: sans source — aucun jeu de données du projet n'en contient.
#:
#: Les six budgets décrivent le **même contexte que la requête** : fertilisé,
#: non irrigué. C'est ce qui rend les marges comparables, et c'est pourquoi le
#: riz y rend 2,5 t/ha et non 6 — un riz pluvial n'est pas un riz irrigué.
EXEMPLE_RECOMMANDATION = {
    "rainfall_mm": 500.0,
    "temperature_celsius": 25.0,
    "fertilizer_used": True,
    "irrigation_used": False,
    "region": "North",
    "soil_type": "Loam",
    "weather_condition": "Sunny",
    "budgets": {
        "Barley": {
            "expected_yield_t_ha": 4.0,
            "variable_costs_eur_ha": 420.0,
            "price_eur_t": 213.0,
        },
        "Cotton": {
            "expected_yield_t_ha": 1.8,
            "variable_costs_eur_ha": 700.0,
            "price_eur_t": 508.0,
        },
        "Maize": {
            "expected_yield_t_ha": 5.0,
            "variable_costs_eur_ha": 650.0,
            "price_eur_t": 267.0,
        },
        "Rice": {
            "expected_yield_t_ha": 2.5,
            "variable_costs_eur_ha": 600.0,
            "price_eur_t": 423.0,
        },
        "Soybean": {
            "expected_yield_t_ha": 2.5,
            "variable_costs_eur_ha": 400.0,
            "price_eur_t": 450.0,
        },
        "Wheat": {
            "expected_yield_t_ha": 4.0,
            "variable_costs_eur_ha": 500.0,
            "price_eur_t": 225.0,
        },
    },
}


class RecommandationRequete(ContexteParcelle):
    """`/recommend` — le contexte, et des budgets facultatifs par culture."""

    # `extra="forbid"` est répété plutôt qu'hérité : la configuration d'un modèle
    # Pydantic fusionne avec celle du parent, mais le redire ici rend la règle
    # lisible à l'endroit où on la cherche.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXEMPLE_RECOMMANDATION]})

    budgets: dict[Culture, BudgetProduction] = Field(
        default_factory=dict,
        description="Zéro à six entrées, clé = culture. Sans au moins deux "
                    "budgets comparables, le classement économique n'est pas "
                    "rendu et la recommandation reste indéterminée. "
                    "**Les valeurs de l'exemple sont illustratives** : aucun "
                    "des deux jeux de données du projet ne contient de prix "
                    "ni de charges, et ces ordres de grandeur ne proviennent "
                    "d'aucune référence agricole validée. Les marges sont "
                    "calculées à partir de vos seuls budgets.")


# --------------------------------------------------------------------------- #
# Sorties — blocs communs
# --------------------------------------------------------------------------- #


class IntervallePrediction(BaseModel):
    """Intervalle de **prédiction**, pas de confiance.

    Il couvre le rendement d'une parcelle nouvelle, non la moyenne d'une
    population : les deux ne mesurent pas la même chose, et c'est le premier
    qui intéresse un agriculteur. Calibré sur les quantiles empiriques des
    résidus, couverture vérifiée sur un jeu n'ayant servi à aucun calibrage.
    """

    lower_t_ha: float = Field(
        ..., description="Borne basse, tronquée à zéro : un rendement ne peut "
                         "être négatif.")
    upper_t_ha: float
    nominal_coverage: float = Field(0.95, description="Niveau visé.")


class ServedModel(BaseModel):
    """De quel modèle vient cette réponse.

    Estampiller chaque prédiction évite d'avoir à deviner, plus tard, quelle
    version a produit quel conseil.
    """

    registry: str
    version: int
    source: Literal["mlflow-registry", "bundled-joblib"] = Field(
        ..., description="Registre MLflow résolu au démarrage, ou artefact "
                         "embarqué dans l'image.")


class QualiteDonnees(BaseModel):
    """Ce qu'il faut savoir pour lire la réponse sans la surinterpréter.

    Toujours présent, dans les deux endpoints. Les limites d'un modèle ne sont
    pas une annexe : les taire reviendrait à présenter des effets mesurés sur
    des données simulées comme des résultats agronomiques.
    """

    training_data: Literal["simulated"] = Field(
        "simulated",
        description="Les données d'entraînement sont simulées. Les effets "
                    "chiffrés ne sont pas transposables à des parcelles "
                    "réelles.")
    data_version: str
    split_version: str
    economic_assumptions_version: str | None = Field(
        None, description="Renseigné seulement si des marges ont été "
                          "calculées.")
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Sorties — /predict
# --------------------------------------------------------------------------- #


class PredictionReponse(BaseModel):
    """Le rendement attendu, et ce qui permet de le lire correctement.

    **Aucune traduction monétaire ici.** Un prix unique appliqué à six
    cultures n'aurait pas de sens agronomique : la dimension économique
    appartient à `/recommend`, où un budget par culture peut les distinguer.
    """

    yield_t_ha: float = Field(
        ..., description="Rendement estimé, en tonnes par hectare.")
    prediction_interval_95: IntervallePrediction
    data_quality: QualiteDonnees
    served_model: ServedModel


# --------------------------------------------------------------------------- #
# Sorties — /recommend
# --------------------------------------------------------------------------- #


class RendementCulture(BaseModel):
    """Le rendement prédit pour une culture, dans le contexte fourni."""

    crop: Culture
    yield_t_ha: float
    prediction_interval_95: IntervallePrediction


class DiagnosticAgronomique(BaseModel):
    """Si la culture est agronomiquement adaptée — ou si on l'ignore.

    `not_assessed` n'est **jamais** à lire comme `favorable`. Tant que le
    référentiel de faisabilité n'est pas audité, c'est l'état que toutes les
    cultures portent, avec son motif.
    """

    crop: Culture
    status: Literal["favorable", "constraint", "not_assessed"]
    criterion: str | None = Field(
        None, description="Critère appliqué, quand il l'a été.")
    source: str | None = Field(
        None, description="Référentiel d'où vient le critère.")
    reason: str | None = Field(
        None, description="Motif, obligatoire quand le statut est "
                          "`not_assessed`.")


class MargeCulture(BaseModel):
    """Marge brute **simplifiée** d'une culture, et le budget qui l'a produite.

    Le budget est renvoyé avec la marge, pas seulement la marge : un chiffre
    économique sans ses hypothèses n'est pas vérifiable.

    « Simplifiée » n'est pas une précaution de style. La marge brute des
    Chambres d'agriculture retranche les charges opérationnelles d'un *produit
    brut* qui peut comprendre les aides et les coproduits — la paille, par
    exemple — et les conventions varient d'une chambre à l'autre. Ici le
    produit brut se réduit à la vente du produit principal. La marge ignore
    aussi les charges de structure, ce qui n'est légitime que si les cultures
    comparées partagent le même système — vrai ici, puisque fertilisation et
    irrigation valent pour toute la requête. Une infrastructure d'irrigation
    reste donc absente de toutes les marges, mais elle ne fausse pas leur
    comparaison : elle pèse pareil sur chacune.
    """

    crop: Culture
    gross_margin_eur_ha: float = Field(
        ..., description="Marge brute simplifiée, **hors aides et "
                         "coproduits** : price_eur_t × expected_yield_t_ha − "
                         "variable_costs_eur_ha.")
    expected_yield_t_ha: float
    variable_costs_eur_ha: float
    price_eur_t: float
    source: str = Field(..., description="`user`, ou `reference:<id>`.")


class SensibilitePrix(BaseModel):
    """Le classement survit-il à une variation de prix ?

    **Convention.** Tous les prix des cultures évaluées varient
    *simultanément*, rendements et charges constants. Le test n'est pas
    tautologique : les charges ne suivent pas le prix, donc multiplier tous
    les prix par λ donne `λ·(prix × rendement) − charges`, et l'ordre des
    marges peut basculer dès que les recettes diffèrent d'une culture à
    l'autre. C'est une hypothèse de simulation, non une fourchette de marché :
    les mouvements indépendants des marchés du blé, du riz ou du coton ne sont
    pas modélisés.

    Un écart de marge plus petit que l'effet d'une variation plausible du prix
    ne départage rien. Les paires concernées sont nommées plutôt que
    silencieusement classées.
    """

    tested_variation: float = Field(
        ..., description="Variation relative appliquée aux prix, par exemple "
                         "0.2 pour ±20 %.")
    convention: str = Field(
        ..., description="Ce qui varie, en clair. Ce champ part dans la "
                         "réponse pour que l'appelant n'ait pas à consulter "
                         "la documentation pour interpréter le résultat.")
    ranking_stable: bool
    indistinguishable_pairs: list[tuple[Culture, Culture]] = Field(
        default_factory=list)


class ClassementEconomique(BaseModel):
    """Les marges, triées — **parmi les cultures évaluées seulement**.

    Le libellé ne dit jamais « la meilleure des six » quand deux budgets ont
    été fournis. Les cultures non évaluées sont nommées, pour que l'absence
    soit visible plutôt que déduite.
    """

    margins: list[MargeCulture] = Field(
        ..., description="Triées par marge brute décroissante.")
    assessed_crops: list[Culture]
    unassessed_crops: list[Culture]
    price_sensitivity: SensibilitePrix
    note: str = Field(
        ..., description="Portée exacte du classement, en clair.")


class Recommandation(BaseModel):
    """L'avis, ou l'absence d'avis assumée.

    `undetermined` est une réponse légitime : moins de deux budgets
    comparables, ou des marges que la sensibilité au prix ne départage pas.
    Trancher quand les données ne tranchent pas serait le vrai défaut.
    """

    status: Literal["options", "undetermined"]
    crops: list[Culture] = Field(
        default_factory=list,
        description="Une entrée, ou plusieurs si rien ne les départage.")
    rationale: str


class RecommandationReponse(BaseModel):
    """Le contrat de recommandation.

    `yield_ranking` est toujours rendu — c'est l'exigence du brief — mais deux
    champs en bornent la lecture : `yield_spread_t_ha`, l'écart entre la
    première et la dernière culture, et `ranking_discriminative`, qui le
    confronte au seuil de pertinence pratique annoncé dans `config.py`. Le
    classement est exact ; il n'est pas forcément discriminant, et le dire
    fait partie de la réponse.

    Ce que ces champs **ne prétendent pas** : trancher si les différences sont
    statistiquement détectables. L'incertitude d'une prédiction individuelle
    n'est pas celle de la différence entre deux prédictions ajustées, et les
    comparer serait une faute. Le seuil est une convention métier.
    """

    yield_ranking: list[RendementCulture] = Field(
        ..., description="Les six cultures, triées sur les prédictions "
                         "**exactes** ; l'arrondi n'intervient qu'à "
                         "l'affichage.")
    yield_ranking_note: str = Field(
        ..., description="Portée pratique du classement, en clair.")
    yield_spread_t_ha: float = Field(
        ..., description="Écart entre la première et la dernière culture du "
                         "classement, en tonnes par hectare.")
    ranking_discriminative: bool = Field(
        ..., description="Faux quand `yield_spread_t_ha` est inférieur au "
                         "seuil de pertinence pratique du projet. C'est une "
                         "convention annoncée, non un test statistique.")
    agronomic_diagnostics: list[DiagnosticAgronomique]
    economic_ranking: ClassementEconomique | None = Field(
        None, description="Absent tant que moins de deux budgets complets "
                          "n'ont été fournis.")
    recommendation: Recommandation
    data_quality: QualiteDonnees
    served_model: ServedModel


# --------------------------------------------------------------------------- #
# Sorties — /logs
# --------------------------------------------------------------------------- #


class EntreeJournal(BaseModel):
    """Un appel observé : ce qui est entré, ce qui est sorti, et quand.

    `response` peut porter un `detail` de validation plutôt qu'une réponse
    métier : les appels rejetés sont journalisés au même titre que les
    autres, et c'est voulu — ils disent quelle saisie échoue.
    """

    timestamp: str = Field(
        ..., description="Date de l'appel, UTC, au format ISO 8601.")
    endpoint: Literal["/predict", "/recommend"]
    status_code: int
    duration_ms: float = Field(
        ..., description="Durée du traitement complet, middleware compris.")
    request: dict | None = Field(
        ..., description="Corps reçu, tel quel. `null` si la requête n'en "
                         "portait pas.")
    response: dict | list | None = Field(
        ..., description="Corps renvoyé, tel quel.")


class JournalReponse(BaseModel):
    """Les derniers appels, le plus récent d'abord.

    `stored` et `capacity` permettent de voir que le tampon a commencé à
    évincer : au-delà de `capacity`, les plus anciens appels n'existent plus.
    """

    returned: int = Field(..., description="Nombre d'entrées dans `entries`.")
    stored: int = Field(..., description="Nombre d'entrées en mémoire.")
    capacity: int = Field(..., description="Taille du tampon circulaire.")
    caveat: str = Field(
        ..., description="Ce que ce journal n'est pas, en clair.")
    entries: list[EntreeJournal]


# --------------------------------------------------------------------------- #
# Sorties — /reference/prices
# --------------------------------------------------------------------------- #


class PrixCultureReference(BaseModel):
    """Le prix à la production d'une culture, et sa dispersion entre pays.

    Les quartiles ne sont pas décoratifs. Ils sont là pour que la médiane ne
    soit pas lue comme « le prix » : celui du maïs s'étend de 186 à 432 €/t
    selon le pays déclarant, soit un facteur 2,3 — bien au-delà de la
    variation de ±20 % que `/recommend` teste en sensibilité.
    """

    crop: Culture
    price_eur_t: float = Field(
        ..., description="Médiane des pays déclarants, en euros par tonne.")
    q1_eur_t: float = Field(
        ..., description="Premier quartile entre pays.")
    q3_eur_t: float = Field(
        ..., description="Troisième quartile entre pays.")
    reporting_countries: int = Field(
        ..., description="Nombre de pays ayant déclaré un prix cette "
                         "année-là. Onze pour le coton, soixante-quinze pour "
                         "le maïs : la médiane n'a pas la même assise selon "
                         "les cultures.")
    faostat_item: str = Field(
        ..., description="Produit FAOSTAT retenu. Pour le coton, le coton "
                         "graine non égrené — ce qui sort de la ferme — et "
                         "non la fibre égrenée, qui est un produit "
                         "industriel vendu quatre fois plus cher.")


class SourcePrixReference(BaseModel):
    """D'où viennent ces prix, et comment ils ont été calculés."""

    source: str
    dataset: str
    element: str
    year: int
    method: str
    currency_conversion: str
    usd_per_eur: float
    extracted_on: str


class PrixReferenceReponse(BaseModel):
    """Des prix indicatifs pour amorcer un budget, et rien de plus.

    **Cette route ne calcule rien.** Elle n'estime aucun rendement, ne compare
    aucune culture et ne recommande rien : elle expose des données de
    référence sourcées. Les calculs appartiennent à `/recommend`, qui reste
    alimenté par les budgets de l'appelant.

    **Elles ne suffisent pas à produire une marge.** Il manque les charges
    opérationnelles, et leur absence est délibérée : des prix mondiaux FAO
    combinés à des charges européennes donnent des marges négatives sur trois
    cultures sur quatre. Fournir des charges de référence non territorialisées
    reviendrait à fabriquer un résultat faux d'apparence crédible.
    """

    prices: list[PrixCultureReference]
    source: SourcePrixReference
    caveat: str = Field(
        ..., description="Ce que ces prix ne disent pas, en clair.")

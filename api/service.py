"""Moteur de service : du contrat validé à la réponse.

Ce module est le seul à connaître le modèle. Il s'interpose entre
`schemas.py` — qui ignore le disque et MLflow — et `main.py`, qui ne traduira
que du HTTP. Toute la logique métier vit donc ici, et se teste sans serveur.

**Deux sources, une seule fonction de chargement.** `AGRITECH_MODEL_URI`
absente : l'artefact embarqué `models/modele_api.joblib`. C'est le cas du
conteneur, qui n'a ni `mlflow.db` ni `mlartifacts/` à sa portée, et dont
l'image n'installe pas MLflow. Variable renseignée
(`models:/agritech-rendement-api@champion`) : le registre MLflow, résolu **une
seule fois au démarrage**. MLflow n'est importé que dans cette branche.

**La provenance vient du fichier d'accompagnement**, écrit par le même export
que l'artefact : les deux sont solidaires. Un registre qui servirait une autre
version que celle décrite par ce fichier fait donc échouer le démarrage —
mieux vaut refuser de servir que d'estampiller une prédiction d'une provenance
fausse.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import joblib
import pandas as pd

from agritech.config import (
    API_METADATA_PATH,
    API_MODEL_PATH,
    API_PANEL_METADATA_PATH,
    API_PANEL_MODEL_PATH,
    CULTURES_PANEL_EN_FRANCAIS,
    DEMI_INTERVALLE_PREDICTION_95_T_HA,
    DOMAINE_ENTRAINEMENT,
    HYPOTHESES_ECONOMIQUES_VERSION,
    MLFLOW_TRACKING_URI,
    MODALITES_CATEGORIELLES,
    PANEL_COUVERTURE_MESUREE,
    PANEL_INTERVALLE_PLANCHER_95_T_HA,
    PANEL_INTERVALLE_RELATIF_95,
    PANEL_PREDICTION_MINIMALE_T_HA,
    PRIX_REFERENCE_FAOSTAT,
    PRIX_REFERENCE_SOURCE,
    SEUIL_ECART_RENDEMENT_PERTINENT_T_HA,
)
from agritech.preprocessing import PANEL_SAISIES
from api.schemas import (
    BudgetProduction,
    ClassementEconomique,
    ClassementNationalRequete,
    ClassementNationalReponse,
    CultureDisponible,
    DiagnosticAgronomique,
    IntervalleEmpirique,
    IntervallePrediction,
    MargeCulture,
    PredictionNationaleReponse,
    PredictionNationaleRequete,
    PredictionReponse,
    PredictionRequete,
    PrixCultureReference,
    PrixReferenceReponse,
    QualiteDonnees,
    Recommandation,
    RecommandationRequete,
    RecommandationReponse,
    RendementCulture,
    RendementNational,
    RendementObserve,
    SensibilitePrix,
    SerieNationale,
    SeriesNationalesReponse,
    ServedModel,
    SourcePrixReference,
)

#: Variable d'environnement qui bascule du joblib embarqué vers le registre.
VARIABLE_URI = "AGRITECH_MODEL_URI"

#: La même bascule, pour la chaîne nationale. Deux variables et non une :
#: les deux artefacts ont des registres distincts et des contrats d'entrée
#: différents, et on doit pouvoir servir l'un depuis le registre sans toucher
#: à l'autre. `scripts/export_api_model.py` affiche ces noms après chaque
#: promotion, et un test confronte les deux listes.
VARIABLE_URI_PANEL = "AGRITECH_PANEL_MODEL_URI"

#: Amplitude du scénario de prix. Hypothèse de simulation, pas une fourchette
#: de marché : rien dans les données ne documente la volatilité des prix.
VARIATION_PRIX = 0.2

CONVENTION_SENSIBILITE = (
    "Tous les prix des cultures évaluées varient simultanément de ±20 %, "
    "rendements et charges constants. Hypothèse de simulation, non une "
    "fourchette de marché : les mouvements indépendants des marchés ne sont "
    "pas modélisés."
)

#: Un centime par hectare ne départage pas deux cultures.
TOLERANCE_MARGE_EUR = 0.01

MOTIF_NON_EVALUE = (
    "Le référentiel de faisabilité agronomique n'est pas audité à ce stade. "
    "`not_assessed` ne doit pas être lu comme « favorable »."
)

CULTURES = MODALITES_CATEGORIELLES["Crop"]


# --------------------------------------------------------------------------- #
# Chargement du modèle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModeleServi:
    """Le pipeline chargé, son identité, et sa provenance d'export."""

    pipeline: object
    served: ServedModel
    provenance: dict


class ArtefactServi(NamedTuple):
    """Où trouver un artefact de service, et par quoi le détourner.

    Les deux chaînes partagent toute la mécanique de chargement — provenance,
    sceau, bascule vers le registre — et ne diffèrent que par ces quatre
    valeurs. Les déclarer ici évite deux fonctions de chargement jumelles,
    dont l'une finirait par ne plus recevoir les corrections de l'autre.
    """

    chemin: Path
    provenance: Path
    variable_uri: str
    #: Ce que le `--source` de l'export appelle cette chaîne, pour que les
    #: messages d'erreur nomment la commande à relancer.
    source: str


def artefact_parcelle() -> ArtefactServi:
    """L'artefact des parcelles simulées, chemins lus **maintenant**.

    Construit à l'appel et non figé dans une constante : les tests détournent
    `API_MODEL_PATH` vers un dossier temporaire pour éprouver le sceau, et une
    liaison faite à l'import continuerait de pointer vers `models/`.
    """
    return ArtefactServi(API_MODEL_PATH, API_METADATA_PATH, VARIABLE_URI,
                         "simulated")


def artefact_panel() -> ArtefactServi:
    """L'artefact du service national, pour la même raison."""
    return ArtefactServi(API_PANEL_MODEL_PATH, API_PANEL_METADATA_PATH,
                         VARIABLE_URI_PANEL, "real")


def _lire_provenance(artefact: ArtefactServi | None = None) -> dict:
    """Le fichier d'accompagnement de l'artefact. Son absence est bloquante."""
    artefact = artefact or artefact_parcelle()
    if not artefact.provenance.exists():
        raise FileNotFoundError(
            f"Provenance absente : {artefact.provenance}. Exécuter "
            f"`uv run python scripts/export_api_model.py "
            f"--source {artefact.source}` avant de servir."
        )
    return json.loads(artefact.provenance.read_text(encoding="utf-8"))


def _resoudre_registre(uri: str) -> tuple[str, int]:
    """Nom et version derrière une URI `models:/`, alias résolu."""
    from mlflow.tracking import MlflowClient

    reste = uri.removeprefix("models:/")
    if "@" in reste:
        nom, alias = reste.split("@", 1)
        version = MlflowClient().get_model_version_by_alias(nom, alias).version
        return nom, int(version)
    nom, version = reste.rsplit("/", 1)
    return nom, int(version)


def _exiger_coherence(provenance: dict, registre: str, version: int) -> None:
    """Registre et provenance doivent décrire le même modèle.

    Les deux sortent du même export. Un écart signifie que l'alias a bougé
    sans réexport : la provenance embarquée décrirait alors une version que
    l'API ne sert pas.
    """
    attendu = (provenance["registre"], int(provenance["version"]))
    if (registre, version) != attendu:
        raise RuntimeError(
            f"Le registre sert {registre} v{version}, la provenance décrit "
            f"{attendu[0]} v{attendu[1]}. Réexécuter "
            "`scripts/export_api_model.py` pour les resynchroniser."
        )


def _exiger_sceau(provenance: dict,
                  artefact: ArtefactServi | None = None) -> None:
    """L'artefact présent doit être celui que la provenance décrit.

    C'est le contrôle qui compte dans un conteneur. Une image fige un
    `.joblib` et un fichier de provenance au moment de sa construction, et
    plus rien ensuite ne dit qu'ils vont ensemble : un artefact périmé, resté
    d'une couche antérieure ou recopié d'une autre image, servirait des
    prédictions en annonçant une provenance qui n'est pas la sienne.

    **Ce que ce contrôle n'atteste pas.** Que l'alias `champion` du registre
    désigne encore cette version. Une image est immuable, un alias ne l'est
    pas : il a pu bouger depuis la construction. `/model/info` annonce donc la
    version **embarquée**, sans prétendre interroger le registre.

    L'échec est délibérément fatal : servir en sachant que l'artefact et sa
    provenance divergent serait pire que de ne pas démarrer.
    """
    artefact = artefact or artefact_parcelle()

    if not artefact.chemin.exists():
        raise FileNotFoundError(
            f"Artefact absent : {artefact.chemin}. L'image doit embarquer "
            f"{artefact.chemin.name} à côté de sa provenance."
        )

    attendue = provenance.get("artefact_sha256")
    if attendue is None:
        raise RuntimeError(
            f"{artefact.provenance.name} ne porte pas d'empreinte : il "
            "précède le contrat de traçabilité courant. Réexécuter "
            "`scripts/export_api_model.py` avant de construire l'image."
        )

    constatee = hashlib.sha256(artefact.chemin.read_bytes()).hexdigest()
    if constatee != attendue:
        raise RuntimeError(
            f"{artefact.chemin.name} n'est pas l'artefact décrit par "
            f"{artefact.provenance.name}.\n"
            f"  attendue  : {attendue}\n"
            f"  constatée : {constatee}\n"
            "  L'image embarque un modèle et une provenance qui ne vont pas "
            "ensemble ; la reconstruire après un export complet."
        )


def charger_modele(uri: str | None = None,
                   artefact: ArtefactServi | None = None) -> ModeleServi:
    """Charge depuis le registre MLflow, ou depuis l'artefact embarqué.

    `artefact` désigne la chaîne à charger ; sans lui, celle des parcelles,
    pour que les appels écrits avant l'arrivée du service national continuent
    de décrire ce qu'ils décrivaient.
    """
    artefact = artefact or artefact_parcelle()
    provenance = _lire_provenance(artefact)
    uri = uri if uri is not None else os.environ.get(artefact.variable_uri)

    if not uri:
        # Le sceau est vérifié **avant** le chargement : désérialiser un
        # pickle dont on sait déjà qu'il ne correspond pas à sa provenance
        # n'apporterait rien et exécuterait son contenu.
        _exiger_sceau(provenance, artefact)
        pipeline = joblib.load(artefact.chemin)
        registre, version = provenance["registre"], int(provenance["version"])
        source = "bundled-joblib"
    else:
        if not uri.startswith("models:/"):
            raise ValueError(
                f"{artefact.variable_uri} attend `models:/nom@alias` ou "
                f"`models:/nom/version` — reçu {uri!r}."
            )
        # Importé ici seulement : MLflow est absent de l'image de l'API.
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        registre, version = _resoudre_registre(uri)
        _exiger_coherence(provenance, registre, version)
        pipeline = mlflow.sklearn.load_model(uri)
        source = "mlflow-registry"

    return ModeleServi(
        pipeline=pipeline,
        served=ServedModel(registry=registre, version=version, source=source),
        provenance=provenance,
    )


def charger_modele_national(uri: str | None = None) -> ModeleServi:
    """Le même chargement, pour l'artefact du panel FAO.

    Une fonction nommée plutôt qu'un argument à la charge de l'appelant :
    `main.py` doit pouvoir charger les deux modèles sans rien savoir des
    chemins, et un test doit pouvoir détourner celui-ci seul.
    """
    return charger_modele(uri, artefact_panel())


@lru_cache(maxsize=1)
def modele_courant() -> ModeleServi:
    """Le modèle du processus : chargé au premier appel, puis mémorisé."""
    return charger_modele()


def info_modele(modele: ModeleServi) -> dict:
    """Ce que `/model/info` expose : la liste des modalités fait foi ici."""
    return {
        "served_model": modele.served.model_dump(),
        "provenance": modele.provenance,
        "accepted_values": {
            champ: list(valeurs)
            for champ, valeurs in MODALITES_CATEGORIELLES.items()
        },
        "training_domain": {
            variable: {"min": bas, "max": haut}
            for variable, (bas, haut) in DOMAINE_ENTRAINEMENT.items()
        },
        "prediction_interval": {
            "half_width_t_ha": DEMI_INTERVALLE_PREDICTION_95_T_HA,
            "nominal_coverage": 0.95,
            "measured_coverage": 0.9492,
            "calibration": "quantiles empiriques des résidus sur `val`",
            "measured_on": "`test`, exclu de la calibration",
        },
    }


# --------------------------------------------------------------------------- #
# Prix de référence
# --------------------------------------------------------------------------- #

#: Ce que la route doit dire d'elle-même, à chaque appel. Rédigé ici plutôt
#: que dans la documentation OpenAPI : un client qui consomme le JSON sans
#: jamais ouvrir Swagger doit lire la réserve avec la donnée.
RESERVE_PRIX = (
    "Prix indicatifs : médiane des pays déclarants à FAOSTAT, et non le prix "
    "payé sur une exploitation donnée. L'écart interquartile est large — de "
    "186 à 432 €/t pour le maïs — et l'assise varie beaucoup d'une culture à "
    "l'autre : 11 pays déclarants pour le coton contre 75 pour le maïs. "
    "Ces prix ne permettent pas de calculer une marge : les charges "
    "opérationnelles manquent, et leur absence est délibérée. Ils servent à "
    "amorcer le champ `price_eur_t` des budgets de /recommend, qui reste "
    "responsable de tout calcul."
)


def prix_reference() -> PrixReferenceReponse:
    """Les prix à la production de référence, avec leur provenance.

    Aucun modèle n'intervient : ce sont des données externes, figées dans
    `config.py` et re-dérivables par `scripts/build_reference_prices.py`.
    La route ne dépend donc pas du chargement de l'artefact — elle répond même
    si le modèle venait à manquer.

    L'ordre suit `PRIX_REFERENCE_FAOSTAT`, c'est-à-dire l'ordre alphabétique
    des cultures. **Pas un tri par prix** : classer des prix de cultures
    différentes suggérerait une comparabilité qui n'existe pas, une tonne de
    coton et une tonne de blé n'ayant ni le même rendement ni les mêmes
    charges derrière elles.
    """
    return PrixReferenceReponse(
        prices=[
            PrixCultureReference(
                crop=culture,
                price_eur_t=valeurs["prix_eur_t"],
                q1_eur_t=valeurs["q1_eur_t"],
                q3_eur_t=valeurs["q3_eur_t"],
                reporting_countries=valeurs["pays"],
                faostat_item=valeurs["item_faostat"],
            )
            for culture, valeurs in PRIX_REFERENCE_FAOSTAT.items()
        ],
        source=SourcePrixReference(
            source=PRIX_REFERENCE_SOURCE["source"],
            dataset=PRIX_REFERENCE_SOURCE["jeu_de_donnees"],
            element=PRIX_REFERENCE_SOURCE["element"],
            year=PRIX_REFERENCE_SOURCE["annee"],
            method=PRIX_REFERENCE_SOURCE["methode"],
            currency_conversion=PRIX_REFERENCE_SOURCE["conversion"],
            usd_per_eur=PRIX_REFERENCE_SOURCE["taux_usd_par_eur"],
            extracted_on=PRIX_REFERENCE_SOURCE["extrait_le"],
        ),
        caveat=RESERVE_PRIX,
    )


# --------------------------------------------------------------------------- #
# Prédiction
# --------------------------------------------------------------------------- #


def _predire(modele: ModeleServi, lignes: list[dict]) -> list[float]:
    """Un seul appel au pipeline, quel que soit le nombre de lignes."""
    return [float(v) for v in modele.pipeline.predict(pd.DataFrame(lignes))]


def _intervalle(valeur: float) -> IntervallePrediction:
    """Demi-largeur constante : la variance des résidus l'est aussi."""
    demi = DEMI_INTERVALLE_PREDICTION_95_T_HA
    return IntervallePrediction(
        lower_t_ha=round(max(valeur - demi, 0.0), 3),
        upper_t_ha=round(valeur + demi, 3),
    )


def _avertissements_domaine(variables: dict) -> list[str]:
    """Hors domaine, Ridge extrapole linéairement et sans plafond."""
    messages = []
    for variable, (bas, haut) in DOMAINE_ENTRAINEMENT.items():
        valeur = variables.get(variable)
        if valeur is None or bas <= valeur <= haut:
            continue
        messages.append(
            f"out_of_domain : {variable} = {valeur:g}, hors du domaine "
            f"observé ({bas:g} à {haut:g}). Le modèle extrapole linéairement "
            "et la couverture de l'intervalle n'y a pas été vérifiée."
        )
    return messages


def _qualite(modele: ModeleServi, requete, avertissements: list[str],
             limitations: list[str], avec_economie: bool) -> QualiteDonnees:
    """Le bloc qui empêche de surinterpréter la réponse."""
    lignes = [
        "Intervalle de prédiction calibré sur les résidus d'un jeu simulé à "
        "bruit gaussien ; sur des parcelles réelles, il devrait être "
        "recalibré.",
        *limitations,
    ]
    omis = requete.facultatives_omises()
    if omis:
        lignes.append(
            f"Champs facultatifs non renseignés : {', '.join(omis)}. Leur "
            "omission déplace la prédiction de 0,0017 t/ha en moyenne, soit "
            "0,34 % du RMSE."
        )
    return QualiteDonnees(
        data_version=modele.provenance["data_version"],
        split_version=modele.provenance["split_version"],
        economic_assumptions_version=(
            HYPOTHESES_ECONOMIQUES_VERSION if avec_economie else None),
        warnings=avertissements,
        limitations=lignes,
    )


def construire_prediction(modele: ModeleServi,
                          requete: PredictionRequete) -> PredictionReponse:
    """`/predict` : un rendement, son intervalle, et de quoi le relativiser."""
    variables = requete.vers_variables()
    (rendement,) = _predire(modele, [variables])
    return PredictionReponse(
        yield_t_ha=round(rendement, 3),
        prediction_interval_95=_intervalle(rendement),
        data_quality=_qualite(modele, requete,
                              _avertissements_domaine(variables), [], False),
        served_model=modele.served,
    )


# --------------------------------------------------------------------------- #
# Classement par rendement
# --------------------------------------------------------------------------- #


def _classement_rendement(
        modele: ModeleServi,
        contexte: dict) -> tuple[list[RendementCulture], float]:
    """Les six cultures dans le même contexte, en un seul appel au modèle.

    **Le tri porte sur la prédiction exacte** ; l'arrondi n'intervient qu'à
    l'affichage. Trier sur la valeur arrondie créait des ex æquo que le tri
    stable départageait par l'ordre de déclaration des cultures — l'ordre
    annoncé n'était alors plus celui du modèle.

    Renvoie aussi l'étendue, calculée elle aussi sur les valeurs exactes.
    """
    lignes = [{**contexte, "Crop": culture} for culture in CULTURES]
    rendements = _predire(modele, lignes)
    ordonnes = sorted(zip(CULTURES, rendements), key=lambda paire: paire[1],
                      reverse=True)
    classement = [
        RendementCulture(crop=culture, yield_t_ha=round(valeur, 3),
                         prediction_interval_95=_intervalle(valeur))
        for culture, valeur in ordonnes
    ]
    return classement, ordonnes[0][1] - ordonnes[-1][1]


def _note_classement(etendue: float) -> str:
    """La portée pratique du classement, dite avec le seuil annoncé.

    Délibérément **sans comparaison à l'intervalle de prédiction** : celui-ci
    mesure l'incertitude d'une parcelle individuelle, pas celle de la
    différence entre deux prédictions ajustées. Les rapprocher laisserait
    croire à une égalité statistique qui n'a pas été établie — et dont la
    significativité n'a pas été évaluée dans ce projet.
    """
    seuil = SEUIL_ECART_RENDEMENT_PERTINENT_T_HA
    constat = (f"Écart entre la première et la dernière culture : "
               f"{etendue:.4f} t/ha, soit {etendue * 1000:.0f} kg/ha. Seuil "
               f"de pertinence pratique retenu pour ce projet : {seuil:g} t/ha "
               "— une convention annoncée, non un seuil de significativité "
               "statistique.")
    if etendue >= seuil:
        return (f"{constat} L'écart l'atteint : le classement a un pouvoir "
                "discriminant, à confronter néanmoins aux contraintes "
                "agronomiques et aux marges.")
    return (f"{constat} L'écart lui est très inférieur : ce classement est "
            "exact mais **sans pouvoir discriminant**. Il ne justifie pas de "
            "choisir une culture plutôt qu'une autre — dans ce jeu de "
            "données, la culture n'explique pas le rendement. C'est le "
            "classement économique qui départage, quand les budgets le "
            "permettent.")


def _diagnostics() -> list[DiagnosticAgronomique]:
    """Toutes les cultures en `not_assessed`, avec leur motif.

    Périmètre assumé de l'étape 3 : déclarer « favorable » sans critère
    réellement évalué serait la seule faute possible ici.
    """
    return [DiagnosticAgronomique(crop=culture, status="not_assessed",
                                  reason=MOTIF_NON_EVALUE)
            for culture in CULTURES]


# --------------------------------------------------------------------------- #
# Classement économique
# --------------------------------------------------------------------------- #


def _marge(budget: BudgetProduction, facteur: float = 1.0) -> float:
    """Marge brute. Seul le prix suit le facteur du scénario."""
    return (budget.price_eur_t * facteur * budget.expected_yield_t_ha
            - budget.variable_costs_eur_ha)


def _signe(ecart: float) -> int:
    """Signe à la tolérance près : sous un centime, aucun départage."""
    if abs(ecart) <= TOLERANCE_MARGE_EUR:
        return 0
    return 1 if ecart > 0 else -1


def _marges(budgets: dict[str, BudgetProduction]) -> list[MargeCulture]:
    """Marges triées décroissant, sur la valeur non arrondie."""
    ordonnes = sorted(budgets.items(), key=lambda kv: _marge(kv[1]),
                      reverse=True)
    return [
        MargeCulture(
            crop=culture,
            gross_margin_eur_ha=round(_marge(budget), 2),
            expected_yield_t_ha=budget.expected_yield_t_ha,
            variable_costs_eur_ha=budget.variable_costs_eur_ha,
            price_eur_t=budget.price_eur_t,
            source="user",
        )
        for culture, budget in ordonnes
    ]


def _sensibilite(budgets: dict[str, BudgetProduction],
                 ordre: list[str]) -> SensibilitePrix:
    """Le classement tient-il quand tous les prix bougent ensemble ?

    Multiplier tous les prix par un facteur ne conserve pas l'ordre : les
    charges ne suivent pas, donc `facteur × (prix × rendement) − charges` se
    réordonne dès que les recettes diffèrent d'une culture à l'autre. Le test
    n'est donc pas tautologique.
    """
    scenarios = [
        {culture: _marge(budget, facteur)
         for culture, budget in budgets.items()}
        for facteur in (1.0, 1 - VARIATION_PRIX, 1 + VARIATION_PRIX)
    ]

    # Une paire n'est départagée que si son ordre strict est le même dans les
    # trois scénarios. Les marges exactement égales sont donc signalées aussi.
    indistinctes = [
        (a, b) for a, b in combinations(ordre, 2)
        if {_signe(marges[a] - marges[b]) for marges in scenarios} != {1}
    ]

    ordres = [sorted(marges, key=marges.get, reverse=True)
              for marges in scenarios[1:]]
    return SensibilitePrix(
        tested_variation=VARIATION_PRIX,
        convention=CONVENTION_SENSIBILITE,
        ranking_stable=all(obtenu == ordre for obtenu in ordres),
        indistinguishable_pairs=indistinctes,
    )


def _classement_economique(
        budgets: dict[str, BudgetProduction]) -> ClassementEconomique:
    """Les marges des seules cultures budgétées, et ce qui manque autour."""
    marges = _marges(budgets)
    ordre = [marge.crop for marge in marges]
    evaluees = [culture for culture in CULTURES if culture in budgets]
    non_evaluees = [culture for culture in CULTURES if culture not in budgets]

    if non_evaluees:
        note = (f"Classement établi parmi les {len(evaluees)} cultures "
                f"disposant d'un budget complet ({', '.join(evaluees)}). "
                f"Non évaluées, faute de budget : {', '.join(non_evaluees)}. "
                "Aucune conclusion de ce bloc ne porte sur les six cultures.")
    else:
        note = ("Les six cultures disposent d'un budget complet : ce "
                "classement est exhaustif.")

    return ClassementEconomique(
        margins=marges,
        assessed_crops=evaluees,
        unassessed_crops=non_evaluees,
        price_sensitivity=_sensibilite(budgets, ordre),
        note=note,
    )


def _recommandation(classement: ClassementEconomique | None) -> Recommandation:
    """L'avis, ou `undetermined` quand les données ne tranchent pas."""
    if classement is None:
        return Recommandation(
            status="undetermined",
            rationale=(
                "Moins de deux budgets complets ont été fournis : aucune "
                "marge n'est comparable. Le classement par rendement ne peut "
                "en tenir lieu : son écart total est sans pouvoir "
                "discriminant, comme l'indique `yield_ranking_note`. "
                "Fournir au moins deux "
                "budgets (système, produit, rendement attendu, charges "
                "variables, prix) pour obtenir un avis."
            ),
        )

    # La tête du classement, élargie à tout ce que la sensibilité au prix n'en
    # sépare pas : annoncer un gagnant unique serait alors une invention.
    tete = classement.margins[0].crop
    cultures = [tete]
    for a, b in classement.price_sensitivity.indistinguishable_pairs:
        autre = b if a == tete else a if b == tete else None
        if autre is not None and autre not in cultures:
            cultures.append(autre)

    portee = f"parmi les {len(classement.assessed_crops)} cultures évaluées"
    if len(cultures) == 1:
        motif = (f"{tete} présente la marge brute simulée la plus élevée "
                 f"{portee} "
                 f"({classement.margins[0].gross_margin_eur_ha:.0f} €/ha), et "
                 "cet ordre résiste à une variation simultanée des prix de "
                 f"±{VARIATION_PRIX:.0%}.")
    else:
        motif = (f"{' et '.join(cultures)} ne sont pas départagées {portee} : "
                 "leur écart de marge n'excède pas l'effet d'une variation "
                 f"des prix de ±{VARIATION_PRIX:.0%}.")

    motif += (" Aucun diagnostic agronomique n'a été appliqué : cet avis est "
              "strictement économique et repose sur les budgets fournis par "
              "l'appelant.")
    if classement.unassessed_crops:
        motif += (" Les cultures sans budget — "
                  f"{', '.join(classement.unassessed_crops)} — n'ont pas été "
                  "comparées ; rien ne permet d'affirmer qu'elles feraient "
                  "moins bien.")

    return Recommandation(status="options", crops=cultures, rationale=motif)


def construire_recommandation(
        modele: ModeleServi,
        requete: RecommandationRequete) -> RecommandationReponse:
    """`/recommend` : les six blocs, dont ceux qui disent ce qu'on ignore."""
    contexte = requete.vers_variables()
    classement, etendue = _classement_rendement(modele, contexte)

    # Deux budgets au minimum : en dessous, aucune marge n'est comparable et
    # le bloc économique reste absent plutôt que trompeur.
    economique = (_classement_economique(requete.budgets)
                  if len(requete.budgets) >= 2 else None)

    limitations = []
    if economique is not None:
        limitations.append(
            "Les budgets sont ceux fournis par l'appelant, et aucun des "
            "jeux de données du projet ne contient de prix ni de charges. La "
            "version d'hypothèses économiques identifie la convention de "
            "calcul — marge brute simplifiée, hors aides et coproduits : "
            "prix × rendement − charges opérationnelles — non des prix de "
            "marché."
        )
    elif requete.budgets:
        limitations.append(
            "Un seul budget a été fourni : il ne permet aucune comparaison, "
            "le classement économique est donc absent."
        )

    return RecommandationReponse(
        yield_ranking=classement,
        yield_ranking_note=_note_classement(etendue),
        yield_spread_t_ha=round(etendue, 6),
        ranking_discriminative=etendue >= SEUIL_ECART_RENDEMENT_PERTINENT_T_HA,
        agronomic_diagnostics=_diagnostics(),
        economic_ranking=economique,
        recommendation=_recommandation(economique),
        data_quality=_qualite(modele, requete,
                              _avertissements_domaine(contexte), limitations,
                              economique is not None),
        served_model=modele.served,
    )


# --------------------------------------------------------------------------- #
# Service national — la seconde chaîne
#
# Ce que ces trois fonctions servent n'est pas une variante de ce qui précède.
# L'unité d'observation change : un pays, une culture, une année, et non une
# parcelle dans des conditions données. D'où des blocs de réponse distincts,
# un intervalle calibré autrement, et aucune ligne de calcul en commun hormis
# la provenance du modèle.
# --------------------------------------------------------------------------- #

#: Les quatre réserves que **toute** réponse de cette chaîne porte, en une
#: ligne chacune. Elles sont courtes par construction : une réponse de
#: `/national/ranking` les aurait sinon fait voyager à côté de dix cultures.
#: Le détail — méthode de calibration, hétérogénéité de l'erreur, variables
#: écartées — vit dans les schémas et le rapport, où on le lit une fois.
LIMITATIONS_NATIONALES = (
    "Rendement **national** annuel, non parcellaire : la moyenne d'un pays sur "
    "une année. Ne remplace pas `/predict`.",
    "Aucune condition de culture n'entre dans le calcul : la prédiction "
    "reconduit les trois derniers rendements connus de la série.",
    f"Intervalle empirique : couverture globale de "
    f"{PANEL_COUVERTURE_MESUREE['globale']:.1%} mesurée sur `test`, non une "
    f"garantie par prédiction.",
)

NOTE_ECONOMIQUE = (
    "Aucune marge : il manque six prix sur dix pour ces cultures, et sur les "
    "quatre tarifées l'ordre change entre les tonnes et la valeur."
)

RESERVE_SERIES = (
    "Seules les séries dont le dernier relevé est l'année de référence sont "
    "servies : un historique interrompu plus tôt ne permet pas de prévoir "
    "l'année suivante."
)


def _derivation(modele: ModeleServi):
    """Le transformateur du pipeline servi, sous l'enveloppe du gel.

    `FrozenEstimator` expose le Pipeline sous `estimator` ; un Pipeline nu
    n'a pas cet attribut, et un double de test non gelé non plus. Le `getattr`
    accepte les deux au lieu d'imposer une structure que seul l'export
    produit.
    """
    pipeline = modele.pipeline
    return getattr(pipeline, "estimator", pipeline).named_steps["derivation"]


def _intervalle_empirique(valeur: float) -> IntervalleEmpirique:
    """Le relatif, élargi par le plancher absolu. L'union des deux.

    Le maximum des deux largeurs, et non leur somme : chacune couvre 95 % à
    elle seule sur son domaine, les additionner produirait un intervalle deux
    fois trop large là où les deux mordent.
    """
    base = max(valeur, PANEL_PREDICTION_MINIMALE_T_HA)
    bas_relatif, haut_relatif = PANEL_INTERVALLE_RELATIF_95
    bas_plancher, haut_plancher = PANEL_INTERVALLE_PLANCHER_95_T_HA

    bas = valeur + min(bas_relatif * base, bas_plancher)
    haut = valeur + max(haut_relatif * base, haut_plancher)
    return IntervalleEmpirique(
        lower_t_ha=round(max(bas, 0.0), 3),
        upper_t_ha=round(haut, 3),
        measured_coverage=PANEL_COUVERTURE_MESUREE["globale"],
    )


def _qualite_nationale(modele: ModeleServi,
                       limitations: tuple[str, ...] = ()) -> QualiteDonnees:
    """Le bloc de réserves de cette chaîne.

    `training_data` vaut `real` : ces rendements sont observés, non simulés.
    C'est la seule bonne nouvelle du bloc, et elle ne dispense d'aucune des
    autres lignes — des données réelles à la mauvaise échelle ne répondent pas
    mieux à la question d'un agriculteur.
    """
    annee = modele.provenance.get("annee_predite")
    datation = (
        f"Prévision pour {annee}, l'année qui suit le dernier relevé du "
        "panel — pas la campagne en cours."
    ) if annee else None

    return QualiteDonnees(
        training_data="real",
        data_version=modele.provenance["data_version"],
        split_version=modele.provenance["split_version"],
        economic_assumptions_version=None,
        warnings=[],
        limitations=[ligne for ligne in (datation, *LIMITATIONS_NATIONALES,
                                         *limitations) if ligne],
    )


def _historique(derivation, area: str, item: str) -> list[RendementObserve]:
    """Les trois rendements servis, du plus récent au plus ancien.

    Les années se déduisent de l'année de référence : `lag1` est celle-là,
    puis l'on remonte d'un an par retard. C'est le décalage que le
    transformateur applique, lu dans l'autre sens.
    """
    retards = derivation.retards(area, item)
    return [RendementObserve(year=derivation.annee_reference_ - rang,
                             yield_t_ha=round(valeur, 4))
            for rang, valeur in enumerate(retards.values())]


def construire_prediction_nationale(
        modele: ModeleServi,
        requete: PredictionNationaleRequete) -> PredictionNationaleReponse:
    """`/national/predict` : un rendement national, et d'où il vient."""
    derivation = _derivation(modele)
    # La prédiction d'abord : c'est le transformateur qui refuse un couple
    # absent de sa table, avec le message qui dit pourquoi.
    (rendement,) = _predire(modele, [{"Area": requete.area,
                                      "Item": requete.item}])

    return PredictionNationaleReponse(
        area=requete.area,
        item=requete.item,
        reference_year=derivation.annee_reference_,
        target_year=derivation.annee_predite_,
        yield_t_ha=round(rendement, 3),
        prediction_interval_95=_intervalle_empirique(rendement),
        history=_historique(derivation, requete.area, requete.item),
        data_quality=_qualite_nationale(modele),
        served_model=modele.served,
    )


def _note_classement_national(cultures: list[str], etendue: float,
                              annee: int) -> str:
    """La portée du classement, dite avec ce qui le produit.

    Un pays peut n'avoir qu'une culture : annoncer un classement dans ce cas
    serait un abus de mot, et la note le dit plutôt que de rendre une liste
    d'un élément sans commentaire.
    """
    if len(cultures) == 1:
        return (f"{cultures[0]} est la seule culture à historique exploitable "
                f"ici : pas de classement, une prévision pour {annee}.")

    return (
        f"Classement par rendement national estimé pour {annee}, non par "
        f"rentabilité. Écart première-dernière : {etendue:.2f} t/ha, dû "
        "surtout au niveau de la culture. Ne dépend d'aucune condition saisie."
    )


def construire_classement_national(
        modele: ModeleServi,
        requete: ClassementNationalRequete) -> ClassementNationalReponse:
    """`/national/ranking` : les cultures du pays, par rendement estimé.

    Seules les cultures que ce pays a effectivement cultivées sont comparées.
    Sans ce garde-fou, le modèle interrogé sur la France placerait le plantain
    en deuxième position : il lui attribuerait l'historique mondial faute du
    sien.
    """
    derivation = _derivation(modele)
    cultures = derivation.cultures_par_pays().get(requete.area)
    if not cultures:
        raise ValueError(
            f"Pays non servable : {requete.area!r}. La table embarquée en "
            f"porte {len(derivation.cultures_par_pays())}, sous leur nom FAO ; "
            "`GET /national/series` en donne la liste."
        )

    rendements = _predire(modele, [{"Area": requete.area, "Item": item}
                                   for item in cultures])
    # Tri sur la valeur exacte, l'arrondi n'intervenant qu'à l'affichage :
    # trier sur l'arrondi créerait des ex æquo que l'ordre de déclaration
    # départagerait, et l'ordre annoncé ne serait plus celui du modèle.
    ordonnes = sorted(zip(cultures, rendements), key=lambda paire: paire[1],
                      reverse=True)

    classement = [
        RendementNational(
            item=item,
            item_fr=CULTURES_PANEL_EN_FRANCAIS.get(item, item),
            yield_t_ha=round(valeur, 3),
            prediction_interval_95=_intervalle_empirique(valeur),
            last_observed_t_ha=round(
                derivation.retards(requete.area, item)["lag1"], 4),
        )
        for item, valeur in ordonnes
    ]
    etendue = ordonnes[0][1] - ordonnes[-1][1]

    return ClassementNationalReponse(
        area=requete.area,
        reference_year=derivation.annee_reference_,
        target_year=derivation.annee_predite_,
        ranking=classement,
        ranking_note=_note_classement_national(
            [item for item, _ in ordonnes], etendue,
            derivation.annee_predite_),
        economic_note=NOTE_ECONOMIQUE,
        data_quality=_qualite_nationale(modele),
        served_model=modele.served,
    )


def series_nationales(modele: ModeleServi) -> SeriesNationalesReponse:
    """`/national/series` : ce que le service sait prédire, et pour quand.

    La table embarquée fait foi. Une interface qui recopierait cette liste
    proposerait tôt ou tard un couple que l'artefact ne connaît pas — c'est
    la même raison qui fait de `/model/info` la référence des modalités de la
    chaîne des parcelles.
    """
    derivation = _derivation(modele)
    par_pays = derivation.cultures_par_pays()

    series = [
        SerieNationale(
            area=aire,
            items=[CultureDisponible(
                item=item,
                item_fr=CULTURES_PANEL_EN_FRANCAIS.get(item, item))
                for item in cultures],
        )
        for aire, cultures in par_pays.items()
    ]

    return SeriesNationalesReponse(
        reference_year=derivation.annee_reference_,
        target_year=derivation.annee_predite_,
        areas=series,
        areas_count=len(series),
        series_count=sum(len(serie.items) for serie in series),
        served_model=modele.served,
        caveat=RESERVE_SERIES,
    )

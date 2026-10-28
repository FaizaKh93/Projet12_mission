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

import joblib
import pandas as pd

from agritech.config import (
    API_METADATA_PATH,
    API_MODEL_PATH,
    DEMI_INTERVALLE_PREDICTION_95_T_HA,
    DOMAINE_ENTRAINEMENT,
    HYPOTHESES_ECONOMIQUES_VERSION,
    MLFLOW_TRACKING_URI,
    MODALITES_CATEGORIELLES,
    PRIX_REFERENCE_FAOSTAT,
    PRIX_REFERENCE_SOURCE,
    SEUIL_ECART_RENDEMENT_PERTINENT_T_HA,
)
from api.schemas import (
    BudgetProduction,
    ClassementEconomique,
    DiagnosticAgronomique,
    IntervallePrediction,
    MargeCulture,
    PredictionReponse,
    PredictionRequete,
    PrixCultureReference,
    PrixReferenceReponse,
    QualiteDonnees,
    Recommandation,
    RecommandationRequete,
    RecommandationReponse,
    RendementCulture,
    SensibilitePrix,
    ServedModel,
    SourcePrixReference,
)

#: Variable d'environnement qui bascule du joblib embarqué vers le registre.
VARIABLE_URI = "AGRITECH_MODEL_URI"

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


def _lire_provenance() -> dict:
    """Le fichier d'accompagnement de l'artefact. Son absence est bloquante."""
    if not API_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Provenance absente : {API_METADATA_PATH}. Exécuter "
            "`uv run python scripts/export_api_model.py` avant de servir."
        )
    return json.loads(API_METADATA_PATH.read_text(encoding="utf-8"))


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


def _exiger_sceau(provenance: dict) -> None:
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
    if not API_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Artefact absent : {API_MODEL_PATH}. L'image doit embarquer "
            f"{API_MODEL_PATH.name} à côté de sa provenance."
        )

    attendue = provenance.get("artefact_sha256")
    if attendue is None:
        raise RuntimeError(
            f"{API_METADATA_PATH.name} ne porte pas d'empreinte : il précède "
            "le contrat de traçabilité courant. Réexécuter "
            "`scripts/export_api_model.py` avant de construire l'image."
        )

    constatee = hashlib.sha256(API_MODEL_PATH.read_bytes()).hexdigest()
    if constatee != attendue:
        raise RuntimeError(
            f"{API_MODEL_PATH.name} n'est pas l'artefact décrit par "
            f"{API_METADATA_PATH.name}.\n"
            f"  attendue  : {attendue}\n"
            f"  constatée : {constatee}\n"
            "  L'image embarque un modèle et une provenance qui ne vont pas "
            "ensemble ; la reconstruire après un export complet."
        )


def charger_modele(uri: str | None = None) -> ModeleServi:
    """Charge depuis le registre MLflow, ou depuis l'artefact embarqué."""
    provenance = _lire_provenance()
    uri = uri if uri is not None else os.environ.get(VARIABLE_URI)

    if not uri:
        # Le sceau est vérifié **avant** le chargement : désérialiser un
        # pickle dont on sait déjà qu'il ne correspond pas à sa provenance
        # n'apporterait rien et exécuterait son contenu.
        _exiger_sceau(provenance)
        pipeline = joblib.load(API_MODEL_PATH)
        registre, version = provenance["registre"], int(provenance["version"])
        source = "bundled-joblib"
    else:
        if not uri.startswith("models:/"):
            raise ValueError(
                f"{VARIABLE_URI} attend `models:/nom@alias` ou "
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

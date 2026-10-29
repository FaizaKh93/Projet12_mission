"""Ce que l'API promet, et ce qu'elle refuse de taire.

`test_preprocessing.py` répond à « l'artefact déployé prédit-il ce que
prédirait le modèle entraîné ? ». Ce module répond à deux autres questions,
qu'une parité de prédiction ne couvre pas :

**Les constantes décrivent-elles encore le modèle sérialisé ?** `config.py`
porte les modalités acceptées et le domaine d'entraînement. Ce sont des copies
— utiles, parce que le contrat Pydantic doit connaître l'énumération avant
tout chargement de modèle, mais des copies. Si un réentraînement changeait les
catégories de l'encodeur, l'API rejetterait des valeurs valides ou en
accepterait d'invalides, **sans rien signaler**. Les quatre premiers tests
existent pour que cette dérive échoue bruyamment.

**Les réponses disent-elles ce qu'elles ignorent ?** Un classement établi sur
deux budgets n'est pas un classement des six cultures ; une marge calculée sur
des prix fournis par l'appelant n'est pas un prix de marché ; `not_assessed`
n'est pas `favorable`. Ces distinctions sont la valeur du produit, pas une
politesse : elles sont donc testées comme le reste.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import OneHotEncoder

from agritech import training
from agritech.config import (
    ALIAS_CHAMPION,
    API_METADATA_PATH,
    API_MODEL_PATH,
    DEMI_INTERVALLE_PREDICTION_95_T_HA,
    DOMAINE_ENTRAINEMENT,
    MLFLOW_DB_PATH,
    MODALITES_CATEGORIELLES,
    PRIX_REFERENCE_SOURCE,
    SEUIL_ECART_RENDEMENT_PERTINENT_T_HA,
    TRAIN_PATH,
)
from agritech.preprocessing import (
    FEATURES_FACULTATIVES,
    FEATURES_OBLIGATOIRES,
    FEATURES_SAISIES,
    SENTINELLE,
)
import scripts.export_api_model as export
from api import journal, service
from api.main import app
from api.schemas import (CHAMP_VERS_VARIABLE, PredictionRequete,
                         RecommandationRequete)

CULTURES = MODALITES_CATEGORIELLES["Crop"]

#: Un contexte au milieu du domaine : rien n'y déclenche d'avertissement.
CONTEXTE = {"rainfall_mm": 500.0, "temperature_celsius": 25.0,
            "fertilizer_used": True, "irrigation_used": False}


def budget(rendement: float, charges: float, prix: float) -> dict:
    """Un budget complet : trois nombres, rien de plus.

    Le système de production ne s'y trouve pas — il est fixé pour toute la
    requête par `fertilizer_used` et `irrigation_used`.
    """
    return {"expected_yield_t_ha": rendement,
            "variable_costs_eur_ha": charges, "price_eur_t": prix}


BUDGET_RIZ = budget(6.0, 800.0, 300.0)
BUDGET_BLE = budget(4.0, 500.0, 180.0)

#: Deux budgets dont l'ordre bascule dans la plage de prix testée. Riz :
#: 200 × 5 − 400 = 600 €/ha. Blé : 200 × 10 − 1 500 = 500 €/ha. Les charges ne
#: suivant pas le prix, le point de bascule est à 1,1.
BUDGETS_QUI_BASCULENT = {"Rice": budget(5.0, 400.0, 200.0),
                         "Wheat": budget(10.0, 1500.0, 200.0)}

#: Trois budgets de marge identique — 500 €/ha chacun — mais de recettes très
#: différentes : 3 000, 2 000 et 1 000 €/ha. Les charges ne suivant pas le
#: prix, aucune des trois paires ne conserve le même ordre strict dans les
#: trois scénarios. Les cultures sont donc indistinctes deux à deux, et l'avis
#: doit les nommer toutes plutôt que d'en élire une.
BUDGETS_INDISTINCTS = {"Rice": budget(10.0, 2500.0, 300.0),
                       "Wheat": budget(10.0, 1500.0, 200.0),
                       "Barley": budget(10.0, 500.0, 100.0)}


@pytest.fixture(scope="module")
def client():
    """Client HTTP sur l'application réelle.

    Le `with` n'est pas décoratif : sans lui le `lifespan` ne s'exécute pas,
    le modèle n'est jamais chargé et chaque route échoue. Cette fixture
    éprouve donc aussi la prédiction de sonde du démarrage.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
def journal_vide():
    """Un journal vide, parce qu'il est partagé par tout le processus.

    Le client est de portée module : tous les tests qui l'ont précédé ont
    déjà rempli le tampon. Sans ce nettoyage, les assertions de comptage
    dépendraient de l'ordre d'exécution.
    """
    journal.JOURNAL.clear()
    yield journal.JOURNAL
    journal.JOURNAL.clear()


@pytest.fixture(scope="module")
def artefact():
    """L'artefact réellement déployé, chargé depuis le disque.

    C'est lui qui fait foi dans les tests de cohérence : les constantes de
    `config.py` doivent décrire ce fichier-là, pas une intention.
    """
    return joblib.load(API_MODEL_PATH)


@pytest.fixture(scope="module")
def provenance():
    """Le fichier d'accompagnement écrit par le même export que l'artefact.

    Troisième source indépendante, à côté du contrat Pydantic et des
    constantes du transformateur : un écart entre les trois est un défaut.
    """
    return json.loads(API_METADATA_PATH.read_text(encoding="utf-8"))


def _encodeur(artefact) -> OneHotEncoder:
    """L'encodeur réellement ajusté, au fond de l'artefact déployé."""
    pretraitement = (artefact.estimator
                     .named_steps["modele"]
                     .named_steps["pretraitement"])
    for _, transformateur, _ in pretraitement.transformers_:
        if isinstance(transformateur, OneHotEncoder):
            return transformateur
        for etape in getattr(transformateur, "named_steps", {}).values():
            if isinstance(etape, OneHotEncoder):
                return etape
    raise AssertionError("aucun OneHotEncoder dans le prétraitement")


def _cles(document) -> list[str]:
    """Toutes les clés d'un document JSON imbriqué."""
    if isinstance(document, dict):
        return [cle for paire in document.items()
                for cle in (paire[0], *_cles(paire[1]))]
    if isinstance(document, list):
        return [cle for element in document for cle in _cles(element)]
    return []


# --------------------------------------------------------------------------- #
# Les constantes décrivent-elles encore le modèle sérialisé ?
# --------------------------------------------------------------------------- #


def test_modalites_coincident_avec_encodeur_ajuste(artefact):
    """`MODALITES_CATEGORIELLES` est exactement ce que l'encodeur a appris.

    C'est le test annoncé en commentaire dans `config.py`. Sans lui, les
    modalités recopiées pourraient diverger du modèle au premier
    réentraînement, et la divergence ne se verrait qu'en production.
    """
    encodeur = _encodeur(artefact)
    obtenu = {nom: tuple(categories) for nom, categories
              in zip(encodeur.feature_names_in_, encodeur.categories_)}
    assert obtenu == MODALITES_CATEGORIELLES


def test_sentinelle_absente_des_modalites_apprises(artefact):
    """La sentinelle n'est pas une catégorie, et c'est ce qui la fait marcher.

    Un champ facultatif omis devient `__INCONNU__`, inconnu de l'encodeur, qui
    l'encode en zéros. Si la sentinelle entrait dans les catégories apprises,
    le chemin « information absente » changerait de comportement en silence.
    """
    encodeur = _encodeur(artefact)
    assert encodeur.handle_unknown == "ignore"
    for categories in encodeur.categories_:
        assert SENTINELLE not in categories


def test_correspondance_champs_couvre_exactement_les_saisies():
    """Les deux vocabulaires se correspondent sans trou ni surplus.

    `rainfall_mm` côté JSON, `Rainfall_mm` côté modèle : la correspondance
    existe, elle est dérivée plutôt que recopiée, et elle est vérifiée ici.
    """
    assert set(CHAMP_VERS_VARIABLE.values()) == set(FEATURES_SAISIES)
    assert set(CHAMP_VERS_VARIABLE) == {v.lower() for v in FEATURES_SAISIES}
    assert set(PredictionRequete.model_fields) == set(CHAMP_VERS_VARIABLE)


def test_obligatoires_et_facultatives_conformes_au_modele(provenance):
    """Contrat, transformateur et artefact publié déclarent la même répartition.

    Trois sources indépendantes : les champs requis de Pydantic, les constantes
    du transformateur, et le fichier de provenance écrit à l'export. Un écart
    entre elles signifierait qu'un champ est devenu facultatif pour l'API mais
    reste obligatoire pour le modèle — ou l'inverse.
    """
    requis = {champ for champ, info in PredictionRequete.model_fields.items()
              if info.is_required()}
    facultatifs = set(PredictionRequete.model_fields) - requis

    variables_requises = {CHAMP_VERS_VARIABLE[c] for c in requis}
    variables_facultatives = {CHAMP_VERS_VARIABLE[c] for c in facultatifs}

    assert variables_requises == set(FEATURES_OBLIGATOIRES)
    assert variables_facultatives == set(FEATURES_FACULTATIVES)
    assert variables_requises == set(provenance["saisies_obligatoires"])
    assert variables_facultatives == set(provenance["saisies_facultatives"])


def test_concordance_couvre_toute_la_provenance_descriptive(provenance):
    """Toute clé décrivant le modèle entre dans le contrôle de promotion.

    `--promouvoir` refuse de poser l'alias si la provenance du disque ne
    décrit pas la version enregistrée. Ce contrôle ne vaut que par sa
    couverture : une clé descriptive ajoutée à l'export sans être inscrite
    dans `CLES_CONCORDANCE` cesserait d'être vérifiée, sans que rien ne le
    signale.

    Les clés exclues ici ne décrivent pas le modèle — son identité dans le
    registre, ses alias, ses dates, la liste de ses saisies.

    Deux assertions plutôt qu'une égalité, parce que les deux écarts ont des
    causes opposées : une clé non contrôlée est un défaut du script, une clé
    absente du fichier signale une provenance à réexporter.
    """
    hors_perimetre = {"registre", "version", "alias", "alias_precedent",
                      "exporte_le", "promu_le", "saisies_obligatoires",
                      "saisies_facultatives"}
    descriptives = set(provenance) - hors_perimetre

    non_controlees = descriptives - set(export.CLES_CONCORDANCE)
    assert not non_controlees, (
        f"clés descriptives hors du contrôle de promotion : {non_controlees}")

    absentes = set(export.CLES_CONCORDANCE) - descriptives
    assert not absentes, (
        f"provenance antérieure à ces clés : {absentes} — réexécuter "
        f"scripts/export_api_model.py")


def test_artefact_attend_les_huit_saisies(artefact):
    """L'artefact déployé déclare les mêmes entrées que le contrat."""
    derivation = artefact.estimator.named_steps["derivation"]
    assert list(derivation.feature_names_in_) == list(FEATURES_SAISIES)


# Ce contrôle confronte la configuration au dataset : il n'a de sens que là
# où le dataset existe. Absent du runner, il est ignoré — pas supprimé.
@pytest.mark.skipif(not TRAIN_PATH.exists(),
                    reason="partitions absentes : bornes vérifiées dans "
                           "l'environnement MLOps")
def test_domaine_declare_encadre_les_donnees_vues():
    """Les bornes de `config.py` sont celles du jeu d'entraînement.

    Elles déclenchent l'avertissement d'extrapolation. Trop larges, elles le
    tairaient là où il est utile ; trop étroites, elles l'émettraient sur des
    saisies parfaitement ordinaires.
    """
    train, _ = training.charger_partitions()
    for variable, (bas, haut) in DOMAINE_ENTRAINEMENT.items():
        observe_bas = float(train[variable].min())
        observe_haut = float(train[variable].max())
        assert bas <= observe_bas, f"{variable} : borne basse trop haute"
        assert observe_haut <= haut, f"{variable} : borne haute trop basse"
        # Serrées : au plus 1 % plus larges que l'étendue réellement observée.
        assert (haut - bas) <= 1.01 * (observe_haut - observe_bas)


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #


def test_predict_rend_le_rendement_et_son_intervalle(client):
    """Un rendement, encadré par l'intervalle calibré."""
    reponse = client.post("/predict", json={**CONTEXTE, "crop": "Wheat"})
    assert reponse.status_code == 200
    corps = reponse.json()
    intervalle = corps["prediction_interval_95"]

    assert corps["yield_t_ha"] > 0
    assert intervalle["nominal_coverage"] == 0.95
    assert (intervalle["lower_t_ha"] < corps["yield_t_ha"]
            < intervalle["upper_t_ha"])
    largeur = intervalle["upper_t_ha"] - intervalle["lower_t_ha"]
    assert largeur == pytest.approx(2 * DEMI_INTERVALLE_PREDICTION_95_T_HA,
                                    abs=2e-3)


def test_intervalle_tronque_a_zero():
    """La borne basse est tronquée : un rendement négatif n'existe pas."""
    intervalle = service._intervalle(0.2)
    assert intervalle.lower_t_ha == 0.0
    assert intervalle.upper_t_ha == pytest.approx(
        0.2 + DEMI_INTERVALLE_PREDICTION_95_T_HA)


def test_extremes_recevables_acceptes(client):
    """Les bornes de recevabilité sont larges à dessein, et tiennent."""
    reponse = client.post("/predict", json={
        "rainfall_mm": 0.0, "temperature_celsius": -20.0,
        "fertilizer_used": False, "irrigation_used": False, "crop": "Wheat"})
    assert reponse.status_code == 200
    assert reponse.json()["prediction_interval_95"]["lower_t_ha"] >= 0.0


def test_predict_ne_renvoie_aucun_champ_monetaire(client):
    """Aucune traduction monétaire dans `/predict`.

    Un prix unique appliqué à six cultures n'a pas de sens agronomique : la
    dimension économique appartient à `/recommend`, où un budget par culture
    peut les distinguer.
    """
    corps = client.post("/predict", json={**CONTEXTE, "crop": "Wheat"}).json()
    monetaires = [cle for cle in _cles(corps)
                  if any(mot in cle for mot in
                         ("eur", "price", "revenue", "margin", "cost"))]
    assert monetaires == []


@pytest.mark.parametrize("charge, cas", [
    ({**CONTEXTE, "crop": "Quinoa"}, "culture hors énumération"),
    ({**CONTEXTE, "crop": "Wheat", "soil_typ": "Loam"}, "champ mal orthographié"),
    ({**CONTEXTE, "crop": "Wheat", "soil_type": "Granite"}, "modalité inconnue"),
    ({**CONTEXTE, "crop": "Wheat", "rainfall_mm": -10.0}, "pluie négative"),
    ({**CONTEXTE, "crop": "Wheat", "temperature_celsius": 300.0}, "température absurde"),
    ({"crop": "Wheat"}, "contexte absent"),
])
def test_entree_invalide_rejetee(client, charge, cas):
    """Le contrat refuse plutôt que de deviner.

    Le champ mal orthographié compte double : accepté, il deviendrait une
    « information absente » sans que personne en soit averti.
    """
    assert client.post("/predict", json=charge).status_code == 422, cas


def test_saisies_obligatoires_seules_acceptees(client):
    """Les trois catégorielles facultatives peuvent être omises, et c'est dit."""
    corps = client.post("/predict", json={**CONTEXTE, "crop": "Wheat"}).json()
    assert any("facultatifs non renseignés" in ligne
               for ligne in corps["data_quality"]["limitations"])


def test_hors_domaine_averti_mais_pas_refuse(client):
    """Avertir, pas refuser : une saison plus humide reste une question légitime."""
    reponse = client.post("/predict", json={
        **CONTEXTE, "crop": "Rice", "rainfall_mm": 1400.0,
        "temperature_celsius": 45.0})
    assert reponse.status_code == 200
    avertissements = reponse.json()["data_quality"]["warnings"]
    assert len(avertissements) == 2
    assert all(ligne.startswith("out_of_domain") for ligne in avertissements)


def test_dans_le_domaine_aucun_avertissement(client):
    """Pas de bruit sur une saisie ordinaire, sinon l'avertissement ne vaut rien."""
    corps = client.post("/predict", json={**CONTEXTE, "crop": "Wheat"}).json()
    assert corps["data_quality"]["warnings"] == []


def test_reponse_estampillee_par_la_provenance(client, provenance):
    """Chaque prédiction porte le modèle servi et les versions de données."""
    corps = client.post("/predict", json={**CONTEXTE, "crop": "Wheat"}).json()
    assert corps["served_model"] == {"registry": provenance["registre"],
                                     "version": provenance["version"],
                                     "source": "bundled-joblib"}
    assert corps["data_quality"]["data_version"] == provenance["data_version"]
    assert corps["data_quality"]["split_version"] == provenance["split_version"]
    assert corps["data_quality"]["training_data"] == "simulated"


# --------------------------------------------------------------------------- #
# /recommend — le classement par rendement
# --------------------------------------------------------------------------- #


def test_classement_rendement_toujours_rendu(client):
    """Les six cultures, triées, quelle que soit l'information économique."""
    classement = client.post("/recommend", json=CONTEXTE).json()["yield_ranking"]
    assert {ligne["crop"] for ligne in classement} == set(CULTURES)
    rendements = [ligne["yield_t_ha"] for ligne in classement]
    assert rendements == sorted(rendements, reverse=True)


def test_tri_sur_les_predictions_exactes(client):
    """Le classement suit les prédictions exactes, pas leur arrondi.

    Trier sur la valeur arrondie au millième créait des ex æquo que le tri
    stable départageait par l'ordre de déclaration des cultures. Deux cultures
    séparées par 0,0002 t/ha sortaient alors dans le mauvais ordre, et le
    classement annoncé n'était plus celui du modèle.
    """
    contexte = RecommandationRequete(**CONTEXTE).vers_variables()
    lignes = [{**contexte, "Crop": culture} for culture in CULTURES]
    exactes = dict(zip(CULTURES,
                       service._predire(service.modele_courant(), lignes)))
    attendu = sorted(CULTURES, key=lambda culture: exactes[culture],
                     reverse=True)

    obtenu = [ligne["crop"] for ligne in
              client.post("/recommend", json=CONTEXTE).json()["yield_ranking"]]
    assert obtenu == attendu

    # Le piège est bien présent : l'arrondi fait apparaître des ex æquo là où
    # les prédictions exactes n'en ont pas. Sans cela, le test ne prouverait
    # rien — il passerait aussi avec l'ancien tri.
    affichees = [round(exactes[culture], 3) for culture in attendu]
    assert len(set(affichees)) < len(affichees)


def test_pouvoir_discriminant_declare(client):
    """L'écart est publié et confronté au seuil annoncé, pas à l'intervalle.

    L'incertitude d'une prédiction individuelle n'est pas celle de la
    différence entre deux prédictions ajustées : les deux ne sont pas
    commensurables. La note ne doit donc pas rapprocher l'écart de
    l'intervalle de prédiction, sous peine de suggérer une égalité
    statistique que rien n'établit — la significativité des écarts entre
    cultures n'a pas été évaluée dans ce projet. Le seuil est une
    convention métier, et la réponse le dit.
    """
    corps = client.post("/recommend", json=CONTEXTE).json()
    classement = corps["yield_ranking"]
    etendue = corps["yield_spread_t_ha"]

    assert etendue == pytest.approx(
        classement[0]["yield_t_ha"] - classement[-1]["yield_t_ha"], abs=1e-3)
    assert corps["ranking_discriminative"] is (
        etendue >= SEUIL_ECART_RENDEMENT_PERTINENT_T_HA)
    assert corps["ranking_discriminative"] is False

    note = corps["yield_ranking_note"]
    assert "sans pouvoir discriminant" in note
    assert f"{SEUIL_ECART_RENDEMENT_PERTINENT_T_HA:g} t/ha" in note
    assert "convention annoncée" in note
    assert "intervalle de prédiction" not in note


def test_aucun_diagnostic_declare_favorable(client):
    """`not_assessed` pour les six, avec son motif. Jamais `favorable`."""
    diagnostics = client.post("/recommend",
                              json=CONTEXTE).json()["agronomic_diagnostics"]
    assert len(diagnostics) == 6
    for diagnostic in diagnostics:
        assert diagnostic["status"] == "not_assessed"
        assert diagnostic["reason"]


# --------------------------------------------------------------------------- #
# /recommend — le classement économique
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("budgets, cas", [
    ({}, "aucun budget"),
    ({"Rice": BUDGET_RIZ}, "un seul budget"),
])
def test_classement_economique_absent_sous_deux_budgets(client, budgets, cas):
    """Sous deux budgets, aucune marge n'est comparable : le bloc est absent."""
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": budgets}).json()
    assert corps["economic_ranking"] is None, cas
    assert corps["recommendation"]["status"] == "undetermined"
    assert corps["recommendation"]["crops"] == []
    assert corps["data_quality"]["economic_assumptions_version"] is None


def test_budget_partiel_rejete(client):
    """Complet ou absent : une charge manquante produirait une marge fausse."""
    incomplet = {cle: valeur for cle, valeur in BUDGET_RIZ.items()
                 if cle != "price_eur_t"}
    reponse = client.post("/recommend", json={
        **CONTEXTE, "budgets": {"Rice": incomplet, "Wheat": BUDGET_BLE}})
    assert reponse.status_code == 422


def test_budget_limite_a_trois_champs(client):
    """Le budget n'accepte que rendement, charges et prix.

    `system` dupliquerait `irrigation_used`, déjà fixé pour toute la requête,
    et permettrait de décrire une parcelle à la fois irriguée et pluviale.
    `product` dupliquerait `crop`. Les deux sont refusés.
    """
    assert set(BUDGET_RIZ) == {"expected_yield_t_ha", "variable_costs_eur_ha",
                               "price_eur_t"}
    for champ, valeur in (("system", "irrigué"), ("product", "riz paddy")):
        reponse = client.post("/recommend", json={
            **CONTEXTE,
            "budgets": {"Rice": {**BUDGET_RIZ, champ: valeur},
                        "Wheat": BUDGET_BLE}})
        assert reponse.status_code == 422, champ


def test_budget_sur_culture_inconnue_rejete(client):
    """Une clé de budget hors énumération est refusée, pas ignorée."""
    reponse = client.post("/recommend", json={
        **CONTEXTE, "budgets": {"Quinoa": BUDGET_RIZ, "Wheat": BUDGET_BLE}})
    assert reponse.status_code == 422


def test_marge_brute_suit_la_convention(client):
    """marge = prix × rendement − charges, et le budget revient avec elle.

    Un chiffre économique sans ses hypothèses ne serait pas vérifiable : la
    réponse renvoie donc le budget qui a produit chaque marge.
    """
    budgets = {"Rice": BUDGET_RIZ, "Wheat": BUDGET_BLE}
    eco = client.post("/recommend", json={**CONTEXTE, "budgets": budgets}
                      ).json()["economic_ranking"]

    attendu = {culture: (b["price_eur_t"] * b["expected_yield_t_ha"]
                         - b["variable_costs_eur_ha"])
               for culture, b in budgets.items()}
    obtenu = {m["crop"]: m["gross_margin_eur_ha"] for m in eco["margins"]}
    assert obtenu == pytest.approx(attendu)

    for marge in eco["margins"]:
        source = budgets[marge["crop"]]
        assert marge["source"] == "user"
        assert marge["price_eur_t"] == source["price_eur_t"]
        assert marge["expected_yield_t_ha"] == source["expected_yield_t_ha"]
        assert marge["variable_costs_eur_ha"] == source["variable_costs_eur_ha"]


def test_cultures_non_evaluees_nommees(client):
    """Deux budgets sur six : l'absence est nommée, pas déduite.

    C'est ce qui empêche une interface de titrer « culture recommandée : riz »
    quand quatre cultures n'ont fait l'objet d'aucune comparaison.
    """
    budgets = {"Rice": BUDGET_RIZ, "Wheat": BUDGET_BLE}
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": budgets}).json()
    eco = corps["economic_ranking"]

    assert eco["assessed_crops"] == ["Rice", "Wheat"]
    assert set(eco["unassessed_crops"]) == set(CULTURES) - {"Rice", "Wheat"}
    assert set(eco["assessed_crops"]).isdisjoint(eco["unassessed_crops"])
    for culture in eco["unassessed_crops"]:
        assert culture in eco["note"]
        assert culture in corps["recommendation"]["rationale"]


def test_recommandation_ne_pretend_jamais_au_meilleur_des_six(client):
    """Une recommandation sur deux budgets ne porte pas sur six cultures."""
    budgets = {"Rice": BUDGET_RIZ, "Wheat": BUDGET_BLE}
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": budgets}).json()
    rationale = corps["recommendation"]["rationale"]

    assert "parmi les 2 cultures évaluées" in rationale
    assert "n'ont pas été comparées" in rationale
    assert "six" not in rationale.lower()


def test_sensibilite_signale_un_basculement(client):
    """Quand un scénario de prix renverse l'ordre, aucun gagnant n'est désigné."""
    corps = client.post("/recommend",
                        json={**CONTEXTE,
                              "budgets": BUDGETS_QUI_BASCULENT}).json()
    sensibilite = corps["economic_ranking"]["price_sensitivity"]

    assert sensibilite["ranking_stable"] is False
    assert sensibilite["indistinguishable_pairs"] == [["Rice", "Wheat"]]
    assert corps["recommendation"]["status"] == "options"
    assert set(corps["recommendation"]["crops"]) == {"Rice", "Wheat"}
    assert "ne sont pas départagées" in corps["recommendation"]["rationale"]


def test_trois_cultures_indistinctes_sont_toutes_nommees(client):
    """Trois marges égales : l'avis les nomme toutes, sans élire de gagnant.

    Le cas manquait, et la couverture de branches l'a révélé. Avec deux
    budgets il n'existe qu'une paire, et elle contient nécessairement la tête
    du classement : la boucle qui élargit l'avis ne tournait qu'une fois et ne
    rencontrait jamais de paire à écarter.

    Trois budgets produisent trois paires : deux contiennent la tête et
    élargissent l'avis, la troisième ne la contient pas et doit être ignorée
    — sans quoi l'avis annoncerait une culture qui n'est pas en concurrence
    avec celle qu'il met en avant.
    """
    corps = client.post("/recommend",
                        json={**CONTEXTE,
                              "budgets": BUDGETS_INDISTINCTS}).json()
    eco = corps["economic_ranking"]
    avis = corps["recommendation"]

    # Les trois marges valent 500 €/ha : aucune ne départage.
    assert {marge["gross_margin_eur_ha"] for marge in eco["margins"]} == {500.0}

    sensibilite = eco["price_sensitivity"]
    assert sensibilite["ranking_stable"] is False
    assert len(sensibilite["indistinguishable_pairs"]) == 3

    assert avis["status"] == "options"
    assert set(avis["crops"]) == set(BUDGETS_INDISTINCTS)
    assert "ne sont pas départagées" in avis["rationale"]


def test_sensibilite_stable_quand_l_ecart_est_large(client):
    """Un écart large survit aux deux scénarios : un seul nom est avancé."""
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": {"Rice": BUDGET_RIZ,
                                                      "Wheat": BUDGET_BLE}}).json()
    sensibilite = corps["economic_ranking"]["price_sensitivity"]

    assert sensibilite["ranking_stable"] is True
    assert sensibilite["indistinguishable_pairs"] == []
    assert sensibilite["tested_variation"] == pytest.approx(0.2)
    assert corps["recommendation"]["crops"] == ["Rice"]


def test_convention_de_sensibilite_dans_la_reponse(client):
    """La convention part dans la réponse, pas seulement dans le code.

    L'appelant ne doit pas avoir à lire la documentation pour savoir ce qui a
    varié, ni pour comprendre que ±20 % est un scénario et non une fourchette
    de marché.
    """
    sensibilite = client.post(
        "/recommend", json={**CONTEXTE, "budgets": {"Rice": BUDGET_RIZ,
                                                    "Wheat": BUDGET_BLE}}
    ).json()["economic_ranking"]["price_sensitivity"]
    convention = sensibilite["convention"]

    assert "simultanément" in convention
    assert "rendements et charges constants" in convention
    assert "fourchette de marché" in convention


def test_marges_egales_non_departagees(client):
    """Deux marges identiques ne produisent pas un gagnant arbitraire.

    L'ordre ne bascule pas — il n'y a rien à faire basculer — mais la paire
    n'est pas départagée pour autant. Les deux faits sont distincts et la
    réponse les sépare.
    """
    budgets = {"Rice": budget(5.0, 400.0, 200.0),
               "Wheat": budget(5.0, 400.0, 200.0)}
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": budgets}).json()
    sensibilite = corps["economic_ranking"]["price_sensitivity"]

    assert sensibilite["ranking_stable"] is True
    assert sensibilite["indistinguishable_pairs"] == [["Rice", "Wheat"]]
    assert set(corps["recommendation"]["crops"]) == {"Rice", "Wheat"}


def test_six_budgets_classement_exhaustif(client):
    """Avec les six budgets, et seulement alors, le classement est exhaustif."""
    budgets = {culture: budget(3.0 + i * 0.5, 400.0 + i * 20, 200.0)
               for i, culture in enumerate(CULTURES)}
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": budgets}).json()
    eco = corps["economic_ranking"]

    assert eco["unassessed_crops"] == []
    assert len(eco["margins"]) == 6
    assert "exhaustif" in eco["note"]
    marges = [m["gross_margin_eur_ha"] for m in eco["margins"]]
    assert marges == sorted(marges, reverse=True)
    assert "n'ont pas été comparées" not in corps["recommendation"]["rationale"]


def test_version_des_hypotheses_seulement_avec_des_marges(client):
    """La version d'hypothèses identifie la convention de calcul, pas des prix."""
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": {"Rice": BUDGET_RIZ,
                                                      "Wheat": BUDGET_BLE}}).json()
    assert corps["data_quality"]["economic_assumptions_version"] is not None
    assert any("budgets sont ceux fournis par l'appelant" in ligne
               for ligne in corps["data_quality"]["limitations"])


# --------------------------------------------------------------------------- #
# Chargement du modèle
# --------------------------------------------------------------------------- #


def test_uri_non_supportee_refusee():
    """Seules les URIs `models:/` sont acceptées, et le refus est explicite."""
    with pytest.raises(ValueError, match="models:/"):
        service.charger_modele("runs:/abc/model")


def test_incoherence_registre_provenance_refusee(provenance):
    """Servir une version que la provenance ne décrit pas est un refus.

    Les deux sortent du même export : un écart signifie que l'alias a bougé
    sans réexport, et la réponse estampillerait alors une provenance fausse.
    Refuser de démarrer vaut mieux que de mal tracer.
    """
    with pytest.raises(RuntimeError, match="Réexécuter"):
        service._exiger_coherence(provenance, "autre-registre", 99)
    service._exiger_coherence(provenance, provenance["registre"],
                              provenance["version"])


@pytest.mark.skipif(not MLFLOW_DB_PATH.exists(),
                    reason="registre MLflow local absent")
def test_parite_entre_artefact_embarque_et_registre(provenance):
    """Les deux sources de chargement rendent la même prédiction.

    L'image sert le joblib embarqué, le poste de développement peut servir le
    registre. Un écart ici voudrait dire que l'API ne prédit pas la même chose
    selon l'endroit où elle tourne.
    """
    embarque = service.charger_modele()
    registre = service.charger_modele(
        f"models:/{provenance['registre']}/{provenance['version']}")

    assert embarque.served.source == "bundled-joblib"
    assert registre.served.source == "mlflow-registry"
    assert embarque.served.version == registre.served.version

    requete = PredictionRequete(**CONTEXTE, crop="Wheat")
    assert (service.construire_prediction(embarque, requete).yield_t_ha
            == service.construire_prediction(registre, requete).yield_t_ha)


@pytest.fixture
def artefact_copie(tmp_path, monkeypatch):
    """Une copie de l'artefact, que les tests peuvent abîmer.

    `models/modele_api.joblib` est l'artefact promu : l'altérer, même le temps
    d'un test, casserait la suite entière et toute exécution ultérieure.
    """
    copie = tmp_path / API_MODEL_PATH.name
    copie.write_bytes(API_MODEL_PATH.read_bytes())
    monkeypatch.setattr(service, "API_MODEL_PATH", copie)
    return copie


def test_sceau_accepte_l_artefact_decrit(artefact_copie, provenance):
    """Le cas nominal : l'empreinte du fichier est bien celle déclarée."""
    service._exiger_sceau(provenance)


def test_sceau_refuse_un_artefact_altere(artefact_copie, provenance):
    """Un octet de plus suffit à faire échouer le démarrage.

    C'est le scénario du conteneur : un `.joblib` périmé resté d'une couche
    antérieure, ou recopié d'une autre image. Il servirait des prédictions en
    annonçant une provenance qui n'est pas la sienne.
    """
    artefact_copie.write_bytes(artefact_copie.read_bytes() + b"\x00")

    with pytest.raises(RuntimeError, match="n'est pas l'artefact décrit"):
        service._exiger_sceau(provenance)


def test_sceau_refuse_une_provenance_sans_empreinte(artefact_copie, provenance):
    """Une provenance antérieure au sceau ne passe pas en silence.

    L'accepter rendrait le contrôle inopérant sur exactement les images qu'il
    devrait refuser : celles construites avant qu'il existe.
    """
    sans_sceau = {cle: valeur for cle, valeur in provenance.items()
                  if cle != "artefact_sha256"}

    with pytest.raises(RuntimeError, match="ne porte pas d'empreinte"):
        service._exiger_sceau(sans_sceau)


def test_sceau_refuse_un_artefact_absent(artefact_copie, provenance):
    """Une image sans modèle échoue au démarrage, pas à la première requête."""
    artefact_copie.unlink()

    with pytest.raises(FileNotFoundError, match="Artefact absent"):
        service._exiger_sceau(provenance)


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #


class _RegistreFactice:
    """Registre MLflow minimal : retient l'alias posé, sans base ni réseau.

    Ce qui est éprouvé ici est l'**enchaînement** — concordance, sceau, pose
    de l'alias, écriture du fichier, reprise —, pas MLflow. Un vrai registre
    y ajouterait une base SQLite à créer et à nettoyer sans rien vérifier de
    plus, et la promotion réelle doit rester un geste manuel.
    """

    def __init__(self, tags: dict):
        """Les tags de l'unique version enregistrée, et aucun alias posé."""
        self.tags = tags
        self.alias: int | None = None
        self.poses = 0

    def __call__(self, *_args, **_kwargs):
        """Se substitue à la classe `MlflowClient`, que l'appelant instancie."""
        return self

    def get_model_version(self, _registre, _version):
        """Seuls les tags sont lus par la promotion."""
        return SimpleNamespace(tags=self.tags)

    def get_model_version_by_alias(self, _registre, _alias):
        """Lève comme MLflow quand l'alias n'est posé sur aucune version."""
        if self.alias is None:
            raise RuntimeError("aucun alias posé")
        return SimpleNamespace(version=self.alias)

    def set_registered_model_alias(self, _registre, _alias, version):
        """Compte les poses, pour distinguer une reprise d'une création."""
        self.alias = int(version)
        self.poses += 1


@pytest.fixture
def promotion(tmp_path, monkeypatch, provenance):
    """Une promotion isolée : copies du disque, registre factice.

    Ni `models/` ni `mlflow.db` ne sont touchés. La provenance réelle peut
    précéder le contrat courant — on part donc d'une copie complétée, et
    chaque test fausse seulement ce qu'il veut éprouver.
    """
    artefact = tmp_path / "modele_api.joblib"
    artefact.write_bytes(API_MODEL_PATH.read_bytes())

    complete = {**provenance, "alias": None,
                "artefact_sha256": export.empreinte(artefact)}
    metadonnees = tmp_path / "modele_api.json"

    def ecrire(**surcharges):
        """Réécrit la provenance de travail, éventuellement faussée."""
        metadonnees.write_text(
            json.dumps({**complete, **surcharges}, indent=1,
                       ensure_ascii=False) + "\n", encoding="utf-8")

    ecrire()
    monkeypatch.setattr(export, "API_METADATA_PATH", metadonnees)
    monkeypatch.setattr(export, "API_MODEL_PATH", artefact)

    registre = _RegistreFactice({cle: str(complete[cle])
                                 for cle in export.CLES_CONCORDANCE})
    monkeypatch.setattr("mlflow.tracking.MlflowClient", registre)

    return SimpleNamespace(
        ecrire=ecrire, registre=registre, json=metadonnees, joblib=artefact,
        lire=lambda: json.loads(metadonnees.read_text(encoding="utf-8")))


# `promouvoir` reçoit un module mlflow dont elle n'utilise rien : le paramètre
# est là pour obliger l'appelant à avoir configuré l'URI de suivi avant
# d'appeler. Les tests passent donc `None`.


def test_promotion_pose_l_alias_des_deux_cotes(promotion):
    """Le registre et le fichier portent le même alias après l'appel."""
    export.promouvoir(None)

    relue = promotion.lire()
    assert relue["alias"] == ALIAS_CHAMPION
    assert promotion.registre.alias == relue["version"]
    assert relue["alias_precedent"] is None


def test_promotion_refuse_une_provenance_incomplete(promotion):
    """Une provenance antérieure au contrat est nommée, pas plantée.

    Sans ce contrôle, une clé de traçabilité ajoutée au script ferait lever
    un `KeyError` nu sur un fichier plus ancien — illisible pour qui
    exécute la commande.
    """
    sans_sceau = {cle: valeur for cle, valeur in promotion.lire().items()
                  if cle != "artefact_sha256"}
    promotion.json.write_text(json.dumps(sans_sceau), encoding="utf-8")

    with pytest.raises(RuntimeError, match="artefact_sha256"):
        export.promouvoir(None)
    assert promotion.registre.alias is None


def test_promotion_refuse_une_provenance_discordante(promotion):
    """Promouvoir une version que la provenance ne décrit pas est un refus."""
    promotion.ecrire(data_version="AUTRE_VERSION")

    with pytest.raises(RuntimeError, match="ne décrit pas"):
        export.promouvoir(None)
    assert promotion.registre.alias is None


def test_promotion_refuse_un_artefact_remplace(promotion):
    """Le sceau porte sur le fichier présent, pas sur la valeur déclarée.

    C'est le seul des contrôles qui détecte un `.joblib` remplacé après
    l'export : les autres comparent deux déclarations, qui porteraient
    toutes deux la même valeur périmée.
    """
    promotion.joblib.write_bytes(promotion.joblib.read_bytes() + b"\x00")

    with pytest.raises(RuntimeError, match="n'est pas le fichier"):
        export.promouvoir(None)
    assert promotion.registre.alias is None


def test_promotion_refuse_un_artefact_absent(promotion):
    """Désigner une version dont le binaire manque laisserait l'image vide."""
    promotion.joblib.unlink()

    with pytest.raises(FileNotFoundError, match="Artefact absent"):
        export.promouvoir(None)
    assert promotion.registre.alias is None


def test_divergence_nommee_puis_reprise_convergente(promotion, monkeypatch):
    """L'écriture échoue après la pose de l'alias : c'est dit, et réparable.

    C'est le défaut que la promotion **ne peut pas** empêcher — deux
    systèmes, deux écritures successives, aucune transaction. Le script doit
    donc nommer l'état divergent et offrir une reprise, au lieu de
    revendiquer une atomicité qu'il n'a pas.
    """
    def refuser(_provenance):
        """Rend l'écriture du fichier impossible, l'alias étant déjà posé."""
        raise OSError(28, "No space left on device (simulé)")

    original = export._ecrire_provenance
    monkeypatch.setattr(export, "_ecrire_provenance", refuser)

    with pytest.raises(RuntimeError, match="DIVERGENCE") as echec:
        export.promouvoir(None)

    # Le message dit quoi faire, et ne promet pas l'atomicité.
    assert "idempotente" in str(echec.value)

    # L'alias est posé au registre ; le fichier l'ignore encore.
    assert promotion.registre.alias == promotion.lire()["version"]
    assert promotion.lire()["alias"] is None

    # Reprise : la même commande, et aucune version créée au passage.
    monkeypatch.setattr(export, "_ecrire_provenance", original)
    export.promouvoir(None)

    assert promotion.lire()["alias"] == ALIAS_CHAMPION
    assert promotion.registre.poses == 2


def test_promotion_relancee_conserve_la_version_precedente(promotion):
    """Le second passage enregistre la version que l'alias désignait déjà.

    Le registre conserve l'historique des versions, pas celui des
    déplacements d'alias : sans cette trace, retrouver la cible d'un retour
    arrière serait un travail manuel.
    """
    export.promouvoir(None)
    assert promotion.lire()["alias_precedent"] is None

    export.promouvoir(None)
    relue = promotion.lire()
    assert relue["alias"] == ALIAS_CHAMPION
    assert relue["alias_precedent"] == relue["version"]


def test_provenance_ecrite_sans_laisser_de_temporaire(promotion):
    """Le fichier intermédiaire ne survit pas au remplacement."""
    export.promouvoir(None)

    assert not list(promotion.json.parent.glob("*.tmp"))
    assert promotion.lire()["alias"] == ALIAS_CHAMPION


# --------------------------------------------------------------------------- #
# Supervision et documentation
# --------------------------------------------------------------------------- #


def test_health_rend_le_modele_charge(client):
    """Un 200 signifie que le modèle est en mémoire et a produit une sonde."""
    corps = client.get("/health").json()
    assert corps["status"] == "ok"
    assert corps["served_model"]["source"] == "bundled-joblib"


def test_model_info_fait_foi_sur_les_modalites(client, artefact):
    """La liste exposée est celle de l'encodeur ajusté, non une copie libre."""
    corps = client.get("/model/info").json()
    encodeur = _encodeur(artefact)
    attendu = {nom: list(categories) for nom, categories
               in zip(encodeur.feature_names_in_, encodeur.categories_)}

    assert corps["accepted_values"] == attendu
    assert (corps["prediction_interval"]["half_width_t_ha"]
            == DEMI_INTERVALLE_PREDICTION_95_T_HA)
    assert corps["provenance"]["registre"] == corps["served_model"]["registry"]


def test_openapi_documente_toutes_les_routes(client):
    """Les six routes et leurs modèles sont documentés.

    Liste écrite en dur, délibérément : une route ajoutée sans être annoncée
    ici fait échouer ce test, ce qui force à se demander si elle a bien sa
    place dans le contrat public.
    """
    schema = client.get("/openapi.json").json()
    assert sorted(schema["paths"]) == ["/health", "/logs", "/model/info",
                                       "/predict", "/recommend",
                                       "/reference/prices"]
    assert "PredictionRequete" in schema["components"]["schemas"]
    assert "crop" in schema["components"]["schemas"]["PredictionRequete"]["required"]


# --------------------------------------------------------------------------- #
# Prix de référence
# --------------------------------------------------------------------------- #


def test_prix_couvrent_exactement_les_cultures_du_modele(client):
    """Les six cultures du modèle, ni plus ni moins.

    Un prix pour une culture que le modèle ignore serait inexploitable ; une
    culture sans prix laisserait un champ vide dans l'interface, sans que
    rien ne le signale.
    """
    corps = client.get("/reference/prices").json()

    assert sorted(p["crop"] for p in corps["prices"]) == sorted(CULTURES)


def test_prix_exposent_leur_dispersion(client):
    """Quartiles ordonnés et assise déclarée, pour chaque culture.

    La médiane seule se lirait comme « le prix ». Les quartiles sont là pour
    que l'ampleur de la dispersion entre pays reste visible.
    """
    for prix in client.get("/reference/prices").json()["prices"]:
        assert prix["q1_eur_t"] <= prix["price_eur_t"] <= prix["q3_eur_t"], \
            f"quartiles incohérents pour {prix['crop']}"
        assert prix["reporting_countries"] > 0
        assert prix["faostat_item"]


def test_prix_annoncent_leur_provenance(client):
    """Un prix sans source n'est pas une donnée, c'est une affirmation."""
    source = client.get("/reference/prices").json()["source"]

    assert source["source"] == "FAOSTAT"
    assert source["year"] == PRIX_REFERENCE_SOURCE["annee"]
    assert source["usd_per_eur"] == PRIX_REFERENCE_SOURCE["taux_usd_par_eur"]


def test_prix_disent_qu_ils_ne_suffisent_pas_a_une_marge(client):
    """La réserve doit nommer ce qui manque : les charges.

    Sans cette mention, la route se lirait comme une source de marge, alors
    qu'elle ne fournit qu'un des trois champs d'un budget.
    """
    reserve = client.get("/reference/prices").json()["caveat"]

    assert "charges" in reserve
    assert "marge" in reserve
    assert "/recommend" in reserve


def test_prix_ne_dependent_pas_du_modele():
    """La route répond sans qu'aucun artefact ne soit chargé.

    Ce sont des données externes : les faire dépendre du modèle rendrait
    l'interface muette sur les prix le jour où l'artefact manque.
    """
    reponse = service.prix_reference()

    assert len(reponse.prices) == len(CULTURES)
    assert reponse.source.source == "FAOSTAT"


def test_prix_de_reference_alimentent_recommend(client):
    """Les deux routes s'enchaînent : prix de référence, puis comparaison.

    C'est le parcours prévu pour l'interface — pré-remplir les budgets avec
    les prix sourcés, puis laisser l'utilisateur ajuster. Un écart de nom ou
    d'unité entre les deux routes le casserait sans bruit.
    """
    prix = {p["crop"]: p["price_eur_t"]
            for p in client.get("/reference/prices").json()["prices"]}

    budgets = {culture: budget(rendement, charges, prix[culture])
               for culture, rendement, charges
               in (("Wheat", 6.0, 800.0), ("Maize", 9.0, 1100.0))}
    corps = client.post("/recommend",
                        json={**CONTEXTE, "budgets": budgets}).json()

    marges = {m["crop"]: m for m in corps["economic_ranking"]["margins"]}
    assert set(marges) == {"Maize", "Wheat"}
    assert marges["Wheat"]["price_eur_t"] == prix["Wheat"]


# --------------------------------------------------------------------------- #
# Journal des appels
# --------------------------------------------------------------------------- #


def test_journal_enregistre_les_deux_endpoints(client, journal_vide):
    """Date, entrée, sortie, durée : les appels métier sont tracés."""
    client.post("/predict", json={**CONTEXTE, "crop": "Wheat"})
    client.post("/recommend", json={**CONTEXTE, "budgets": {
        "Rice": BUDGET_RIZ, "Wheat": BUDGET_BLE}})

    corps = client.get("/logs").json()
    assert corps["stored"] == 2
    assert corps["returned"] == 2
    assert corps["capacity"] == journal.CAPACITE
    assert "pas un journal d'audit" in corps["caveat"]

    # Le plus récent d'abord : la recommandation a été appelée en second.
    recent, precedent = corps["entries"]
    assert (recent["endpoint"], precedent["endpoint"]) == ("/recommend",
                                                           "/predict")
    assert precedent["status_code"] == 200
    assert precedent["duration_ms"] > 0
    assert precedent["timestamp"].startswith("20")
    assert precedent["request"]["crop"] == "Wheat"
    assert precedent["response"]["yield_t_ha"] > 0
    assert (recent["response"]["economic_ranking"]["assessed_crops"]
            == ["Rice", "Wheat"])


def test_journal_capture_les_appels_rejetes(client, journal_vide):
    """Un 422 est journalisé : c'est lui qui dit ce que l'appelant rate.

    Un journal des seuls succès tairait exactement l'information utile à la
    supervision — quelles saisies échouent, et sur quel champ.
    """
    client.post("/predict", json={**CONTEXTE, "crop": "Quinoa"})
    entree = client.get("/logs").json()["entries"][0]

    assert entree["status_code"] == 422
    assert entree["request"]["crop"] == "Quinoa"
    assert entree["response"]["detail"][0]["loc"] == ["body", "crop"]


def test_journal_conserve_un_corps_illisible(client, journal_vide):
    """Un corps non-JSON est gardé en texte tronqué plutôt qu'écarté.

    C'est le cas qu'un middleware doit capturer et qu'un handler ne voit
    jamais : la requête n'atteint même pas la validation du modèle.
    """
    client.post("/predict", content=b"{ceci n'est pas du json",
                headers={"Content-Type": "application/json"})
    entree = client.get("/logs").json()["entries"][0]

    assert entree["status_code"] == 422
    assert "_non_json" in entree["request"]


def test_journal_ignore_les_routes_de_service(client, journal_vide):
    """`/health` noierait le journal, `/logs` s'auto-alimenterait."""
    client.get("/health")
    client.get("/model/info")
    client.get("/logs")
    assert client.get("/logs").json()["stored"] == 0


def test_journal_borne_a_sa_capacite(journal_vide):
    """Le tampon évince les plus anciens au lieu de croître sans fin."""
    for i in range(journal.CAPACITE + 50):
        journal_vide.append({"timestamp": f"t{i}", "endpoint": "/predict",
                             "status_code": 200, "duration_ms": 1.0,
                             "request": None, "response": None})

    assert len(journal_vide) == journal.CAPACITE
    assert [e["timestamp"] for e in journal.lire(3)] == ["t149", "t148", "t147"]


def test_journal_limite_respectee(client, journal_vide):
    """`limit` borne ce qui est rendu, pas ce qui est conservé."""
    for _ in range(5):
        client.post("/predict", json={**CONTEXTE, "crop": "Wheat"})

    corps = client.get("/logs?limit=2").json()
    assert (corps["returned"], corps["stored"]) == (2, 5)


@pytest.mark.parametrize("limite", [0, 101])
def test_journal_limite_hors_bornes_rejetee(client, limite):
    """`limit` est borné par le contrat, pas silencieusement corrigé."""
    assert client.get(f"/logs?limit={limite}").status_code == 422


def test_enumeration_des_cultures_dans_le_schema():
    """L'appelant voit les valeurs acceptées avant d'en essayer une mauvaise."""
    schema = json.dumps(PredictionRequete.model_json_schema())
    for culture in CULTURES:
        assert f'"{culture}"' in schema

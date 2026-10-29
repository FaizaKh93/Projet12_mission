"""L'interface tient-elle ses promesses sans l'API en face ?

**Pourquoi ces tests existent.** Un script Streamlit ne se teste pas en le
regardant tourner : les pannes qui comptent — API éteinte, délai dépassé,
saisie refusée — n'arrivent jamais pendant une démonstration. Elles arrivent
chez l'utilisateur.

**Deux niveaux.** Les fonctions HTTP se testent directement, avec `requests`
remplacé par un faux serveur. L'interface se teste par `AppTest`, le banc
d'essai officiel de Streamlit, qui exécute réellement `app.py` et rapporte
toute exception. Le faux serveur étant posé sur le module `requests`
lui-même, il vaut aussi pour le script exécuté par `AppTest`.

**Aucun de ces tests ne joint l'API.** Ils doivent passer en intégration
continue, où aucun conteneur ne tourne.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.app import (
    URL_API,
    ErreurAPI,
    CHARGES_EXEMPLE,
    bornes_stables,
    charges_modifiees,
    completer,
    construire_budgets,
    csv_resultats,
    fusionner_rendements,
    graphique_intervalle,
    info_modele,
    message_sensibilite,
    nombre_fr,
    predire,
    recommander,
    tableau_budgets,
    tableau_marges,
    traduire,
)

#: Un classement économique complet, tel que l'API le rend.
CLASSEMENT = {
    "margins": [
        {"crop": "Maize", "gross_margin_eur_ha": 1302.6,
         "expected_yield_t_ha": 9.0, "variable_costs_eur_ha": 1100.0,
         "price_eur_t": 267.0, "source": "user"},
        {"crop": "Wheat", "gross_margin_eur_ha": 612.5,
         "expected_yield_t_ha": 6.5, "variable_costs_eur_ha": 850.0,
         "price_eur_t": 225.0, "source": "user"},
    ],
    "assessed_crops": ["Maize", "Wheat"],
    "unassessed_crops": ["Barley", "Cotton", "Rice", "Soybean"],
    "price_sensitivity": {"tested_variation": 0.2, "convention": "±20 %",
                          "ranking_stable": True,
                          "indistinguishable_pairs": []},
    "note": "classement établi parmi 2 cultures",
}

CHEMIN_APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")

CULTURES = ["Barley", "Cotton", "Maize", "Rice", "Soybean", "Wheat"]

CONTRAT = {
    "served_model": {"registry": "agritech-rendement-api", "version": 2,
                     "source": "bundled-joblib"},
    # Recopié de la forme réelle de `/model/info` : le panneau technique et
    # l'en-tête du CSV lisent chacune de ces clés.
    "provenance": {
        "registre": "agritech-rendement-api", "version": 2,
        "alias": "champion", "modele": "ridge",
        "registre_source": "agritech-rendement", "version_source": "2",
        "run_source": "265a884eaed7478fb37e8f5ff4a18f7f",
        "schema_entree": "8-saisies", "data_version": "abc",
        "split_version": "def", "hypotheses_economiques": "v1-2026-10",
        "sklearn_version": "1.9.1", "pandas_version": "3.0.6",
        "numpy_version": "2.5.4", "artefact_sha256": "87f9452",
        "exporte_le": "2026-10-28T09:41:45+00:00",
        "alias_precedent": None, "promu_le": "2026-10-28T09:42:32+00:00",
    },
    "accepted_values": {
        "Crop": CULTURES,
        "Region": ["East", "North", "South", "West"],
        "Soil_Type": ["Chalky", "Clay", "Loam", "Peaty", "Sandy", "Silt"],
        "Weather_Condition": ["Cloudy", "Rainy", "Sunny"],
    },
    "training_domain": {"Rainfall_mm": {"min": 100.0, "max": 1000.0},
                        "Temperature_Celsius": {"min": 15.0, "max": 40.0}},
    "prediction_interval": {"half_width_t_ha": 0.979, "nominal_coverage": 0.95,
                            "measured_coverage": 0.9492},
}

QUALITE = {"training_data": "simulated", "data_version": "abc",
           "split_version": "def", "economic_assumptions_version": None,
           "warnings": [], "limitations": ["donnees simulees"]}

PREDICTION = {
    "yield_t_ha": 6.387,
    "prediction_interval_95": {"lower_t_ha": 5.408, "upper_t_ha": 7.366,
                               "nominal_coverage": 0.95},
    "data_quality": QUALITE,
    "served_model": CONTRAT["served_model"],
}

RECOMMANDATION = {
    "yield_ranking": [
        {"crop": culture, "yield_t_ha": 6.39,
         "prediction_interval_95": {"lower_t_ha": 5.41, "upper_t_ha": 7.37,
                                    "nominal_coverage": 0.95}}
        for culture in CULTURES
    ],
    "yield_ranking_note": "classement sans pouvoir discriminant",
    "yield_spread_t_ha": 0.006473,
    "ranking_discriminative": False,
    "agronomic_diagnostics": [],
    "economic_ranking": None,
    "recommendation": {"status": "undetermined", "crops": [],
                       "rationale": "moins de deux budgets"},
    "data_quality": QUALITE,
    "served_model": CONTRAT["served_model"],
}

PRIX = {
    "prices": [
        {"crop": culture, "price_eur_t": prix, "q1_eur_t": prix - 40,
         "q3_eur_t": prix + 60, "reporting_countries": 50,
         "faostat_item": culture}
        for culture, prix in zip(CULTURES, [213, 508, 267, 423, 450, 225])
    ],
    "source": {"source": "FAOSTAT", "dataset": "PP", "element": "5532",
               "year": 2024, "method": "mediane", "currency_conversion": "BCE",
               "usd_per_eur": 1.08238, "extracted_on": "2026-10-28"},
    "caveat": "prix indicatifs",
}

CONTEXTE = {"rainfall_mm": 650.0, "temperature_celsius": 22.0,
            "fertilizer_used": True, "irrigation_used": True}


# --------------------------------------------------------------------------- #
# Le faux serveur
# --------------------------------------------------------------------------- #


class FausseReponse:
    """Le minimum de `requests.Response` dont le code appelant se sert."""

    def __init__(self, charge, code: int = 200):
        self._charge = charge
        self.status_code = code
        self.ok = code < 400
        self.text = json.dumps(charge) if not isinstance(charge, str) else charge

    def json(self):
        if isinstance(self._charge, str):
            raise ValueError("corps non JSON")
        return self._charge


@pytest.fixture(autouse=True)
def cache_vide():
    """Vide le cache de l'interface avant chaque test.

    `_contrat()` et `_prix()` sont mémorisés cinq minutes — c'est voulu : les
    relire à chaque frappe d'un curseur ajouterait un aller-retour par
    interaction. Mais le cache survit d'une exécution d'`AppTest` à l'autre
    dans un même processus : sans ce nettoyage, un test de panne réseau
    trouverait le contrat déjà en mémoire et ne verrait aucune panne.
    """
    st.cache_data.clear()
    yield
    st.cache_data.clear()


@pytest.fixture
def serveur(monkeypatch):
    """Remplace `requests.request` et enregistre les appels reçus.

    Posé sur le module `requests`, il intercepte aussi bien les appels directs
    des tests que ceux du script exécuté par `AppTest`.
    """
    appels = []
    routes = {
        "/model/info": FausseReponse(CONTRAT),
        "/reference/prices": FausseReponse(PRIX),
        "/predict": FausseReponse(PREDICTION),
        "/recommend": FausseReponse(RECOMMANDATION),
    }

    def repondre(methode, url, json=None, timeout=None):
        appels.append({"methode": methode, "url": url, "corps": json})
        chemin = url.removeprefix(URL_API)
        return routes.get(chemin, FausseReponse({"detail": "inconnu"}, 404))

    monkeypatch.setattr(requests, "request", repondre)
    return {"appels": appels, "routes": routes}


@pytest.fixture
def panne(monkeypatch):
    """Fabrique une panne réseau à la demande."""
    def poser(exception):
        def lever(*_, **__):
            raise exception
        monkeypatch.setattr(requests, "request", lever)
    return poser


# --------------------------------------------------------------------------- #
# La couche HTTP
# --------------------------------------------------------------------------- #


def test_contrat_lu_depuis_l_api(serveur):
    """Les valeurs acceptées viennent de l'API, pas du code de l'interface."""
    assert info_modele()["accepted_values"]["Crop"] == CULTURES
    assert serveur["appels"][0]["url"].endswith("/model/info")


def test_predict_envoie_le_contexte_et_la_culture(serveur):
    """Le corps de la requête est exactement celui qu'attend le contrat."""
    predire(CONTEXTE, "Wheat")
    corps = serveur["appels"][-1]["corps"]

    assert corps == {**CONTEXTE, "crop": "Wheat"}
    assert serveur["appels"][-1]["methode"] == "POST"


def test_recommend_sans_budget_n_envoie_pas_la_cle(serveur):
    """L'API distingue « aucun budget » de « budgets vides ».

    Transmettre `budgets: {}` laisserait croire que l'appelant a évalué zéro
    culture, là où il n'a simplement rien à comparer.
    """
    recommander(CONTEXTE)
    assert "budgets" not in serveur["appels"][-1]["corps"]

    recommander(CONTEXTE, {})
    assert "budgets" not in serveur["appels"][-1]["corps"]


def test_recommend_transmet_les_budgets(serveur):
    """Les budgets complets partent tels quels."""
    budgets = {"Wheat": {"expected_yield_t_ha": 6.5,
                         "variable_costs_eur_ha": 850.0,
                         "price_eur_t": 225.0}}
    recommander(CONTEXTE, budgets)
    assert serveur["appels"][-1]["corps"]["budgets"] == budgets


def test_api_injoignable_donne_la_commande_de_demarrage(panne):
    """Un agriculteur doit lire quoi faire, pas une trace Python."""
    panne(requests.ConnectionError("refused"))

    with pytest.raises(ErreurAPI) as capture:
        info_modele()

    message = str(capture.value)
    assert URL_API in message
    assert "docker start" in message and "docker run" in message


def test_delai_depasse_est_nomme_comme_tel(panne):
    """Un délai dépassé n'est pas une erreur de saisie : le dire."""
    panne(requests.Timeout("too slow"))

    with pytest.raises(ErreurAPI, match="n'a pas répondu"):
        info_modele()


def test_422_nomme_le_champ_refuse(serveur):
    """Le message doit désigner la saisie fautive.

    Recracher le JSON de Pydantic serait exact et inutilisable. Le champ et
    la raison suffisent, et c'est tout ce que l'appelant peut corriger.
    """
    serveur["routes"]["/predict"] = FausseReponse(
        {"detail": [{"loc": ["body", "crop"],
                     "msg": "Input should be 'Barley', 'Cotton'…"}]}, 422)

    with pytest.raises(ErreurAPI) as capture:
        predire(CONTEXTE, "Quinoa")

    message = str(capture.value)
    assert "crop" in message
    assert "Input should be" in message


def test_erreur_serveur_remonte_le_code(serveur):
    """Un 500 n'est pas une faute de l'utilisateur, et ne doit pas en avoir l'air."""
    serveur["routes"]["/predict"] = FausseReponse({"detail": "boom"}, 500)

    with pytest.raises(ErreurAPI, match="500"):
        predire(CONTEXTE, "Wheat")


# --------------------------------------------------------------------------- #
# La construction des budgets
# --------------------------------------------------------------------------- #


def test_une_ligne_sans_charges_n_est_pas_budgetee():
    """Charges vides = culture non évaluée, jamais une marge inventée."""
    tableau = pd.DataFrame([
        {"culture": "Blé", "crop": "Wheat", "price_eur_t": 225.0,
         "expected_yield_t_ha": 6.5, "variable_costs_eur_ha": 850.0},
        {"culture": "Maïs", "crop": "Maize", "price_eur_t": 267.0,
         "expected_yield_t_ha": 9.0, "variable_costs_eur_ha": None},
    ])

    budgets = construire_budgets(tableau)

    assert set(budgets) == {"Wheat"}
    assert budgets["Wheat"]["variable_costs_eur_ha"] == 850.0


def test_des_charges_nulles_sont_un_budget_valide():
    """Zéro euro de charges est une saisie, pas une absence.

    Confondre les deux écarterait l'utilisateur qui déclare, à juste titre,
    ne rien dépenser sur une culture.
    """
    tableau = pd.DataFrame([
        {"culture": "Orge", "crop": "Barley", "price_eur_t": 213.0,
         "expected_yield_t_ha": 6.4, "variable_costs_eur_ha": 0.0},
    ])

    assert construire_budgets(tableau)["Barley"]["variable_costs_eur_ha"] == 0.0


def test_les_trois_colonnes_sont_preremplies():
    """Prix nommé, rendement prédit, charges d'exemple.

    Les charges ne viennent d'aucune source : le test fixe seulement qu'elles
    arrivent remplies et qu'elles diffèrent d'une culture à l'autre.
    """
    tableau = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)

    assert list(tableau["crop"]) == CULTURES
    assert tableau["price_eur_t"].tolist() == [213, 508, 267, 423, 450, 225]
    assert (tableau["expected_yield_t_ha"] == 6.39).all()
    assert tableau["variable_costs_eur_ha"].notna().all()
    assert tableau["variable_costs_eur_ha"].nunique() == len(CULTURES)


def test_une_culture_inconnue_arrive_sans_charges():
    """Rien n'est inventé pour une culture absente du dictionnaire.

    Si le modèle était réentraîné sur une septième culture, l'interface la
    laisserait vide plutôt que de lui prêter un chiffre.
    """
    classement = [*RECOMMANDATION["yield_ranking"],
                  {"crop": "Sorghum", "yield_t_ha": 5.1}]
    prix = {"prices": [*PRIX["prices"],
                       {"crop": "Sorghum", "price_eur_t": 190}]}

    tableau = tableau_budgets(classement, prix)

    ligne = tableau[tableau["crop"] == "Sorghum"].iloc[0]
    assert pd.isna(ligne["variable_costs_eur_ha"])
    assert "Sorghum" not in construire_budgets(tableau)
    # Une case vide qui n'avait pas d'exemple n'est pas une intervention.
    assert charges_modifiees(tableau) is False


def test_les_charges_d_exemple_intactes_sont_reconnues():
    """Le tableau sorti de `tableau_budgets` n'a, par définition, rien de saisi."""
    tableau = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)

    assert charges_modifiees(tableau) is False


@pytest.mark.parametrize("valeur", [700.0, None])
def test_une_charge_touchee_est_reconnue(valeur):
    """Saisir un chiffre comme vider la case comptent pour une intervention."""
    tableau = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)
    tableau.loc[tableau["crop"] == "Wheat", "variable_costs_eur_ha"] = valeur

    assert charges_modifiees(tableau) is True


def test_un_prix_modifie_par_l_utilisateur_est_celui_qui_part():
    """Les colonnes proposées restent modifiables, et la modification compte."""
    tableau = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)
    tableau.loc[tableau["crop"] == "Wheat", "price_eur_t"] = 300.0
    tableau.loc[tableau["crop"] == "Wheat", "expected_yield_t_ha"] = 8.0
    tableau.loc[tableau["crop"] == "Wheat", "variable_costs_eur_ha"] = 900.0

    budget = construire_budgets(tableau)["Wheat"]

    assert budget["price_eur_t"] == 300.0
    assert budget["expected_yield_t_ha"] == 8.0


# --------------------------------------------------------------------------- #
# L'interface
# --------------------------------------------------------------------------- #


def test_l_application_demarre_sans_exception(serveur):
    """Le premier contrôle : le script s'exécute entièrement."""
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()

    assert not application.exception
    assert application.title[0].value.startswith("Agritech Answers")


def test_les_curseurs_sont_bornes_par_le_domaine_de_l_api(serveur):
    """Les bornes ne sont écrites nulle part dans l'interface.

    Si le modèle était réentraîné sur un domaine plus large, les curseurs
    suivraient sans qu'une ligne de `app.py` change.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    bornes = {curseur.label: (curseur.min, curseur.max)
              for curseur in application.slider}

    assert bornes["Pluie de la saison (mm)"] == (100.0, 1000.0)
    assert bornes["Température moyenne (°C)"] == (15.0, 40.0)


def test_les_cultures_proposees_viennent_de_l_api(serveur):
    """Aucune liste de cultures n'est recopiée dans l'interface.

    Le menu expose les libellés traduits — c'est ce que l'utilisateur lit —
    mais leur nombre et leur ordre sont ceux de `/model/info`.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    menu = next(m for m in application.selectbox if m.label == "Culture")

    assert menu.options == [traduire(culture) for culture in CULTURES]


def test_une_prediction_affiche_le_rendement_et_sa_plage(serveur):
    """Le chiffre clair, et la fourchette qui l'accompagne toujours."""
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    application.button[0].click().run()

    assert not application.exception
    assert "6,39 t/ha" in application.metric[0].value
    assert any("5,4" in note.value and "7,4" in note.value
               for note in application.caption)
    # La couverture annoncée doit être rattachée aux données simulées.
    note = next(note.value for note in application.caption
                if "simulées" in note.value)
    assert "94,92 %" in note
    # La conversion décimale ne doit toucher que le nombre : appliquée à la
    # phrase entière, elle changeait aussi ses points en virgules.
    assert "n'en est que le centre.**" in note
    assert "sort de cette plage." in note


def test_la_fourchette_est_aussi_montree_en_barre(serveur):
    """Deux nombres dans une phrase se lisent comme une précision.

    Le graphique n'est pas décoratif : c'est lui qui porte l'incertitude. Les
    trois valeurs viennent de `/predict`, aucune n'est recalculée.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    application.button[0].click().run()

    graphiques = application.get("vega_lite_chart")
    assert graphiques, "l'intervalle doit être tracé"

    figure = graphique_intervalle(PREDICTION)
    donnees = figure.data.iloc[0]
    assert donnees["bas"] == PREDICTION["prediction_interval_95"]["lower_t_ha"]
    assert donnees["haut"] == PREDICTION["prediction_interval_95"]["upper_t_ha"]
    assert donnees["estimation"] == PREDICTION["yield_t_ha"]

    # La spécification réellement envoyée au navigateur : axe ancré à zéro,
    # sans quoi une plage étroite paraîtrait large.
    spec = figure.to_dict()
    assert spec["layer"][0]["encoding"]["x"]["scale"]["domain"][0] == 0


def test_le_classement_non_discriminant_est_signale(serveur):
    """La réserve de l'API remonte à l'écran, elle n'est pas filtrée."""
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    assert not application.exception
    assert any("discriminant" in avis.value for avis in application.warning)


def test_api_eteinte_arrete_l_application_avec_un_message(panne):
    """Sans contrat, aucune interface : un formulaire vide serait pire."""
    panne(requests.ConnectionError("refused"))
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()

    assert not application.exception
    assert application.error
    assert "docker start" in application.error[0].value


def test_les_sources_et_limites_sont_toujours_presentes(serveur):
    """La note sur les pesticides et les données simulées ne dépend d'aucun clic."""
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    textes = " ".join(bloc.value for bloc in application.markdown)

    assert "pesticides" in textes.lower()
    assert "simulées" in textes


def test_un_champ_facultatif_renseigne_part_dans_la_requete(serveur):
    """Choisir un type de sol doit changer ce que l'API reçoit."""
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    menu = next(m for m in application.selectbox if m.label == "Type de sol")
    menu.set_value(traduire("Loam")).run()
    application.button[0].click().run()

    corps = serveur["appels"][-1]["corps"]
    assert corps["soil_type"] == "Loam"


def test_un_avertissement_de_l_api_remonte_a_l_ecran(serveur):
    """Une saisie hors domaine ne doit pas se perdre entre l'API et l'écran."""
    serveur["routes"]["/predict"] = FausseReponse({
        **PREDICTION,
        "data_quality": {**QUALITE,
                         "warnings": ["out_of_domain : Rainfall_mm = 1400"]},
    })
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()
    application.button[0].click().run()

    assert any("out_of_domain" in avis.value for avis in application.warning)


def test_une_panne_pendant_la_prediction_est_affichee(serveur, monkeypatch):
    """Le contrat est chargé, puis l'API tombe : l'écran doit le dire.

    Ce chemin diffère de celui du démarrage — l'interface est déjà construite,
    et c'est l'appel de prédiction qui échoue.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()

    def tomber(*_, **__):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "request", tomber)
    application.button[0].click().run()

    assert not application.exception
    assert any("Impossible de joindre" in erreur.value
               for erreur in application.error)


# --------------------------------------------------------------------------- #
# Le classement économique
# --------------------------------------------------------------------------- #


def test_le_tableau_des_marges_garde_l_ordre_et_la_source():
    """L'ordre du classement vient de l'API, pas d'un tri de l'interface."""
    tableau = tableau_marges(CLASSEMENT)

    assert list(tableau["Culture"]) == ["Maïs", "Blé"]
    assert list(tableau["Marge brute (€/ha)"]) == [1303, 612]
    assert set(tableau["Source"]) == {"user"}


def test_aucune_alerte_quand_le_classement_resiste_aux_prix():
    """Une alerte systématique finirait par ne plus être lue."""
    assert message_sensibilite(CLASSEMENT["price_sensitivity"]) is None


def test_le_parcours_complet_mene_au_classement_economique(serveur):
    """Le parcours en deux temps, de bout en bout.

    Les rendements d'abord, les charges ensuite, les marges en troisième.
    C'est le chemin que suivra l'agriculteur, et le seul qui exerce
    l'affichage du classement économique.

    Les charges sont posées dans l'état du tableau éditable, comme Streamlit
    le fait quand l'utilisateur tape dans une cellule.
    """
    serveur["routes"]["/recommend"] = FausseReponse({
        **RECOMMANDATION,
        "economic_ranking": CLASSEMENT,
        "recommendation": {"status": "options", "crops": ["Maize"],
                           "rationale": "Maïs présente la marge la plus "
                                        "élevée parmi les 2 cultures évaluées"},
    })

    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    # Blé (ligne 0) et Maïs (ligne 2) reçoivent des charges saisies ; les
    # quatre autres gardent celles d'exemple, d'où les six budgets envoyés.
    application.session_state["editeur_budgets"] = {
        "edited_rows": {0: {"variable_costs_eur_ha": 700.0},
                        2: {"variable_costs_eur_ha": 1100.0}},
        "added_rows": [], "deleted_rows": []}
    application.run()

    calculer = next(bouton for bouton in application.button
                    if bouton.label.startswith("Calculer les marges"))
    assert "6 cultures" in calculer.label
    calculer.click().run()

    assert not application.exception
    assert application.session_state["economique"]["economic_ranking"]
    assert any("marge la plus élevée" in avis.value
               for avis in application.success)
    assert any("parmi 2 cultures" in note.value for note in application.caption)

    # Deux charges ayant été saisies, le classement n'est plus un exemple.
    assert application.session_state["sur_exemples"] is False
    assert not any("charges d'exemple" in alerte.value
                   for alerte in application.warning)


def test_changer_un_interrupteur_garde_les_charges_saisies(serveur):
    """Le parcours qui donne son sens à la section.

    Saisir ses charges, basculer l'irrigation, relancer : les rendements
    doivent être neufs et les charges intactes. Les ressaisir à chaque essai
    rendrait la comparaison inutilisable.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    # Charges saisies sur le blé (ligne 0).
    application.session_state["editeur_budgets"] = {
        "edited_rows": {0: {"variable_costs_eur_ha": 777.0}},
        "added_rows": [], "deleted_rows": []}
    application.run()

    # Un premier classement économique est obtenu, puis rendu caduc.
    serveur["routes"]["/recommend"] = FausseReponse({
        **RECOMMANDATION, "economic_ranking": CLASSEMENT,
        "recommendation": {"status": "options", "crops": ["Maize"],
                           "rationale": "marge la plus élevée"}})
    next(bouton for bouton in application.button
         if bouton.label.startswith("Calculer les marges")).click().run()
    assert "economique" in application.session_state

    # Seconde comparaison : l'irrigation a relevé les rendements.
    serveur["routes"]["/recommend"] = FausseReponse({
        **RECOMMANDATION, "economic_ranking": CLASSEMENT,
        "recommendation": {"status": "options", "crops": ["Maize"],
                           "rationale": "marge la plus élevée"},
        "yield_ranking": [{**ligne, "yield_t_ha": 7.59}
                          for ligne in RECOMMANDATION["yield_ranking"]]})
    application.toggle[1].set_value(True).run()
    application.button[0].click().run()

    assert not application.exception
    # Les marges ont été refaites toutes seules, sans second clic : c'est
    # l'effet du changement d'interrupteur qu'on veut voir sur les barres.
    assert "economique" in application.session_state
    # Le panneau de droite affiche donc des barres, et non l'invitation à
    # saisir des charges qui sont déjà là.
    assert not any("Saisissez vos charges" in note.value
                   for note in application.info)

    envoye = [appel for appel in serveur["appels"]
              if appel["corps"] and "budgets" in appel["corps"]][-1]["corps"]
    assert envoye["irrigation_used"] is True
    # La charge saisie a survécu au changement d'interrupteur…
    assert envoye["budgets"]["Wheat"]["variable_costs_eur_ha"] == 777.0
    # …et les rendements, eux, ont bien été remplacés.
    assert envoye["budgets"]["Wheat"]["expected_yield_t_ha"] == 7.59


def test_fusionner_conserve_les_charges_et_remplace_les_rendements():
    """La règle, isolée de Streamlit."""
    base = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)
    base.loc[base["crop"] == "Wheat", "variable_costs_eur_ha"] = 777.0

    neuf = [{**ligne, "yield_t_ha": 7.59}
            for ligne in RECOMMANDATION["yield_ranking"]]
    fusionne = fusionner_rendements(base, neuf, PRIX)

    assert (fusionne["expected_yield_t_ha"] == 7.59).all()
    ligne = fusionne[fusionne["crop"] == "Wheat"].iloc[0]
    assert ligne["variable_costs_eur_ha"] == 777.0


def test_fusionner_repart_de_zero_si_les_cultures_changent():
    """Un modèle réentraîné sur d'autres cultures invalide le tableau."""
    base = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)
    base.loc[base["crop"] == "Wheat", "variable_costs_eur_ha"] = 777.0

    neuf = [{"crop": "Wheat", "yield_t_ha": 7.0}]
    fusionne = fusionner_rendements(
        neuf and base, neuf,
        {"prices": [{"crop": "Wheat", "price_eur_t": 225}]})

    assert list(fusionne["crop"]) == ["Wheat"]
    assert fusionne.iloc[0]["variable_costs_eur_ha"] == CHARGES_EXEMPLE["Wheat"]


def test_un_classement_sur_les_charges_intactes_est_signale(serveur):
    """Sans une seule charge saisie, l'écran refuse de faire passer le
    classement pour un conseil."""
    serveur["routes"]["/recommend"] = FausseReponse({
        **RECOMMANDATION,
        "economic_ranking": CLASSEMENT,
        "recommendation": {"status": "options", "crops": ["Maize"],
                           "rationale": "Maïs présente la marge la plus "
                                        "élevée parmi les 2 cultures évaluées"},
    })

    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    calculer = next(bouton for bouton in application.button
                    if bouton.label.startswith("Calculer les marges"))
    calculer.click().run()

    assert application.session_state["sur_exemples"] is True
    assert any("charges d'exemple" in alerte.value
               for alerte in application.warning)


def test_vider_les_charges_ramene_sous_le_seuil_de_comparaison(serveur):
    """Cinq cases vidées : il ne reste qu'un budget, et l'API n'est pas appelée.

    C'est le chemin inverse du précédent — l'utilisateur écarte les cultures
    qui ne l'intéressent pas et descend sous les deux budgets exigés.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    # Tout sauf le maïs (ligne 2).
    application.session_state["editeur_budgets"] = {
        "edited_rows": {ligne: {"variable_costs_eur_ha": None}
                        for ligne in (0, 1, 3, 4, 5)},
        "added_rows": [], "deleted_rows": []}
    application.run()

    assert any("1 culture(s) budgétée(s)" in note.value
               for note in application.info)
    assert not any(bouton.label.startswith("Calculer les marges")
                   for bouton in application.button)


def test_l_echelle_monte_mais_ne_redescend_jamais():
    """Une échelle qui se recale rend les comparaisons impossibles.

    C'est tout l'intérêt de figer : une marge qui baisse doit se voir baisser,
    et non réoccuper la même largeur sur une échelle rétrécie.
    """
    depart = bornes_stables((0.0, 0.0), [1400.0, 3100.0, 2600.0], 200.0)
    assert depart == (0.0, 3400.0)

    # Toutes les valeurs s'effondrent : l'échelle ne bouge pas.
    assert bornes_stables(depart, [200.0, 300.0, 250.0], 200.0) == depart

    # Une valeur dépasse : elle monte, pour ne rien tronquer.
    assert bornes_stables(depart, [3500.0], 200.0) == (0.0, 3800.0)


def test_l_echelle_descend_sous_zero_pour_une_marge_negative():
    """Des charges supérieures au produit brut donnent une marge négative.

    La masquer en coupant l'axe à zéro serait le pire des affichages : la
    culture paraîtrait simplement sans marge.
    """
    bas, haut = bornes_stables((0.0, 3400.0), [-850.0, 1200.0], 200.0)

    assert bas <= -850.0
    assert haut == 3400.0


def test_les_deux_panneaux_portent_leur_unite_dans_le_titre(serveur):
    """`st.bar_chart` intervertit les axes quand les barres sont
    horizontales : l'unité finissait du côté des cultures."""
    serveur["routes"]["/recommend"] = FausseReponse({
        **RECOMMANDATION, "economic_ranking": CLASSEMENT,
        "recommendation": {"status": "options", "crops": ["Maize"],
                           "rationale": "marge la plus élevée"}})
    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()
    next(bouton for bouton in application.button
         if bouton.label.startswith("Calculer les marges")).click().run()

    titres = [bloc.value for bloc in application.markdown]
    assert "**Rendement attendu (t/ha)**" in titres
    assert "**Marge brute simulée (€/ha)**" in titres

    # Les deux panneaux gardent une échelle propre, mémorisée entre deux
    # réponses de l'API.
    assert application.session_state["echelle_rendement"][1] > 0
    assert application.session_state["echelle_marge"][1] > 0


@pytest.mark.parametrize("valeur, decimales, attendu", [
    (1400.0, 0, "1 400"),
    (6.39, 2, "6,39"),
    (-850.0, 0, "-850"),
    (3400.0, 0, "3 400"),
])
def test_les_nombres_suivent_la_convention_francaise(valeur, decimales,
                                                     attendu):
    """Vega-Lite écrirait « 1,400 » : virgule des milliers, point décimal."""
    assert nombre_fr(valeur, decimales) == attendu


def test_une_culture_sans_budget_est_nommee_et_non_omise(serveur):
    """Omise, elle disparaîtrait du débat ; à zéro, elle paraîtrait sans
    marge. Le défaut avait été corrigé sur la figure du rapport, pas ici."""
    deux = {**CLASSEMENT,
            "margins": CLASSEMENT["margins"][:2],
            "not_evaluated": ["Cotton", "Rice", "Soybean", "Wheat"]}
    serveur["routes"]["/recommend"] = FausseReponse({
        **RECOMMANDATION, "economic_ranking": deux,
        "recommendation": {"status": "options", "crops": ["Maize"],
                           "rationale": "marge la plus élevée"}})

    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()
    next(bouton for bouton in application.button
         if bouton.label.startswith("Calculer les marges")).click().run()

    assert not application.exception
    # Les deux panneaux sont tracés, celui des marges compris.
    assert len(application.get("vega_lite_chart")) == 2


def test_les_graduations_suivent_la_convention_francaise(serveur):
    """Vega-Lite graduerait « 1,200 » et « 0.5 ».

    Les valeurs écrites à côté des barres sont formatées en Python ; les
    graduations, elles, sont rendues par le navigateur. Sans cette locale,
    les deux conventions se côtoieraient sur le même graphique.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    spec = json.loads(application.get("vega_lite_chart")[0].proto.spec)

    assert spec["config"]["locale"]["number"]["decimal"] == ","
    assert spec["config"]["locale"]["number"]["thousands"] == " "


def test_l_echelle_se_reinitialise_a_la_demande(serveur):
    """Le cliquet ne se rétracte jamais seul : il faut pouvoir le rendre.

    Sans ce bouton, passer à un scénario sans rapport avec le précédent
    obligerait à recharger la page pour retrouver une échelle adaptée.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()
    assert application.session_state["echelle_rendement"][1] > 0

    remettre = next(bouton for bouton in application.button
                    if bouton.label.startswith("Réinitialiser l'échelle"))
    remettre.click().run()

    assert not application.exception
    # Elle est reconstruite sur la réponse en cours, pas laissée à zéro.
    assert application.session_state["echelle_rendement"] == (0.0, 7.0)


def test_une_marge_negative_fait_apparaitre_la_ligne_zero(serveur):
    """La limite entre gain et perte ne doit pas se confondre avec une
    graduation quelconque."""
    perte = {**CLASSEMENT, "margins": [
        {**CLASSEMENT["margins"][0], "gross_margin_eur_ha": -850.0},
        {**CLASSEMENT["margins"][1], "gross_margin_eur_ha": 1200.0}]}
    serveur["routes"]["/recommend"] = FausseReponse({
        **RECOMMANDATION, "economic_ranking": perte,
        "recommendation": {"status": "options", "crops": ["Maize"],
                           "rationale": "marge la plus élevée"}})

    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()
    next(bouton for bouton in application.button
         if bouton.label.startswith("Calculer les marges")).click().run()

    assert application.session_state["echelle_marge"][0] <= -850.0
    specs = [bloc.proto.spec for bloc in application.get("vega_lite_chart")]
    # Le panneau des rendements, lui, n'a rien de négatif à borner.
    assert sum('"type": "rule"' in spec for spec in specs) == 1


def test_une_culture_sans_valeur_garde_sa_ligne():
    """La règle, isolée du graphique.

    Omise, la culture disparaîtrait du débat ; ramenée à zéro, elle
    paraîtrait sans marge — un zéro se compare, une absence non. Le défaut
    avait été corrigé sur la figure du rapport, pas encore dans l'écran.
    """
    deux = pd.DataFrame([{"culture": "Maïs", "crop": "Maize", "marge": 1302.6},
                         {"culture": "Blé", "crop": "Wheat", "marge": 900.0}])
    etiquettes = [traduire(culture) for culture in CULTURES]

    complet = completer(deux, etiquettes, "marge", decimales=0)

    assert list(complet["culture"]) == etiquettes
    assert complet["marge"].isna().sum() == 4
    assert (complet["marge"] == 0).sum() == 0
    # Les absentes n'affichent aucun nombre ; les présentes, le leur.
    assert list(complet.loc[complet["marge"].isna(), "valeur"]) == [""] * 4
    assert "1 303" in list(complet["valeur"])


def test_le_csv_distingue_ce_qui_est_propose_de_ce_qui_est_saisi():
    """L'export doit être rattachable : version du modèle et origine des
    valeurs, sinon le fichier ne prouve rien une fois détaché de l'écran."""
    edite = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)
    edite.loc[edite["crop"] == "Wheat", "variable_costs_eur_ha"] = 777.0
    edite.loc[edite["crop"] == "Maize", "price_eur_t"] = 300.0

    texte = csv_resultats(RECOMMANDATION, edite, None, PRIX, CONTRAT)
    entete = [ligne for ligne in texte.splitlines() if ligne.startswith("#")]
    lignes = {ligne.split(";")[1]: ligne
              for ligne in texte.splitlines() if ";" in ligne}

    assert any("agritech-rendement-api v2" in ligne for ligne in entete)
    assert any("SIMULÉES" in ligne for ligne in entete)
    assert any("pas un conseil agronomique" in ligne for ligne in entete)
    assert "exemple de l'interface" in lignes["Barley"]
    assert "saisi" in lignes["Wheat"]
    assert "FAOSTAT 2024" in lignes["Barley"]
    assert "saisi" in lignes["Maize"]


def test_le_csv_laisse_vide_une_marge_non_evaluee():
    """Une marge absente n'est pas une marge nulle : un zéro se comparerait."""
    edite = tableau_budgets(RECOMMANDATION["yield_ranking"], PRIX)
    edite.loc[edite["crop"] == "Cotton", "variable_costs_eur_ha"] = None
    economique = {**RECOMMANDATION, "economic_ranking": CLASSEMENT}

    texte = csv_resultats(RECOMMANDATION, edite, economique, PRIX, CONTRAT)
    coton = next(ligne for ligne in texte.splitlines()
                 if ligne.startswith("Coton;"))

    assert coton.endswith(";;non")          # marge vide, puis « non »
    assert "0,0" not in coton.split(";")[-2]


def test_un_telechargement_est_propose_avec_le_classement(serveur):
    """Le parcours doit se terminer par quelque chose qu'on emporte."""
    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    assert application.download_button


def test_le_panneau_technique_separe_le_servi_de_la_promotion(serveur):
    """Trois blocs, et un bouton qui ne promet pas ce qu'il ne fait pas.

    Déplacer l'alias `champion` dans MLflow ne change rien à un service déjà
    lancé. L'écran doit le dire, sinon le panneau induit en erreur.
    """
    application = AppTest.from_file(CHEMIN_APP, default_timeout=30).run()

    titres = [bloc.value for bloc in application.markdown]
    assert any("sert en ce moment" in titre for titre in titres)
    assert any("vient cet artefact" in titre for titre in titres)
    assert any("registre enregistrait" in titre for titre in titres)

    assert any("ne recharge aucun modèle" in note.value
               and "bundled-joblib" in note.value
               for note in application.caption)
    relire = next(bouton for bouton in application.button
                  if bouton.label == "Relire les informations de l'API")

    # Le bouton vide le cache : l'API est réinterrogée là où elle ne l'aurait
    # pas été avant cinq minutes.
    avant = len([appel for appel in serveur["appels"]
                 if appel["url"].endswith("/model/info")])
    relire.click().run()

    assert not application.exception
    assert len([appel for appel in serveur["appels"]
                if appel["url"].endswith("/model/info")]) > avant


def test_une_culture_hors_du_catalogue_est_dite_saisie():
    """Sans proposition de référence, l'origine ne peut pas être affirmée."""
    classement = [{"crop": "Sorghum", "yield_t_ha": 5.1}]
    prix = {"prices": [{"crop": "Sorghum", "price_eur_t": 190}]}
    edite = tableau_budgets(classement, prix)
    edite.loc[:, "variable_costs_eur_ha"] = 400.0

    texte = csv_resultats({"yield_ranking": classement}, edite, None,
                          prix, CONTRAT)

    assert "saisi" in texte.splitlines()[-1]


def test_la_formule_et_sa_convention_precedent_la_saisie(serveur):
    """La marge brute est annoncée avant le tableau, et justifiée."""
    application = AppTest.from_file(CHEMIN_APP, default_timeout=60).run()
    application.radio[0].set_value("Recommandation").run()
    application.button[0].click().run()

    formules = [bloc.value for bloc in application.get("latex")]
    assert any("Marge brute" in formule and "produit brut" in formule
               for formule in formules)
    assert any("RICA" in note.value and "charges fixes" in note.value
               for note in application.caption)
    assert any("exemples, pas des références" in alerte.value
               for alerte in application.warning)


def test_une_alerte_nomme_les_cultures_non_departagees():
    """Quand l'ordre bascule, l'écran doit dire entre lesquelles."""
    alerte = message_sensibilite({
        "tested_variation": 0.2, "ranking_stable": False,
        "indistinguishable_pairs": [["Maize", "Wheat"]]})

    assert "Maïs et Blé" in alerte
    assert "20 %" in alerte


# --------------------------------------------------------------------------- #
# Les 422 mal formés
# --------------------------------------------------------------------------- #


def test_un_422_au_corps_illisible_reste_affichable(serveur):
    """Une passerelle peut renvoyer un 422 sans JSON ; l'écran doit tenir."""
    serveur["routes"]["/predict"] = FausseReponse("<html>refusé</html>", 422)

    with pytest.raises(ErreurAPI, match="Saisie refusée"):
        predire(CONTEXTE, "Wheat")


def test_un_422_au_detail_textuel_est_repris_tel_quel(serveur):
    """L'API renvoie parfois `detail` en texte, pas en liste d'erreurs."""
    serveur["routes"]["/predict"] = FausseReponse(
        {"detail": "culture inconnue"}, 422)

    with pytest.raises(ErreurAPI, match="culture inconnue"):
        predire(CONTEXTE, "Quinoa")

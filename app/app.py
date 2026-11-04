"""Interface Streamlit : elle interroge l'API, elle ne calcule rien.

**Trois modes, deux modèles.** « Prédiction » et « Recommandation » interrogent
le modèle de parcelle ; « Rendements nationaux » interroge celui du panel FAO,
qui prolonge l'historique d'un pays et d'une culture. Le second mode n'affiche
aucun formulaire de parcelle : son modèle n'en lit aucun champ.

**Aucune logique métier ici.** Pas de modèle, pas de règle de dérivation, pas
de seuil recopié. Tout vient de l'API — y compris la liste des cultures et les
bornes des curseurs, lues dans `/model/info`. Recopier ces valeurs créerait un
second endroit où une divergence peut s'installer, et une interface qui
proposerait une culture que le modèle ne connaît plus.

**Deux familles de fonctions, volontairement séparées.** Celles du haut ne
font que du HTTP : elles ignorent qu'une interface existe, et se testent sans
Streamlit. `main()` et ce qu'elle appelle ne font que de l'affichage. La
frontière est ce qui rend `tests/test_app.py` possible.

    uv run streamlit run app/app.py

L'adresse du service se règle par `AGRITECH_API_URL` — indispensable en
conteneur, où l'API ne s'appelle pas `localhost`.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Dialogue avec l'API — aucune dépendance à Streamlit sous cette ligne
# --------------------------------------------------------------------------- #

#: Adresse du service. `localhost` en développement ; dans un `docker compose`,
#: l'API porte le nom de son service, pas celui de la machine.
URL_API = os.environ.get("AGRITECH_API_URL", "http://localhost:8000").rstrip("/")

#: Lecture courte **en local** : une API qui ne répond pas en 10 s est en
#: panne, pas lente. Un hébergement gratuit renverse ce raisonnement — Render
#: endort le service après quinze minutes sans visite et met près d'une minute
#: à le réveiller. Le réglage se relève donc par l'environnement, sans toucher
#: au code ni pénaliser le développement.
DELAI_S = float(os.environ.get("AGRITECH_API_TIMEOUT_S", "10"))

#: Nombre d'essais avant de déclarer l'API injoignable. **Un seul en local** :
#: une API absente doit se voir tout de suite, pas après trois attentes. En
#: ligne, le premier appel réveille un conteneur endormi et les suivants
#: aboutissent — plusieurs essais courts valent mieux qu'un seul très long,
#: qui laisserait l'écran muet pendant deux minutes avant d'échouer.
TENTATIVES = max(1, int(os.environ.get("AGRITECH_API_RETRIES", "1")))

#: Pause entre deux essais. Le réveil n'est pas instantané : marteler le
#: service ne l'accélère pas.
PAUSE_S = 3.0

#: Codes rendus par l'**hébergeur** quand le service redémarre, et non par
#: l'application : la requête n'a pas été traitée, la rejouer ne risque donc
#: rien. Un 500 est au contraire une réponse de l'application — la redemander
#: rendrait la même erreur, en trois fois plus de temps.
PASSERELLE = (502, 503, 504)


class ErreurAPI(Exception):
    """Un échec d'appel, déjà traduit en message lisible.

    Le détail technique ne remonte pas jusqu'à l'agriculteur : il voit ce
    qu'il peut faire, pas une trace d'exception.
    """


def _apres(essais: int) -> str:
    """Ne nomme le nombre d'essais que s'il y en a eu plusieurs."""
    return "" if essais == 1 else f", après {essais} essais"


def _injoignable() -> str:
    """Le conseil dépend de l'endroit où l'API est censée se trouver.

    Dire « lancez le conteneur » à quelqu'un qui ouvre un lien public serait
    une instruction qu'il ne peut pas suivre, et qui laisse croire à une
    panne là où le service dort seulement.
    """
    if "localhost" in URL_API or "127.0.0.1" in URL_API:
        return (f"Impossible de joindre l'API à {URL_API}. Si le conteneur "
                "existe déjà : `docker start agritech-api`. Au premier "
                "lancement : `docker run -d --name agritech-api -p 8000:8000 "
                "agritech-api:v2`")
    return (f"Impossible de joindre l'API à {URL_API}. Le service est "
            "peut-être en veille : réessayez dans une minute.")


def _appeler(methode: str, chemin: str, corps: dict | None = None) -> dict:
    """Un appel HTTP, et une erreur parlante quand il échoue.

    **Seuls l'injoignable et le trop lent sont rejoués.** Un 4xx ou un 5xx
    est une réponse : la redemander ne la changerait pas, et masquerait une
    saisie fautive derrière trois attentes. Les routes appelées ici sont
    toutes sans effet de bord, donc rejouables sans risque de doublon.
    """
    url = f"{URL_API}{chemin}"
    for essai in range(1, TENTATIVES + 1):
        dernier = essai == TENTATIVES
        try:
            reponse = requests.request(methode, url, json=corps,
                                       timeout=DELAI_S)
        except requests.Timeout as erreur:
            if dernier:
                raise ErreurAPI(
                    f"L'API n'a pas répondu en {DELAI_S:.0f} secondes"
                    f"{_apres(essai)}."
                ) from erreur
        except requests.ConnectionError as erreur:
            if dernier:
                raise ErreurAPI(_injoignable()) from erreur
        else:
            if dernier or reponse.status_code not in PASSERELLE:
                break
        time.sleep(PAUSE_S)

    if reponse.status_code == 422:
        raise ErreurAPI(_traduire_422(reponse))
    if reponse.status_code in PASSERELLE:
        raise ErreurAPI(
            f"Le service n'est pas encore prêt (HTTP {reponse.status_code})"
            f"{_apres(TENTATIVES)}. Il redémarre peut-être : réessayez dans "
            "une minute."
        )
    if not reponse.ok:
        raise ErreurAPI(
            f"L'API a répondu {reponse.status_code}. "
            f"{reponse.text[:200]}"
        )
    return reponse.json()


def _traduire_422(reponse: requests.Response) -> str:
    """Nomme le champ refusé plutôt que de recracher le JSON de Pydantic.

    Un 422 vient toujours de la saisie. Dire *lequel* des champs pose
    problème est la seule information utile à l'appelant.
    """
    try:
        details = reponse.json().get("detail", [])
    except ValueError:
        return f"Saisie refusée par l'API : {reponse.text[:200]}"

    if isinstance(details, str):
        return f"Saisie refusée : {details}"

    messages = []
    for detail in details:
        champ = " → ".join(str(part) for part in detail.get("loc", [])[1:])
        messages.append(f"{champ or 'requête'} : {detail.get('msg', '')}")
    return "Saisie refusée par l'API. " + " ; ".join(messages)


def info_modele() -> dict:
    """Valeurs acceptées, domaine d'entraînement et provenance du modèle."""
    return _appeler("GET", "/model/info")


def prix_reference() -> dict:
    """Prix à la production sourcés, pour amorcer les budgets."""
    return _appeler("GET", "/reference/prices")


def predire(contexte: dict, culture: str) -> dict:
    """Le rendement d'une culture dans le contexte fourni."""
    return _appeler("POST", "/predict", {**contexte, "crop": culture})


def recommander(contexte: dict, budgets: dict | None = None) -> dict:
    """Les six cultures classées, et la comparaison économique si budgets.

    `budgets` reste absent du corps quand il est vide : l'API distingue
    « aucun budget » de « budgets vides », et c'est elle qui décide de rendre
    ou non le bloc économique.
    """
    corps = dict(contexte)
    if budgets:
        corps["budgets"] = budgets
    return _appeler("POST", "/recommend", corps)


def series_nationales() -> dict:
    """Les couples (pays, culture) que le service national sait prédire.

    C'est le catalogue de ce mode : les pays, leurs cultures et les libellés
    français viennent tous de là. Les recopier ici ferait proposer un couple
    que l'artefact ne connaît pas le jour où la table change.
    """
    return _appeler("GET", "/national/series")


def predire_national(pays: str, culture: str) -> dict:
    """Le rendement national attendu pour ce pays et cette culture."""
    return _appeler("POST", "/national/predict",
                    {"area": pays, "item": culture})


def classer_national(pays: str) -> dict:
    """Les cultures du pays, classées par rendement national estimé."""
    return _appeler("POST", "/national/ranking", {"area": pays})


def construire_budgets(tableau: pd.DataFrame) -> dict:
    """Ne retient que les lignes dont les charges sont renseignées.

    Une charge vide signifie « je n'ai pas évalué cette culture ». Lui
    substituer un zéro, ou la moyenne des autres, fabriquerait une marge que
    personne n'a fournie — et l'API classerait une culture sur un chiffre
    inventé.
    """
    budgets = {}
    for ligne in tableau.to_dict("records"):
        charges = ligne.get("variable_costs_eur_ha")
        if charges is None or pd.isna(charges):
            continue
        budgets[ligne["crop"]] = {
            "expected_yield_t_ha": float(ligne["expected_yield_t_ha"]),
            "variable_costs_eur_ha": float(charges),
            "price_eur_t": float(ligne["price_eur_t"]),
        }
    return budgets


# --------------------------------------------------------------------------- #
# Interface — tout ce qui suit dépend de Streamlit
# --------------------------------------------------------------------------- #

#: Les libellés du jeu de données sont en anglais ; l'utilisateur ne l'est pas.
EN_FRANCAIS = {
    "Barley": "Orge", "Cotton": "Coton", "Maize": "Maïs",
    "Rice": "Riz", "Soybean": "Soja", "Wheat": "Blé",
    "East": "Est", "North": "Nord", "South": "Sud", "West": "Ouest",
    "Chalky": "Calcaire", "Clay": "Argileux", "Loam": "Limoneux",
    "Peaty": "Tourbeux", "Sandy": "Sableux", "Silt": "Silteux",
    "Cloudy": "Nuageux", "Rainy": "Pluvieux", "Sunny": "Ensoleillé",
}

NON_RENSEIGNE = "— non renseigné —"


def traduire(valeur: str) -> str:
    """Le libellé français, ou la valeur telle quelle si elle n'est pas là."""
    return EN_FRANCAIS.get(valeur, valeur)


@st.cache_data(ttl=300, show_spinner=False)
def _contrat() -> dict:
    """Le contrat de l'API, relu toutes les cinq minutes.

    Mis en cache parce qu'il ne change qu'au redéploiement : le relire à
    chaque frappe d'un curseur ajouterait un aller-retour par interaction.
    """
    return info_modele()


@st.cache_data(ttl=300, show_spinner=False)
def _prix() -> dict:
    """Les prix de référence, mêmes raisons."""
    return prix_reference()


@st.cache_data(ttl=300, show_spinner=False)
def _series() -> dict:
    """Le catalogue national : 204 pays et leurs cultures.

    Mis en cache pour la même raison que le contrat, et avec plus de force
    encore : sans cela, changer de pays dans une liste déroulante relirait
    1 106 couples à chaque frappe.
    """
    return series_nationales()


def saisir_contexte(contrat: dict) -> dict:
    """Le formulaire de parcelle, bâti depuis le contrat de l'API."""
    domaine = contrat["training_domain"]
    valeurs = contrat["accepted_values"]

    colonne_gauche, colonne_droite = st.columns(2)
    with colonne_gauche:
        pluie = st.slider(
            "Pluie de la saison (mm)",
            min_value=float(domaine["Rainfall_mm"]["min"]),
            max_value=float(domaine["Rainfall_mm"]["max"]),
            value=650.0, step=10.0,
            help="Bornes du domaine sur lequel le modèle a été entraîné.")
        fertilise = st.toggle("Parcelle fertilisée", value=True)
    with colonne_droite:
        temperature = st.slider(
            "Température moyenne (°C)",
            min_value=float(domaine["Temperature_Celsius"]["min"]),
            max_value=float(domaine["Temperature_Celsius"]["max"]),
            value=22.0, step=0.5)
        irrigue = st.toggle("Parcelle irriguée", value=True)

    with st.expander("Champs facultatifs — région, sol, météo"):
        st.caption(
            "Les laisser vides ne change pratiquement rien : ces trois "
            "champs pèsent quelques kilogrammes par hectare, pour des "
            "raisons expliquées dans « Sources et limites » en bas de page. "
            "La réponse de l'API signale les champs laissés vides.")
        trois = st.columns(3)
        facultatifs = {}
        for colonne, (champ, libelle) in zip(trois, (
                ("Region", "Région"), ("Soil_Type", "Type de sol"),
                ("Weather_Condition", "Météo"))):
            with colonne:
                choix = st.selectbox(
                    libelle, [NON_RENSEIGNE, *valeurs[champ]],
                    format_func=traduire, key=f"facultatif_{champ}")
                if choix != NON_RENSEIGNE:
                    facultatifs[champ.lower()] = choix

    return {"rainfall_mm": pluie, "temperature_celsius": temperature,
            "fertilizer_used": fertilise, "irrigation_used": irrigue,
            **facultatifs}


def afficher_qualite(reponse: dict) -> None:
    """Les réserves que l'API attache à sa réponse, jamais masquées."""
    qualite = reponse["data_quality"]
    for avertissement in qualite.get("warnings", []):
        st.warning(avertissement)
    with st.expander("Ce que cette réponse ne garantit pas"):
        for limite in qualite.get("limitations", []):
            st.markdown(f"- {limite}")
        servi = reponse["served_model"]
        st.caption(
            f"Modèle servi : {servi['registry']} v{servi['version']} "
            f"({servi['source']}) · données {qualite['data_version']} · "
            f"découpage {qualite['split_version']}")


def bornes_stables(actuelles: tuple[float, float],
                   valeurs, pas: float) -> tuple[float, float]:
    """L'échelle s'élargit quand il le faut, et ne se rétracte jamais.

    Une échelle qui se recale sur chaque réponse rend les comparaisons
    impossibles : une valeur plus faible peut occuper la même largeur
    qu'avant. Figée, une marge qui baisse se voit baisser.

    Le zéro reste toujours dans le cadre, et le bas descend si une marge
    devient négative — ce qui arrive dès que les charges dépassent le produit
    brut, et qu'il serait trompeur de ne pas montrer.
    """
    bas, haut = actuelles
    haut = max(haut, math.ceil(max(valeurs) * 1.05 / pas) * pas)
    bas = min(bas, math.floor(min(min(valeurs), 0.0) * 1.05 / pas) * pas)
    return bas, haut


#: Convention française pour les graduations d'axe, que Vega-Lite écrirait
#: sinon « 1,200 » et « 0.5 » — deux conventions décimales sur le même écran
#: se remarquent. C'est une définition de locale d3 ; `currency` est exigé par
#: le format même si aucun axe n'affiche de monnaie.
LOCALE_FR = {"number": {"decimal": ",", "thousands": " ",
                        "grouping": [3], "currency": ["", " €"]}}


def nombre_fr(valeur: float, decimales: int = 0) -> str:
    """Virgule décimale et espace fine pour les milliers.

    Écrit à la main parce que le formatage de Vega-Lite suit la convention
    anglophone — « 1,400 » pour mille quatre cents — et qu'aucune locale
    française n'est installée côté navigateur.
    """
    return (f"{valeur:,.{decimales}f}"
            .replace(",", " ").replace(".", ","))


def completer(donnees: pd.DataFrame, etiquettes: list[str], champ: str,
              decimales: int) -> pd.DataFrame:
    """Une ligne par culture de l'ordre, même sans valeur.

    Une culture absente du classement économique doit garder sa ligne : sans
    elle, elle disparaîtrait du graphique, et l'utilisateur ne saurait pas
    qu'elle existe. Sa valeur reste vide — elle n'est pas ramenée à zéro, qui
    se comparerait.
    """
    complet = donnees.set_index("culture").reindex(etiquettes).reset_index()
    complet["valeur"] = [
        "" if pd.isna(valeur) else nombre_fr(valeur, decimales)
        for valeur in complet[champ]]
    complet["zero"] = 0.0
    return complet


def barres(donnees: pd.DataFrame, champ: str, titre: str, unite: str,
           couleur: str, ordre: list[str], pas: float,
           decimales: int = 0) -> None:
    """Un graphique à barres horizontal, à l'ordre et à l'échelle imposés.

    **L'ordre est le même sur les deux panneaux**, et il ne dépend pas des
    valeurs. Trier chacun par sa grandeur interdirait la lecture horizontale,
    et en interactif les barres sauteraient à chaque charge modifiée.

    **L'unité est dans le titre, pas sur un axe.** `st.bar_chart` intervertit
    les axes quand les barres sont horizontales — la documentation l'énonce —
    et le libellé des valeurs se retrouvait du côté des cultures.

    **Une culture absente est nommée, pas omise.** Sans sa ligne, elle
    disparaîtrait du débat ; avec une barre de longueur nulle, elle
    paraîtrait sans marge. Elle reçoit donc une mention à la place de sa
    barre.
    """
    st.markdown(f"**{titre} ({unite})**")
    etiquettes = [traduire(culture) for culture in ordre]
    complet = completer(donnees, etiquettes, champ, decimales)

    cle = f"echelle_{champ}"
    ancien = st.session_state.get(cle)
    bornes = bornes_stables(ancien or (0.0, 0.0),
                            complet[champ].dropna(), pas)
    st.session_state[cle] = bornes

    axe = alt.X(f"{champ}:Q", title=None,
                scale=alt.Scale(domain=list(bornes), nice=False))
    axe_y = alt.Y("culture:N", title=None, sort=etiquettes)

    figure = alt.layer(
        alt.Chart(complet).mark_bar(color=couleur, height=26).encode(
            x=axe, y=axe_y,
            tooltip=[alt.Tooltip("culture:N", title="Culture"),
                     alt.Tooltip("valeur:N", title=unite)]),
        # La valeur exacte à côté de la barre : six libellés se lisent, là
        # où une série dense ne le permettrait pas.
        alt.Chart(complet).mark_text(
            align="left", dx=6, fontSize=12, color="#2d3748").encode(
            x=axe, y=axe_y, text=alt.Text("valeur:N")),
        alt.Chart(complet[complet[champ].isna()]).mark_text(
            align="left", dx=6, fontSize=11, color="#718096",
            fontStyle="italic").encode(
            x=alt.X("zero:Q", title=None,
                    scale=alt.Scale(domain=list(bornes), nice=False)),
            y=axe_y, text=alt.value("non évaluée — budget incomplet")),
        # Une règle à zéro, seulement lorsqu'une marge négative a fait
        # descendre l'échelle : sans elle, la limite entre gain et perte se
        # confondrait avec une graduation quelconque. Une seule ligne de
        # données, sinon la règle serait tracée six fois.
        *([alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(
            color="#4a5568", strokeWidth=1.2).encode(
            x=alt.X("zero:Q", title=None,
                    scale=alt.Scale(domain=list(bornes), nice=False)))]
          if bornes[0] < 0 else []),
    ).properties(height=320).configure(locale=LOCALE_FR)
    st.altair_chart(figure, width="stretch")

    note = (f"Échelle figée de {nombre_fr(bornes[0])} à "
            f"{nombre_fr(bornes[1])} {unite} : elle s'élargit si une valeur "
            "la dépasse, elle ne se rétracte pas.")
    if ancien is not None and bornes != ancien:
        note += (f" **Elle vient d'être élargie** — l'ancienne, de "
                 f"{nombre_fr(ancien[0])} à {nombre_fr(ancien[1])}, aurait "
                 "tronqué une valeur.")
    st.caption(note)


def graphique_intervalle(reponse: dict) -> alt.LayerChart:
    """L'intervalle en barre, l'estimation en losange dessus.

    Deux nombres dans une phrase se lisent comme une précision ; une barre se
    lit comme une incertitude. C'est le seul message de ce graphique, d'où
    l'axe ancré à zéro : sans lui, une plage étroite paraîtrait large.

    Rien n'est calculé ici — les trois valeurs viennent telles quelles de
    `/predict`.
    """
    intervalle = reponse["prediction_interval_95"]
    donnees = pd.DataFrame([{
        "bas": intervalle["lower_t_ha"],
        "haut": intervalle["upper_t_ha"],
        "estimation": reponse["yield_t_ha"],
        "ligne": "",
        # Écrits à part : le formatage de Vega-Lite donnerait « 5.4 », et
        # deux conventions décimales sur le même écran se remarquent.
        "txt_bas": nombre_fr(intervalle["lower_t_ha"], 1),
        "txt_haut": nombre_fr(intervalle["upper_t_ha"], 1),
    }])
    echelle = alt.Scale(domain=[0, intervalle["upper_t_ha"] * 1.15], nice=False)
    # `tickCount` est indispensable ici : sans lui, un domaine non arrondi
    # fait graduer tous les 0,2 et l'axe devient illisible.
    axe = alt.Axis(title="t/ha", grid=True, tickCount=9)

    plage = alt.Chart(donnees).mark_bar(height=30, color="#bcd3ea").encode(
        x=alt.X("bas:Q", scale=echelle, axis=axe),
        x2=alt.X2("haut:Q"),
        y=alt.Y("ligne:N", title=None))
    # Losange plutôt que trait : un trait vertical se confondrait avec la
    # grille, et serait lu comme une seconde borne.
    centre = alt.Chart(donnees).mark_point(
        shape="diamond", size=260, filled=True, color="#2b6cb0").encode(
        x=alt.X("estimation:Q", scale=echelle, axis=axe),
        y=alt.Y("ligne:N", title=None))
    # Les bornes écrites **dans** la barre, à ses deux extrémités. Sous elle
    # elles tombaient sur les graduations et se lisaient en double ;
    # au-dessus, le bord du cadre les rognait.
    etiquettes = [
        alt.Chart(donnees).mark_text(
            dx=decalage, fontSize=13, color="#2d3748", align=alignement,
            fontWeight="bold").encode(
            x=alt.X(f"{champ}:Q", scale=echelle, axis=axe),
            y=alt.Y("ligne:N", title=None),
            text=alt.Text(f"txt_{champ}:N"))
        for champ, alignement, decalage in (("bas", "left", 6),
                                            ("haut", "right", -6))
    ]
    return (alt.layer(plage, centre, *etiquettes)
            .properties(height=110).configure(locale=LOCALE_FR))


def graphique_trajectoire(reponse: dict) -> alt.LayerChart:
    """Les trois relevés, puis la prévision et son intervalle.

    **C'est le graphique qui explique le modèle.** Il calcule une moyenne
    pondérée des trois dernières années : montrer la suite rend visible ce
    qu'il fait — il prolonge une trajectoire — là où un chiffre seul se
    lirait comme une mesure de l'année à venir.

    Les années sont en catégories et non en quantités : quatre points sur un
    axe continu produiraient des graduations décimales, et « 2016,5 » ne
    désigne rien.
    """
    observes = pd.DataFrame([
        {"annee": str(releve["year"]), "rendement": releve["yield_t_ha"]}
        for releve in reponse["history"]]).sort_values("annee")
    intervalle = reponse["prediction_interval_95"]
    prevision = pd.DataFrame([{
        "annee": str(reponse["target_year"]),
        "rendement": reponse["yield_t_ha"],
        "bas": intervalle["lower_t_ha"], "haut": intervalle["upper_t_ha"]}])

    annees = [*observes["annee"], *prevision["annee"]]
    haut = max(intervalle["upper_t_ha"], observes["rendement"].max()) * 1.1
    axe_x = alt.X("annee:N", title=None, sort=annees)
    # L'échelle est construite à part : les deux couches doivent partager le
    # même domaine, et relire celui d'un encodage déjà posé ne rend pas un
    # objet réutilisable.
    echelle = alt.Scale(domain=[0, haut], nice=False)
    axe_y = alt.Y("rendement:Q", title="t/ha", scale=echelle)

    trace = alt.Chart(observes).mark_line(
        color="#2f855a", strokeWidth=2, point=alt.OverlayMarkDef(
            color="#2f855a", size=90)).encode(x=axe_x, y=axe_y)
    # L'intervalle en barre verticale : une incertitude se lit comme une
    # étendue, jamais comme deux nombres dans une phrase.
    plage = alt.Chart(prevision).mark_rule(
        strokeWidth=12, color="#bcd3ea").encode(
        x=axe_x, y=alt.Y("bas:Q", scale=echelle, title="t/ha"),
        y2=alt.Y2("haut:Q"))
    centre = alt.Chart(prevision).mark_point(
        shape="diamond", size=260, filled=True, color="#2b6cb0").encode(
        x=axe_x, y=axe_y)
    return (alt.layer(plage, trace, centre)
            .properties(height=260).configure(locale=LOCALE_FR))


def graphique_classement(classement: dict) -> alt.LayerChart:
    """Les cultures du pays en barres, avec leur dernier relevé en repère.

    L'ordre est celui de l'API — le classement **est** le message. Le point
    noir rappelle le rendement effectivement observé : quand il tombe loin de
    la barre, c'est que la série est accidentée et que le modèle revient vers
    sa moyenne.
    """
    donnees = pd.DataFrame([{
        "culture": rang["item_fr"],
        "prevision": rang["yield_t_ha"],
        "observe": rang["last_observed_t_ha"],
    } for rang in classement["ranking"]])
    ordre = list(donnees["culture"])

    axe_y = alt.Y("culture:N", title=None, sort=ordre)
    barres_ = alt.Chart(donnees).mark_bar(color="#2f855a", height=18).encode(
        x=alt.X("prevision:Q", title="t/ha"), y=axe_y,
        tooltip=["culture", "prevision", "observe"])
    repere = alt.Chart(donnees).mark_point(
        shape="circle", size=70, filled=True, color="#2d3748").encode(
        x=alt.X("observe:Q", title="t/ha"), y=axe_y)
    return (alt.layer(barres_, repere)
            .properties(height=max(150, 34 * len(ordre)))
            .configure(locale=LOCALE_FR))


def mode_national(series: dict) -> None:
    """Un pays, une culture : le rendement national de l'année à venir.

    **Aucune condition n'est demandée, et l'interface doit le dire.** Ce
    modèle ne lit ni la pluie ni la température : il prolonge l'historique
    national de la série choisie. Afficher les curseurs de parcelle ici
    laisserait croire qu'ils comptent.
    """
    cible, reference = series["target_year"], series["reference_year"]
    st.info(
        f"**Rendements nationaux, pas parcellaires.** Ce second modèle "
        f"prolonge l'historique FAO d'un pays et d'une culture. Il prédit "
        f"**{cible}**, l'année qui suit le dernier relevé disponible "
        f"({reference}) — pas la campagne en cours. Il ne demande aucune "
        f"condition de culture : elles ne déplaceraient la prédiction que "
        f"d'un centième de tonne.")

    cultures_par_pays = {serie["area"]: serie["items"]
                         for serie in series["areas"]}
    gauche, droite = st.columns(2)
    with gauche:
        pays = st.selectbox(f"Pays ({series['areas_count']} disponibles)",
                            list(cultures_par_pays),
                            index=list(cultures_par_pays).index("France")
                            if "France" in cultures_par_pays else 0,
                            help="Noms tels que publiés par la FAO.")
    disponibles = {culture["item"]: culture["item_fr"]
                   for culture in cultures_par_pays[pays]}
    with droite:
        culture = st.selectbox(
            f"Culture ({len(disponibles)} dans ce pays)", list(disponibles),
            format_func=lambda item: disponibles[item],
            help="Seules les cultures dont ce pays a un historique "
                 "exploitable sont proposées.")

    if not st.button("Estimer le rendement national", type="primary"):
        return

    with st.spinner("Interrogation de l'API…"):
        reponse = predire_national(pays, culture)
        classement = classer_national(pays)

    intervalle = reponse["prediction_interval_95"]
    gauche, droite = st.columns([1, 2])
    with gauche:
        dernier = reponse["history"][0]
        st.metric(
            f"{disponibles[culture]} — {pays}, {cible}",
            f"{nombre_fr(reponse['yield_t_ha'], 2)} t/ha",
            delta=f"{nombre_fr(reponse['yield_t_ha'] - dernier['yield_t_ha'], 2)}"
                  f" t/ha vs {dernier['year']}")
        st.caption(f"entre {nombre_fr(intervalle['lower_t_ha'], 1)} et "
                   f"{nombre_fr(intervalle['upper_t_ha'], 1)} t/ha")
    with droite:
        st.altair_chart(graphique_trajectoire(reponse), width="stretch")

    couverture = nombre_fr(intervalle["measured_coverage"] * 100, 1)
    st.caption(
        f"**La fourchette est empirique.** Sa couverture a été mesurée à "
        f"{couverture} % globalement sur des années que le modèle n'avait pas "
        "vues, mais elle n'est pas garantie pour une prédiction donnée : elle "
        "varie selon le niveau de rendement et selon la culture.")

    st.divider()
    st.subheader(f"Les cultures de ce pays, classées — {pays}")
    st.caption(classement["ranking_note"])
    if len(classement["ranking"]) > 1:
        st.altair_chart(graphique_classement(classement), width="stretch")
        st.caption(
            "Barre verte : le rendement estimé pour l'année à venir. Point "
            f"noir : le rendement relevé en {reference}.")
    st.warning(classement["economic_note"])

    afficher_qualite(reponse)


def mode_prediction(contexte: dict, contrat: dict) -> None:
    """Une culture, un rendement, une fourchette."""
    culture = st.selectbox("Culture", contrat["accepted_values"]["Crop"],
                           format_func=traduire)

    if not st.button("Estimer le rendement", type="primary"):
        return

    with st.spinner("Interrogation de l'API…"):
        reponse = predire(contexte, culture)

    intervalle = reponse["prediction_interval_95"]
    gauche, droite = st.columns([1, 2])
    with gauche:
        st.metric(f"Rendement attendu — {traduire(culture)}",
                  f"{nombre_fr(reponse['yield_t_ha'], 2)} t/ha")
        st.caption(f"entre {nombre_fr(intervalle['lower_t_ha'], 1)} et "
                   f"{nombre_fr(intervalle['upper_t_ha'], 1)} t/ha")
    with droite:
        st.altair_chart(graphique_intervalle(reponse), width="stretch")

    # Le nombre est converti seul. Appliquer `.replace(".", ",")` à la phrase
    # entière changeait aussi les points de ses trois autres phrases — les
    # littéraux adjacents se concatènent avant l'appel de méthode.
    couverture = nombre_fr(
        contrat["prediction_interval"]["measured_coverage"] * 100, 2)
    st.caption(
        "**La barre est le résultat, le losange n'en est que le centre.** "
        "Une parcelle sur vingt sort de cette plage. La couverture annoncée "
        "a été mesurée sur les données du prototype — qui sont simulées : "
        f"{couverture} % des parcelles de contrôle y tombaient. Rien "
        "n'établit que cette couverture tienne sur une exploitation réelle.")

    afficher_qualite(reponse)


#: Charges opérationnelles de départ, **sans source**. Ce sont des ordres de
#: grandeur hiérarchisés par intensité connue — coton et riz irrigué en tête,
#: céréales à paille en bas — destinés à éviter une grille vide, pas à chiffrer
#: une parcelle. Trois raisons de les différencier par culture plutôt que de
#: poser une valeur commune : les six rendements prédits sont identiques, donc
#: des charges égales feraient du classement des marges une recopie du
#: classement des prix ; `config.py` a mesuré que des charges européennes
#: combinées aux prix mondiaux FAO donnent des marges négatives sur trois
#: cultures sur quatre ; et une valeur unique se lirait comme une moyenne.
#: Repère d'interface, donc, et non donnée : c'est la seule raison pour
#: laquelle il est acceptable de l'écrire ici plutôt que de la demander à
#: l'API. Tout ce qui entoure le tableau le répète.
CHARGES_EXEMPLE = {
    "Barley": 450.0, "Wheat": 500.0, "Soybean": 550.0,
    "Maize": 800.0, "Rice": 950.0, "Cotton": 1200.0,
}


def tableau_budgets(classement: list[dict], prix: dict) -> pd.DataFrame:
    """Le tableau éditable : trois colonnes proposées, toutes à vérifier.

    Le prix vient de FAOSTAT, le rendement de la prédiction, les charges de
    `CHARGES_EXEMPLE` — qui ne sont la donnée de personne. Une culture absente
    de ce dictionnaire arrive sans charges : rien n'est inventé pour une
    culture que l'interface ne connaissait pas au moment de son écriture.
    """
    par_culture = {ligne["crop"]: ligne for ligne in prix["prices"]}
    return pd.DataFrame([
        {"culture": traduire(ligne["crop"]),
         "crop": ligne["crop"],
         "price_eur_t": float(par_culture[ligne["crop"]]["price_eur_t"]),
         "expected_yield_t_ha": float(ligne["yield_t_ha"]),
         "variable_costs_eur_ha": CHARGES_EXEMPLE.get(ligne["crop"])}
        for ligne in classement
    ])


def csv_resultats(reponse: dict, edite: pd.DataFrame, economique: dict | None,
                  prix: dict, contrat: dict) -> str:
    """L'écran en un fichier, avec de quoi savoir d'où vient chaque chiffre.

    Trois précautions qui comptent plus que le tableau lui-même : une culture
    non évaluée a une marge **vide**, jamais zéro ; chaque colonne proposée est
    doublée d'une colonne d'origine, qui distingue ce que l'application a
    suggéré de ce que l'utilisateur a saisi ; et l'en-tête porte la version du
    modèle servi, sans quoi le fichier ne serait rattachable à rien.

    Séparateur `;` et virgule décimale : c'est ce qu'attend un tableur
    configuré en français, et l'en-tête le dit pour qui le lira autrement.
    """
    predits = {ligne["crop"]: float(ligne["yield_t_ha"])
               for ligne in reponse["yield_ranking"]}
    faostat = {ligne["crop"]: float(ligne["price_eur_t"])
               for ligne in prix["prices"]}
    marges = {marge["crop"]: marge["gross_margin_eur_ha"]
              for marge in economique["economic_ranking"]["margins"]} \
        if economique else {}

    lignes = []
    for ligne in edite.to_dict("records"):
        culture = ligne["crop"]
        charges = ligne.get("variable_costs_eur_ha")
        vide = charges is None or pd.isna(charges)
        lignes.append({
            "culture": traduire(culture),
            "culture_source": culture,
            "rendement_predit_t_ha": predits.get(culture),
            "rendement_retenu_t_ha": ligne["expected_yield_t_ha"],
            "rendement_origine": _origine(
                ligne["expected_yield_t_ha"], predits.get(culture),
                "modèle"),
            "prix_eur_t": ligne["price_eur_t"],
            "prix_origine": _origine(ligne["price_eur_t"],
                                     faostat.get(culture), "FAOSTAT 2024"),
            "charges_eur_ha": None if vide else float(charges),
            "charges_origine": "non renseigné" if vide else _origine(
                float(charges), CHARGES_EXEMPLE.get(culture),
                "exemple de l'interface"),
            # Vide, et surtout pas zéro : une marge nulle se compare, une
            # marge absente non.
            "marge_brute_eur_ha": marges.get(culture),
            "evaluee": "oui" if culture in marges else "non",
        })

    servi, origine = contrat["served_model"], contrat["provenance"]
    entete = "\n".join(
        "# " + texte for texte in [
            "Agritech Answers — résultats exportés le "
            + datetime.now().isoformat(timespec="seconds"),
            f"Modèle servi : {servi['registry']} v{servi['version']}, "
            f"algorithme {origine['modele']}, run {origine['run_source']}, "
            f"chargé depuis {servi['source']}",
            "Rendements : prédiction d'un modèle entraîné sur des données "
            "SIMULÉES. Non transposable à une parcelle réelle.",
            "Prix : FAOSTAT 2024, médiane mondiale des pays déclarants — pas "
            "un prix français, pas un prix de contrat.",
            "Charges : aucune source dans ce projet. « exemple de "
            "l'interface » désigne un ordre de grandeur non validé.",
            "Marge brute = prix × rendement − charges opérationnelles, hors "
            "aides et coproduits. Ce n'est pas un bénéfice.",
            "Une culture non évaluée a une marge vide, et non nulle.",
            "Ce fichier documente une simulation. Ce n'est pas un conseil "
            "agronomique validé.",
            "Séparateur « ; », virgule décimale.",
        ]) + "\n"
    return entete + pd.DataFrame(lignes).to_csv(
        index=False, sep=";", decimal=",")


def _origine(retenu, propose, source: str) -> str:
    """« saisi » dès que la valeur s'écarte de ce que l'application proposait."""
    if propose is None:
        return "saisi"
    return source if float(retenu) == float(propose) else "saisi"


def fusionner_rendements(base: pd.DataFrame | None,
                         classement: list[dict], prix: dict) -> pd.DataFrame:
    """Rendements neufs, charges conservées.

    Changer un interrupteur — fertilisation, irrigation — puis relancer la
    comparaison doit mettre à jour la colonne prédite **sans** effacer les
    charges déjà saisies : c'est exactement l'effet qu'on cherche à mesurer.
    Le tableau n'est donc reconstruit que si les cultures ont changé.

    Une cellule que l'utilisateur a modifiée lui-même reste la sienne, y
    compris un rendement : `st.data_editor` réapplique ses saisies par-dessus
    ce qu'on lui passe. Corriger un rendement est un geste délibéré, et le
    défaire à la comparaison suivante serait plus déroutant que de le garder.
    """
    predits = {ligne["crop"]: float(ligne["yield_t_ha"])
               for ligne in classement}
    if base is None or set(base["crop"]) != set(predits):
        return tableau_budgets(classement, prix)
    return base.assign(expected_yield_t_ha=base["crop"].map(predits))


def charges_modifiees(tableau: pd.DataFrame) -> bool:
    """Vrai dès qu'une charge s'écarte de l'exemple fourni.

    Sert à savoir si le classement affiché repose encore sur des chiffres que
    personne n'a validés. Vider une case compte comme une intervention : c'est
    un retrait délibéré de la culture, pas un oubli.
    """
    for ligne in tableau.to_dict("records"):
        defaut = CHARGES_EXEMPLE.get(ligne["crop"])
        charges = ligne.get("variable_costs_eur_ha")
        if charges is None or pd.isna(charges):
            if defaut is not None:
                return True
            continue
        if defaut is None or float(charges) != defaut:
            return True
    return False


def mode_recommandation(contexte: dict, contrat: dict) -> None:
    """Deux temps : les rendements d'abord, les marges ensuite."""
    if st.button("Comparer les six cultures", type="primary"):
        with st.spinner("Interrogation de l'API…"):
            st.session_state["rendements"] = recommander(contexte)
            st.session_state["contexte"] = contexte
            # Les marges affichées portaient sur les rendements précédents.
            # Les laisser à l'écran les ferait passer pour un résultat à
            # jour ; les effacer renverrait l'utilisateur au bouton. On les
            # refait donc, plus bas, une fois le tableau de budgets connu.
            # Les charges, elles, survivent : elles sont dans l'état du
            # tableau, pas ici.
            st.session_state["refaire_les_marges"] = (
                "economique" in st.session_state)
            for cle in ("marges", "economique", "sur_exemples"):
                st.session_state.pop(cle, None)

    reponse = st.session_state.get("rendements")
    if reponse is None:
        st.info("Renseignez votre parcelle, puis lancez la comparaison.")
        return

    cultures = [ligne["crop"] for ligne in reponse["yield_ranking"]]
    ordre = sorted(cultures, key=traduire)
    rendements = pd.DataFrame([
        {"culture": traduire(ligne["crop"]), "crop": ligne["crop"],
         "rendement": ligne["yield_t_ha"]}
        for ligne in reponse["yield_ranking"]
    ]).set_index("crop").loc[ordre].reset_index()

    marges = st.session_state.get("marges")
    panneau_gauche, panneau_droit = st.columns(2)
    with panneau_gauche:
        barres(rendements, "rendement", "Rendement attendu", "t/ha",
               "#2b6cb0", ordre, pas=0.5, decimales=2)
    with panneau_droit:
        if marges is None:
            st.markdown("**Marge brute simulée (€/ha)**")
            st.info("Saisissez vos charges ci-dessous pour obtenir ce "
                    "second panneau.")
        else:
            barres(marges, "marge", "Marge brute simulée", "€/ha",
                   "#c05621", ordre, pas=200.0)

    # L'échelle ne se rétracte jamais d'elle-même. Passer à un scénario sans
    # rapport avec le précédent — d'un riz très margé à une céréale — doit
    # pouvoir la ramener au contexte du moment, sans recharger la page.
    if any(f"echelle_{champ}" in st.session_state
           for champ in ("rendement", "marge")):
        if st.button("Réinitialiser l'échelle des graphiques"):
            for champ in ("rendement", "marge"):
                st.session_state.pop(f"echelle_{champ}", None)
            st.rerun()

    if not reponse["ranking_discriminative"]:
        st.warning(reponse["yield_ranking_note"])

    st.dataframe(
        rendements[["culture", "rendement"]].rename(
            columns={"culture": "Culture", "rendement": "Rendement (t/ha)"}),
        hide_index=True, width="stretch")

    _comparer_les_marges(reponse, ordre)
    afficher_qualite(st.session_state.get("economique", reponse))


def _formule_marge() -> None:
    """La formule appliquée, et la raison pour laquelle c'est celle-là.

    Écrite ici plutôt que lue dans la réponse : l'API énonce la convention en
    prose dans ses `limitations`, elle ne renvoie pas de formule. Les deux
    textes disent la même chose et doivent le rester.
    """
    st.latex(
        r"\text{Marge brute (€/ha)} = "
        r"\underbrace{\text{Prix (€/t)} \times \text{Rendement (t/ha)}}"
        r"_{\text{produit brut (€/ha)}}"
        r" \;-\; \text{Charges opérationnelles (€/ha)}"
    )
    st.caption(
        "**Marge brute, et non bénéfice.** C'est l'indicateur que la FAO et "
        "le RICA européen retiennent pour comparer des cultures entre elles. "
        "Il ne soustrait que les charges opérationnelles — semences, engrais, "
        "traitements, irrigation, travaux par tiers. Les charges fixes — "
        "matériel, foncier, main-d'œuvre permanente — en sont exclues parce "
        "qu'elles sont engagées **avant** le choix de la culture : elles "
        "pèsent sur le résultat de l'exploitation, pas sur le départage des "
        "six options. Les aides et les coproduits ne sont pas comptés non "
        "plus. Le chiffre obtenu classe des cultures ; il ne dit pas ce que "
        "rapporte la parcelle.")


def _comparer_les_marges(reponse: dict, ordre: list[str]) -> None:
    """Le second temps : budgets saisis, puis classement économique."""
    st.divider()
    st.subheader("Comparer les marges")
    st.caption(
        "Le **prix** vient de FAOSTAT 2024 — médiane mondiale des pays "
        "déclarants. Le **rendement attendu** est la prédiction du modèle, "
        "entraîné sur des données simulées : ajustez-le à ce que vous "
        "attendez sur votre parcelle. Les **charges** n'ont aucune source "
        "dans ce projet. Les trois colonnes sont modifiables.")

    _formule_marge()

    st.warning(
        "**Les charges affichées sont des exemples, pas des références.** "
        "Aucun jeu de données du projet n'en contient : ce sont des ordres de "
        "grandeur, posés pour que le tableau ne s'ouvre pas vide. Remplacez-"
        "les par vos chiffres — le classement ne vaut qu'à ce prix. Vider une "
        "case retire la culture de la comparaison.")

    base = fusionner_rendements(st.session_state.get("budgets_base"),
                                reponse["yield_ranking"], _prix())
    st.session_state["budgets_base"] = base

    edite = st.data_editor(
        base.set_index("crop").loc[ordre].reset_index(),
        key="editeur_budgets", hide_index=True, width="stretch",
        column_order=("culture", "price_eur_t", "expected_yield_t_ha",
                      "variable_costs_eur_ha"),
        column_config={
            "culture": st.column_config.TextColumn("Culture", disabled=True),
            "price_eur_t": st.column_config.NumberColumn(
                "Prix (€/t) · FAOSTAT", min_value=1.0, step=1.0,
                format="%.0f"),
            "expected_yield_t_ha": st.column_config.NumberColumn(
                "Rendement attendu (t/ha) · modèle", min_value=0.1,
                max_value=50.0, step=0.1, format="%.2f"),
            "variable_costs_eur_ha": st.column_config.NumberColumn(
                "Charges (€/ha) · exemple à remplacer", min_value=0.0,
                step=10.0, format="%.0f",
                help="Valeur d'exemple sans source. Saisissez la vôtre, ou "
                     "videz la case pour écarter la culture."),
        })

    budgets = construire_budgets(edite)
    # Relu avant la sortie anticipée : un drapeau laissé en place se
    # déclencherait plus tard, à un moment que l'utilisateur n'a pas demandé.
    refaire = st.session_state.pop("refaire_les_marges", False)
    if len(budgets) < 2:
        st.info(f"{len(budgets)} culture(s) budgétée(s). L'API compare à "
                "partir de deux budgets complets.")
        return

    # Les rendements viennent de changer et des marges étaient affichées :
    # on les refait sur les charges déjà saisies. Changer un interrupteur
    # doit déplacer les barres, pas renvoyer au bouton.
    if refaire:
        with st.spinner("Mise à jour des marges…"):
            _appeler_les_marges(budgets, edite)
        st.rerun()

    if st.button(f"Calculer les marges ({len(budgets)} cultures)"):
        with st.spinner("Interrogation de l'API…"):
            _appeler_les_marges(budgets, edite)
        st.rerun()

    _afficher_classement_economique()
    st.download_button(
        "Télécharger les résultats (CSV)",
        data=csv_resultats(reponse, edite,
                           st.session_state.get("economique"),
                           _prix(), _contrat()),
        file_name="agritech-resultats.csv", mime="text/csv")


def tableau_marges(classement: dict) -> pd.DataFrame:
    """Le classement économique, prêt à afficher.

    La colonne `Source` est conservée telle que l'API la rend — `user` pour
    toute ligne, y compris celles dont le prix et le rendement venaient d'une
    proposition. C'est le contrat : l'appelant a validé ces chiffres en
    lançant le calcul. L'en-tête du tableau de saisie, lui, dit d'où chaque
    colonne était partie.
    """
    return pd.DataFrame([
        {"Culture": traduire(marge["crop"]),
         "Marge brute (€/ha)": round(marge["gross_margin_eur_ha"]),
         "Prix (€/t)": marge["price_eur_t"],
         "Rendement (t/ha)": marge["expected_yield_t_ha"],
         "Charges (€/ha)": marge["variable_costs_eur_ha"],
         "Source": marge["source"]}
        for marge in classement["margins"]
    ])


def message_sensibilite(sensibilite: dict) -> str | None:
    """L'alerte quand le classement ne survit pas au scénario de prix.

    Rend `None` quand l'ordre tient : il n'y a alors rien à signaler, et une
    alerte systématique finirait par ne plus être lue.
    """
    if sensibilite["ranking_stable"]:
        return None
    paires = ", ".join(f"{traduire(a)} et {traduire(b)}"
                       for a, b in sensibilite["indistinguishable_pairs"])
    # Espace avant le pourcentage : convention typographique française, que
    # le formatage `:.0%` de Python ne produit pas.
    variation = sensibilite["tested_variation"] * 100
    return ("Le classement ne résiste pas à une variation des prix de "
            f"±{variation:.0f} % : {paires} ne sont pas départagées.")


def _appeler_les_marges(budgets: dict, edite: pd.DataFrame) -> None:
    """Demande le classement économique et range ce qu'il faut afficher.

    `sur_exemples` est retenu **au moment de l'appel** : le tableau peut être
    réédité ensuite sans que le classement affiché, lui, soit recalculé.
    """
    economique = recommander(st.session_state["contexte"], budgets)
    st.session_state["economique"] = economique
    st.session_state["sur_exemples"] = not charges_modifiees(edite)
    st.session_state["marges"] = pd.DataFrame([
        {"culture": traduire(marge["crop"]), "crop": marge["crop"],
         "marge": marge["gross_margin_eur_ha"]}
        for marge in economique["economic_ranking"]["margins"]
    ])


def _afficher_classement_economique() -> None:
    """L'avis de l'API, rendu tel quel — ni durci, ni adouci."""
    economique = st.session_state.get("economique")
    if economique is None:
        return

    classement = economique["economic_ranking"]
    if st.session_state.get("sur_exemples"):
        st.warning(
            "Classement obtenu **sur les charges d'exemple** : aucune n'a été "
            "modifiée. Il illustre le calcul, il ne conseille rien.")
    st.dataframe(tableau_marges(classement), hide_index=True, width="stretch")

    avis = economique["recommendation"]
    (st.success if avis["status"] == "options" else st.info)(avis["rationale"])

    alerte = message_sensibilite(classement["price_sensitivity"])
    if alerte:
        st.warning(alerte)
    st.caption(classement["note"])


def sources_et_limites() -> None:
    """Ce que l'application ne dit pas, à l'écart du formulaire."""
    with st.expander("Sources et limites"):
        st.markdown(
            "**Pourquoi ces variables, et pas d'autres.** L'application "
            "demande la pluie, la température, la fertilisation et "
            "l'irrigation. Les pesticides n'y figurent pas : la source "
            "disponible donne un tonnage national annuel, qui ne peut être "
            "rattaché à une parcelle.\n\n"
            "**Les champs facultatifs ne déplacent presque rien.** Région, "
            "sol et météo sont bien transmis au modèle, mais changer l'un "
            "d'eux fait varier la prédiction de quelques kilogrammes par "
            "hectare, contre une fourchette large de près de deux tonnes. Ce "
            "n'est pas un défaut du modèle : les données d'entraînement sont "
            "simulées à partir de la pluie, de la température, de la "
            "fertilisation et de l'irrigation seules. Ces trois champs "
            "resteraient donc sans effet tant que l'entraînement n'aura pas "
            "eu lieu sur des relevés réels.\n\n"
            "**Les données d'entraînement sont simulées.** Les chiffres "
            "affichés démontrent la chaîne complète, de la donnée à "
            "l'application ; ils ne sont pas transposables à une parcelle "
            "réelle.\n\n"
            "**Les prix sont des médianes mondiales**, pas des prix "
            "français, et les charges n'ont aucune source dans ce projet.\n\n"
            "**Deux modèles, deux questions.** Les modes « Prédiction » et "
            "« Recommandation » interrogent un modèle de parcelle entraîné "
            "sur des données simulées. Le mode « Rendements nationaux » "
            "interroge un second modèle, entraîné sur les relevés réels de "
            "la FAO, mais à l'échelle d'un pays et d'une année : il prolonge "
            "un historique national et ne décrit aucune parcelle. Les deux "
            "ne se remplacent pas, et leurs erreurs ne se comparent pas.\n\n"
            "**Le classement national n'est pas un classement de "
            "rentabilité.** Il manque six prix sur dix pour ces cultures, et "
            "sur les quatre tarifées l'ordre change entre les tonnes et la "
            "valeur. Qu'une pomme de terre rende dix fois plus qu'un soja ne "
            "dit rien de ce qu'elle rapporte.\n\n"
            "**Cette application ne calcule rien.** Toute valeur affichée "
            "vient de l'API.")


def informations_techniques(contrat: dict, series: dict | None = None) -> None:
    """Qui a répondu — en trois blocs qu'il serait faux de confondre.

    Le modèle **servi** est celui que l'API a chargé à son démarrage. Sa
    **provenance** dit d'où vient cet artefact. Sa **promotion** est l'état du
    registre MLflow au moment de l'export — un enregistrement, pas un lien
    vivant. Déplacer l'alias `champion` dans MLflow ne change rien à un
    service déjà lancé : l'image embarque un `.joblib`, et même en mode
    registre l'artefact est lu une fois, au démarrage.
    """
    with st.expander("Informations techniques du modèle"):
        servi, origine = contrat["served_model"], contrat["provenance"]

        st.markdown("**Ce que l'API sert en ce moment**")
        st.dataframe(_lignes({
            "Registre": servi["registry"],
            "Version servie": servi["version"],
            "Chargé depuis": servi["source"],
            "Adresse du service": URL_API,
        }), hide_index=True, width="stretch")

        st.markdown("**D'où vient cet artefact**")
        st.dataframe(_lignes({
            "Algorithme": origine["modele"],
            "Run MLflow": origine["run_source"],
            "Version des données": origine["data_version"],
            "Version du découpage": origine["split_version"],
            "Empreinte SHA-256": origine["artefact_sha256"],
            "Schéma d'entrée": origine["schema_entree"],
            "Hypothèses économiques": origine["hypotheses_economiques"],
            "Exporté le": origine["exporte_le"],
            "scikit-learn / pandas / numpy": " · ".join([
                origine["sklearn_version"], origine["pandas_version"],
                origine["numpy_version"]]),
        }), hide_index=True, width="stretch")

        st.markdown("**Ce que le registre enregistrait à l'export**")
        st.dataframe(_lignes({
            "Registre d'entraînement": origine["registre_source"],
            "Version source": origine["version_source"],
            "Alias": origine["alias"],
            "Alias précédent": origine["alias_precedent"] or "aucun",
            "Promu le": origine["promu_le"],
        }), hide_index=True, width="stretch")

        if series:
            servi_national = series["served_model"]
            st.markdown("**Le second modèle, national**")
            st.dataframe(_lignes({
                "Registre": servi_national["registry"],
                "Version servie": servi_national["version"],
                "Chargé depuis": servi_national["source"],
                "Dernière année observée": series["reference_year"],
                "Année prédite": series["target_year"],
                "Pays servis": series["areas_count"],
                "Couples (pays, culture)": series["series_count"],
            }), hide_index=True, width="stretch")
            st.caption(
                "Ce modèle a son propre registre et son propre alias : les "
                "deux artefacts n'attendent pas les mêmes entrées — huit "
                "saisies de parcelle d'un côté, un pays et une culture de "
                "l'autre — et leurs versions ne sont pas interchangeables.")

        if st.button("Relire les informations de l'API"):
            _contrat.clear()
            _prix.clear()
            _series.clear()
            st.rerun()
        st.caption(
            "Ce bouton relit ce que l'API **déclare** — utile après un "
            "redéploiement, puisque l'application garde ces informations cinq "
            "minutes. Il ne recharge aucun modèle et ne promeut rien : "
            "l'artefact servi est lu au démarrage du service. Déplacer "
            "l'alias dans MLflow demande de redémarrer l'API, et de "
            "reconstruire l'image quand le modèle y est embarqué — ce qui est "
            f"le cas ici (`{servi['source']}`).")


def _lignes(valeurs: dict) -> pd.DataFrame:
    """Un dictionnaire rendu en deux colonnes lisibles."""
    return pd.DataFrame({"Information": list(valeurs),
                         "Valeur": [str(valeur) for valeur in valeurs.values()]})


def main() -> None:
    """Le point d'entrée : barre latérale, mode, formulaire, résultat."""
    st.set_page_config(page_title="Agritech Answers", page_icon="🌾",
                       layout="wide")
    st.title("Agritech Answers — estimation de rendement")

    # Le premier appel est celui qui réveille un service endormi : il peut
    # durer, et un écran muet pendant ce temps se lit comme une panne.
    try:
        with st.spinner("Connexion au service…"):
            contrat = _contrat()
    except ErreurAPI as erreur:
        st.error(str(erreur))
        if st.button("Réessayer"):
            st.rerun()
        st.stop()

    try:
        series = _series()
    except ErreurAPI as erreur:
        # Le catalogue national manque : les deux modes de parcelle restent
        # utilisables, et c'est tout l'intérêt de ne pas s'arrêter ici.
        series = None
        st.warning(f"Service national indisponible. {erreur}")

    with st.sidebar:
        st.header("Mode")
        modes = ["Prédiction", "Recommandation"]
        legendes = ["Le rendement d'une parcelle",
                    "Comparer les six cultures"]
        if series:
            modes.append("Rendements nationaux")
            legendes.append(f"Un pays, une culture, {series['target_year']}")
        mode = st.radio("Que voulez-vous faire ?", modes, captions=legendes,
                        label_visibility="collapsed")
        st.divider()
        servi = contrat["served_model"]
        st.caption(f"API : {URL_API}")
        st.caption(f"Parcelle : {servi['registry']} v{servi['version']}")
        if series:
            national = series["served_model"]
            st.caption(f"National : {national['registry']} "
                       f"v{national['version']}")

    try:
        if mode == "Rendements nationaux":
            # Pas de formulaire de parcelle ici : ce modèle n'en lit aucun
            # champ, et les afficher laisserait croire qu'ils comptent.
            mode_national(series)
        else:
            st.subheader("Votre parcelle")
            contexte = saisir_contexte(contrat)
            st.divider()
            if mode == "Prédiction":
                mode_prediction(contexte, contrat)
            else:
                mode_recommandation(contexte, contrat)
    except ErreurAPI as erreur:
        st.error(str(erreur))

    st.divider()
    sources_et_limites()
    informations_techniques(contrat, series)


if __name__ == "__main__":
    main()

"""Produit les figures du rapport de l'étape 3, à partir des vraies sources.

**Pourquoi un script plutôt que des images tracées une fois.** Une figure
recopiée à la main dérive sans prévenir : le modèle change, le prix change, et
l'image continue d'affirmer l'ancien chiffre. Ici tout est lu à l'exécution —
le modèle depuis l'artefact de service, la calibration et les prix depuis
`config.py`. Relancer le script après un réexport remet les figures d'accord
avec ce que l'API sert réellement.

**Les seules valeurs fixes sont des hypothèses de démonstration**, rassemblées
dans `BUDGETS_DEMONSTRATION` et annoncées comme telles sur la figure. Les
charges opérationnelles n'existent dans aucune source du projet : les inventer
en silence serait la faute que tout le rapport s'emploie à ne pas commettre.

**Le script ne modifie rien.** Il charge l'artefact embarqué en lecture seule,
n'ouvre pas MLflow, ne touche ni au registre ni aux données préparées. Sa
seule écriture est celle des quatre PNG.

    uv run python scripts/build_report_figures.py

Public visé : le rapport est lu par des agriculteurs. Chaque figure doit se
comprendre sans connaissance en statistique, **sans pour autant affirmer plus
que ce qui a été mesuré** — l'intervalle est calibré sur des données simulées,
et la figure 2 le dit en toutes lettres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(RACINE))

from agritech.config import (  # noqa: E402
    BLUE,
    DEMI_INTERVALLE_PREDICTION_95_T_HA,
    GREEN,
    MODALITES_CATEGORIELLES,
    ORANGE,
    PRIX_REFERENCE_FAOSTAT,
    PRIX_REFERENCE_SOURCE,
    SEUIL_ECART_RENDEMENT_PERTINENT_T_HA,
)
from api.schemas import EXEMPLE_RECOMMANDATION  # noqa: E402
from api.service import charger_modele  # noqa: E402

DOSSIER = RACINE / "reports" / "figures"

#: Gris de texte secondaire : les réserves doivent rester lisibles sans voler
#: la vedette à la donnée.
GRIS = "#4a5568"
GRIS_CLAIR = "#a0aec0"

CULTURES = MODALITES_CATEGORIELLES["Crop"]

#: Noms français : le lecteur n'a pas à connaître les libellés du jeu de
#: données.
EN_FRANCAIS = {"Barley": "Orge", "Cotton": "Coton", "Maize": "Maïs",
               "Rice": "Riz", "Soybean": "Soja", "Wheat": "Blé"}

#: Parcelle de démonstration, au milieu du domaine d'entraînement. Les
#: rendements affichés en sont **calculés**, jamais écrits.
PARCELLE = {"rainfall_mm": 650.0, "temperature_celsius": 22.0,
            "fertilizer_used": True, "irrigation_used": True}

#: Les six budgets de l'exemple documenté du contrat (`api/schemas.py`),
#: repris tels quels plutôt que réinventés ici : le dépôt n'a ainsi qu'un
#: seul jeu de chiffres de démonstration. Les prix sont ceux de FAOSTAT ;
#: les rendements attendus et les charges sont **fournis par l'agriculteur**
#: dans le contrat de l'API, et n'ont donc aucune source — la figure le dit.
BUDGETS_DEMONSTRATION = EXEMPLE_RECOMMANDATION["budgets"]


def fr(valeur: float, decimales: int = 0) -> str:
    """Nombre à la française : virgule décimale, espace pour les milliers.

    matplotlib écrit « 6.39 » et « 1,303 ». Dans un rapport destiné à des
    agriculteurs français, les deux se lisent de travers.
    """
    return (f"{valeur:,.{decimales}f}"
            .replace(",", " ").replace(".", ","))


def format_axe(ax, axe: str = "x", decimales: int = 0) -> None:
    """Applique la notation française aux graduations."""
    getattr(ax, f"{axe}axis").set_major_formatter(
        FuncFormatter(lambda valeur, _: fr(valeur, decimales)))


def enregistrer(figure, nom: str) -> None:
    """Même réglage que les figures des étapes 1 et 2, pour l'homogénéité."""
    chemin = DOSSIER / nom
    figure.savefig(chemin, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"  écrit  reports/figures/{nom} — "
          f"{chemin.stat().st_size / 1024:.0f} Ko")


def depouiller(ax, grille: str | None = "x") -> None:
    """Axes discrets : la donnée doit porter le regard, pas le cadre."""
    # Le thème seaborn allume les deux grilles ; celle qui suit les catégories
    # n'apporte rien et barre les annotations.
    ax.grid(False)
    if grille:
        ax.grid(axis=grille, color=GRIS_CLAIR, alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(colors=GRIS, labelsize=10)
    for cote in ("top", "right", "left"):
        ax.spines[cote].set_visible(False)
    ax.spines["bottom"].set_color(GRIS_CLAIR)


def rendements(modele, cultures=CULTURES) -> dict[str, float]:
    """Le rendement prédit pour chaque culture, sur la parcelle de démo."""
    import pandas as pd

    from api.schemas import PredictionRequete

    lignes = [PredictionRequete(**PARCELLE, crop=c).vers_variables()
              for c in cultures]
    valeurs = modele.pipeline.predict(pd.DataFrame(lignes))
    return dict(zip(cultures, (float(v) for v in valeurs)))


# --------------------------------------------------------------------------- #
# Figure 1 — le parcours d'une question
# --------------------------------------------------------------------------- #

#: Géométrie du schéma, en fraction de la figure. Les hauteurs de bloc sont
#: **calculées** depuis le nombre de lignes : les fixer à la main faisait
#: déborder le texte dès qu'une ligne s'ajoutait.
LIGNE = 0.060
ENTETE = 0.095
MARGE_BASSE = 0.040


def hauteur(lignes: list[str]) -> float:
    """La hauteur qu'il faut à un bloc pour contenir son texte."""
    return ENTETE + len(lignes) * LIGNE + MARGE_BASSE


def _boite(ax, x, y, largeur, titre, lignes, couleur, fond="#ffffff") -> float:
    """Trace un bloc et renvoie sa hauteur, pour empiler le suivant."""
    haut = hauteur(lignes)
    ax.add_patch(FancyBboxPatch(
        (x, y), largeur, haut, boxstyle="round,pad=0.010",
        linewidth=1.6, edgecolor=couleur, facecolor=fond, zorder=2))
    ax.text(x + largeur / 2, y + haut - 0.032, titre, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=couleur, zorder=3)
    for index, ligne in enumerate(lignes):
        ax.text(x + 0.020, y + haut - ENTETE - index * LIGNE, ligne,
                ha="left", va="top", fontsize=9.4, color=GRIS, zorder=3)
    return haut


def figure_parcours() -> None:
    """Les deux parcours de l'API, de la saisie à la réponse.

    Premier point à faire passer : **cinq variables ne sont jamais saisies**.
    L'application les reconstruit, et c'est ce qui rend le modèle utilisable
    par quelqu'un qui ne connaît que sa parcelle.

    Second point, invisible en texte : les deux routes ne demandent pas la
    même chose. `/predict` veut une culture ; `/recommend` les essaie toutes
    les six, et n'a donc pas à la demander.
    """
    figure, ax = plt.subplots(figsize=(13.5, 6.0))
    # L'axe occupe toute la figure : les marges par défaut laissaient une
    # bande blanche entre le dernier bloc et la note de pied.
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    for x, titre in ((0.02, "1 · Ce que vous saisissez"),
                     (0.385, "2 · Ce que l'application reconstruit"),
                     (0.705, "3 · Ce que vous recevez")):
        ax.text(x, 1.0, titre, fontsize=11.5, fontweight="bold",
                color="#1a202c", ha="left", va="top")

    # --- Colonne 1 : les saisies, empilées du haut vers le bas ------------ #
    y, gap = 0.945, 0.028
    for titre, lignes, couleur, fond in (
        ("Votre parcelle — 4 champs obligatoires",
         ["Pluie de la saison (mm)",
          "Température moyenne (°C)",
          "Fertilisation : oui / non",
          "Irrigation : oui / non"], BLUE, "#f7fafc"),
        ("3 champs facultatifs",
         ["Région · Type de sol · Météo"], GRIS_CLAIR, "#ffffff"),
        ("Puis, selon votre question",
         ["/predict : la culture visée",
          "/recommend : vos budgets"], ORANGE, "#ffffff"),
    ):
        y -= hauteur(lignes)
        _boite(ax, 0.02, y, 0.30, titre, lignes, couleur, fond)
        y -= gap

    # --- Colonne 2 : la dérivation ---------------------------------------- #
    lignes_derivation = ["Jamais saisies — toujours déduites :",
                         "· pluie faible sans irrigation",
                         "· 4 indicateurs de référence",
                         "   propres à chaque culture"]
    y_derivation = 0.48 - hauteur(lignes_derivation) / 2
    _boite(ax, 0.385, y_derivation, 0.265, "5 variables calculées",
           lignes_derivation, GREEN, "#f0fff4")
    ax.text(0.385, y_derivation - 0.045,
            "Les exposer permettrait de décrire\nune parcelle impossible.",
            fontsize=9, color=GRIS, style="italic", va="top")

    # --- Colonne 3 : les réponses ----------------------------------------- #
    reponses = (
        ("/predict", ["1 rendement attendu",
                      "+ la fourchette qui l'encadre"], BLUE, "#f7fafc"),
        ("/recommend", ["Les 6 cultures classées par rendement",
                        "+ la comparaison des marges,",
                        "à partir de 2 budgets complets"],
         ORANGE, "#fffaf0"),
    )
    total = sum(hauteur(lignes) for _, lignes, _, _ in reponses) + 0.06
    y = 0.48 + total / 2
    for titre, lignes, couleur, fond in reponses:
        y -= hauteur(lignes)
        _boite(ax, 0.705, y, 0.285, titre, lignes, couleur, fond)
        y -= 0.06

    _fleche = lambda d, a: ax.add_patch(FancyArrowPatch(  # noqa: E731
        d, a, arrowstyle="-|>", mutation_scale=16, linewidth=1.4,
        color=GRIS_CLAIR, zorder=1))
    _fleche((0.330, 0.50), (0.378, 0.50))
    _fleche((0.658, 0.50), (0.698, 0.66))
    _fleche((0.658, 0.50), (0.698, 0.35))

    # Hors du cadre : dans l'axe, cette note recouvrait le dernier bloc.
    figure.text(0.02, -0.01,
                "Le modèle est le même pour les deux routes. Ce qui change, "
                "c'est la question posée — et, pour /recommend, l'ajout d'un "
                "calcul économique qui ne vient pas du modèle.",
                fontsize=9, color=GRIS, style="italic", va="top")

    enregistrer(figure, "api_parcours_requete.png")


# --------------------------------------------------------------------------- #
# Figure 2 — ce que vaut une prédiction
# --------------------------------------------------------------------------- #


def figure_intervalle(modele) -> None:
    """Un rendement n'est pas une promesse : c'est un centre et une plage.

    **La formulation est le travail principal de cette figure.** Écrire
    « 19 fois sur 20, votre rendement tombera dans cette plage » serait faux :
    la couverture a été mesurée sur des données simulées, et rien ne garantit
    qu'elle vaille sur une parcelle réelle. L'annotation dit donc ce qui a été
    mesuré et où — en français lisible, pas en vocabulaire de statisticien.
    """
    valeur = rendements(modele, ["Wheat"])["Wheat"]
    demi = DEMI_INTERVALLE_PREDICTION_95_T_HA
    bas, haut = max(valeur - demi, 0.0), valeur + demi

    figure, ax = plt.subplots(figsize=(10.5, 2.9))
    ax.grid(False)

    ax.hlines(0, bas, haut, color=BLUE, linewidth=13, alpha=0.22, zorder=1)
    for borne in (bas, haut):
        ax.vlines(borne, -0.055, 0.055, color=BLUE, linewidth=2.4, zorder=2)
        ax.text(borne, -0.10, fr(borne, 1), ha="center", va="top",
                fontsize=11.5, color=BLUE, fontweight="bold")
    ax.plot([valeur], [0], "o", markersize=13, color=ORANGE,
            markeredgecolor="white", markeredgewidth=2, zorder=3)

    ax.text(valeur, 0.13, f"{fr(valeur, 1)} t/ha", ha="center", va="bottom",
            fontsize=15, fontweight="bold", color=ORANGE)
    ax.text(valeur, 0.09, "rendement le plus probable", ha="center",
            va="bottom", fontsize=9.5, color=GRIS)
    ax.text((bas + haut) / 2, -0.23, "une parcelle sur vingt sort de la plage",
            ha="center", va="top", fontsize=10.5, color=BLUE)

    ax.set_xlim(bas - 0.75, haut + 0.75)
    ax.set_ylim(-0.38, 0.36)
    ax.set_yticks([])
    ax.set_xlabel("rendement en tonnes par hectare", fontsize=10, color=GRIS)
    depouiller(ax, grille=None)
    format_axe(ax, "x", 1)

    ax.set_title("Une prédiction est un centre et une plage, jamais un "
                 "chiffre seul", fontsize=13, fontweight="bold",
                 color="#1a202c", pad=14, loc="left")

    figure.text(
        0.5, -0.14,
        "Mesuré sur les données du prototype : 94,92 % des parcelles d'un jeu "
        "de contrôle — écarté de la calibration — tombent dans leur plage.\n"
        "Ces données sont simulées. Cette fiabilité n'a pas été vérifiée sur "
        "des parcelles réelles, et l'intervalle devrait y être recalibré.",
        ha="center", va="top", fontsize=9, color=GRIS, style="italic")

    enregistrer(figure, "api_intervalle_prediction.png")


# --------------------------------------------------------------------------- #
# Figure 3 — ce que /recommend départage vraiment
# --------------------------------------------------------------------------- #


def figure_rendement_vs_marge(modele) -> None:
    """Le résultat métier du projet, en une image.

    Deux panneaux plutôt qu'un axe double : les unités diffèrent, et
    superposer des tonnes et des euros laisserait croire qu'on peut les
    comparer directement.

    **Les deux axes partent de zéro.** Zoomer à gauche ferait apparaître des
    écarts spectaculaires là où il n'y en a pas — le piège que ce projet
    documente depuis l'étape 2.

    Les rendements du panneau droit ne sont pas ceux du panneau gauche, et
    c'est le contrat de l'API qui le veut : une tonne de coton et une tonne
    de blé ne se vendent pas au même prix ni au même volume, donc
    l'agriculteur déclare son propre rendement commercialisable. La légende
    l'explique au lecteur.
    """
    predits = rendements(modele)
    # Même formule que `_marge()` dans l'API : prix × rendement − charges.
    marges = {
        culture: (budget["price_eur_t"] * budget["expected_yield_t_ha"]
                  - budget["variable_costs_eur_ha"])
        for culture, budget in BUDGETS_DEMONSTRATION.items()
    }
    etendue = max(predits.values()) - min(predits.values())

    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # --- Les deux panneaux partagent le même ordre, alphabétique ---------- #
    # Trier chaque panneau par sa propre grandeur empêchait la lecture
    # horizontale : le maïs était cinquième à gauche et premier à droite. Un
    # ordre figé rend la comparaison immédiate ; le classement, lui, reste
    # dans le texte et dans la réponse de l'API.
    positions = list(range(len(CULTURES)))

    gauche.barh(positions, [predits[c] for c in CULTURES], color=BLUE,
                alpha=0.85, height=0.58)
    gauche.set_xlim(0, max(predits.values()) * 1.30)
    gauche.set_xlabel("rendement attendu (t/ha)", fontsize=10, color=GRIS)
    gauche.set_title("Le modèle ne départage pas les cultures", fontsize=12.5,
                     fontweight="bold", color="#1a202c", loc="left", pad=12)
    for index, culture in enumerate(CULTURES):
        gauche.text(predits[culture] + 0.1, index, fr(predits[culture], 2),
                    va="center", fontsize=9.5, color=GRIS)

    # --- Panneau droit : mêmes emplacements, même ordre ------------------ #
    maxi = max(marges.values())
    droite.barh(positions, [marges[c] for c in CULTURES], color=ORANGE,
                alpha=0.9, height=0.58)
    droite.set_xlim(0, maxi * 1.32)
    droite.set_xlabel("marge brute simulée (€/ha)", fontsize=10, color=GRIS)
    droite.set_title("Ce sont vos chiffres économiques qui tranchent",
                     fontsize=12.5, fontweight="bold", color="#1a202c",
                     loc="left", pad=12)
    for index, culture in enumerate(CULTURES):
        droite.text(marges[culture] + maxi * 0.015, index,
                    f"{fr(marges[culture])} €", va="center", fontsize=9.5,
                    color=GRIS)

    for ax in (gauche, droite):
        ax.set_yticks(positions)
        ax.set_yticklabels([EN_FRANCAIS[c] for c in CULTURES], color=GRIS)
        ax.set_ylim(len(CULTURES) - 0.45, -0.55)

    for ax in (gauche, droite):
        depouiller(ax)
    format_axe(gauche, "x", 0)
    format_axe(droite, "x", 0)

    figure.text(
        0.075, -0.02,
        f"Écart de la première à la dernière culture : "
        f"{fr(etendue * 1000)} kg/ha, contre un seuil de "
        f"{fr(SEUIL_ECART_RENDEMENT_PERTINENT_T_HA * 1000)} kg/ha pour qu'un "
        "écart compte.\nLa réponse de l'API porte donc « classement non "
        "discriminant ».",
        ha="left", va="top", fontsize=9, color=GRIS, style="italic")

    figure.text(
        0.5, -0.17,
        "À gauche, le rendement que le modèle prédit. À droite, le rendement "
        "commercialisable que l'agriculteur déclare lui-même, culture par "
        "culture : une tonne de coton ne se vend ni au prix ni au volume "
        "d'une tonne de blé.\n"
        "Prix issus de FAOSTAT. Rendements attendus et charges : hypothèses "
        "de démonstration — aucune source du projet ne donne de charges. La "
        "marge la plus élevée n'est pas un avis agronomique : aucune "
        "contrainte de sol, de rotation ou de débouché n'est évaluée.",
        ha="center", va="top", fontsize=9, color=GRIS, style="italic")

    enregistrer(figure, "api_rendement_vs_marge.png")


# --------------------------------------------------------------------------- #
# Figure 4 — les prix de référence et leur dispersion
# --------------------------------------------------------------------------- #


def figure_prix() -> None:
    """Médiane et écart interquartile, par culture.

    **La barre n'est pas une marge d'erreur.** Elle montre l'étalement des
    prix *entre pays déclarants* — un prix à la production n'a pas la même
    valeur au Canada et au Bangladesh. Ce n'est ni l'incertitude statistique
    de la médiane, ni la variation d'un prix local dans le temps. La figure le
    dit, parce que la confusion est la lecture naturelle d'une barre.

    L'ordre reste alphabétique, comme dans l'API : trier par prix suggérerait
    une comparabilité entre cultures qui n'existe pas.
    """
    cultures = list(PRIX_REFERENCE_FAOSTAT)
    figure, ax = plt.subplots(figsize=(10.5, 4.4))

    for index, culture in enumerate(cultures):
        valeurs = PRIX_REFERENCE_FAOSTAT[culture]
        ax.hlines(index, valeurs["q1_eur_t"], valeurs["q3_eur_t"], color=BLUE,
                  linewidth=5, alpha=0.3, zorder=1)
        ax.plot([valeurs["prix_eur_t"]], [index], "o", markersize=11,
                color=BLUE, markeredgecolor="white", markeredgewidth=1.8,
                zorder=3)
        ax.text(valeurs["prix_eur_t"], index - 0.28,
                f"{fr(valeurs['prix_eur_t'])} €", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=BLUE)
        ax.text(1.015, index, f"{valeurs['pays']} pays",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9.5,
                color=GRIS)

    ax.set_yticks(range(len(cultures)))
    ax.set_yticklabels([EN_FRANCAIS[c] for c in cultures], fontsize=11)
    ax.set_ylim(len(cultures) - 0.4, -0.6)
    ax.set_xlabel("prix payé au producteur, en euros par tonne", fontsize=10,
                  color=GRIS)
    ax.set_xlim(0, max(v["q3_eur_t"] for v in PRIX_REFERENCE_FAOSTAT.values())
                * 1.06)
    depouiller(ax)
    format_axe(ax, "x", 0)

    ax.set_title("Un prix de référence est une fourchette, pas un tarif",
                 fontsize=13, fontweight="bold", color="#1a202c", pad=14,
                 loc="left")

    source = PRIX_REFERENCE_SOURCE
    figure.text(
        0.5, -0.09,
        "Point : prix médian. Barre : moitié centrale des pays déclarants, du "
        "1er au 3e quartile — une dispersion géographique, ni une marge "
        "d'erreur, ni une variation dans le temps.\n"
        f"{source['source']} {source['annee']}, {source['methode']}. "
        f"Conversion {source['conversion']}. Un prix à 11 pays déclarants "
        "couvre moins de terrain qu'un prix à 75 ; il n'est pas pour autant "
        "moins juste.",
        ha="center", va="top", fontsize=9, color=GRIS, style="italic")

    enregistrer(figure, "prix_reference_faostat.png")


# --------------------------------------------------------------------------- #
# Figure 5 — l'architecture de l'étape 4
# --------------------------------------------------------------------------- #


def figure_architecture() -> None:
    """Qui fait quoi, de l'agriculteur au modèle.

    Deux choses que le schéma doit faire passer, et qui se perdent en prose :

    **La frontière HTTP.** L'interface ne contient aucune logique de modèle.
    Elle pose des questions et affiche des réponses ; tout le reste est
    derrière une adresse. C'est ce qui permet de déployer les deux séparément.

    **MLflow n'est pas dans la chaîne de production.** Il a servi à choisir et
    à tracer le modèle, et le conteneur ne l'embarque pas. Le représenter dans
    le flux laisserait croire à une dépendance d'exécution ; il est donc posé
    à l'écart, relié par un trait discontinu qui ne porte pas de requête.
    """
    figure, ax = plt.subplots(figsize=(13.5, 6.4))
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    # Les trois blocs sont alignés **par le haut**, pas centrés : centrés,
    # celui de six lignes débordait sur la rangée du dessous.
    sommet = 0.955
    colonnes = (
        ("agriculteur", 0.015, 0.195, "L'agriculteur", [
            "Décrit sa parcelle",
            "Saisit ses charges",
            "Lit les résultats",
            "Emporte le CSV"], GRIS, "#ffffff"),
        ("streamlit", 0.245, 0.265, "Streamlit — app/app.py", [
            "Trois modes, graphiques, export",
            "Cultures, bornes et pays lus dans l'API",
            "",
            "Aucune logique de modèle :",
            "rien n'est calculé ici"], BLUE, "#f7fafc"),
        ("fastapi", 0.645, 0.340, "FastAPI — service de prédiction", [
            "/predict · /recommend — la parcelle",
            "/national/predict · /national/ranking",
            "/national/series — le panel FAO",
            "/model/info · /reference/prices",
            "",
            "/health · /logs — supervision"], ORANGE, "#fffaf0"),
    )
    bas = {}
    for cle, x, largeur, titre, lignes, couleur, fond in colonnes:
        haut = hauteur(lignes)
        _boite(ax, x, sommet - haut, largeur, titre, lignes, couleur, fond)
        bas[cle] = sommet - haut

    # --- Ce que l'API a derrière elle ------------------------------------- #
    for x, titre, lignes in (
        (0.645, "Deux artefacts .joblib", ["Parcelle et panel FAO, scellés",
                                           "et embarqués dans l'image"]),
        (0.830, "Référentiel FAOSTAT", ["Prix 2024, médiane mondiale",
                                        "re-dérivable par script"]),
    ):
        _boite(ax, x, 0.105, 0.155, titre, lignes, GREEN, "#f0fff4")

    _fleche = lambda d, a: ax.add_patch(FancyArrowPatch(  # noqa: E731
        d, a, arrowstyle="-|>", mutation_scale=15, linewidth=1.4,
        color=GRIS_CLAIR, zorder=1))
    milieu = (sommet + bas["streamlit"]) / 2
    _fleche((0.212, milieu), (0.241, milieu))
    _fleche((0.514, milieu + 0.055), (0.641, milieu + 0.055))
    _fleche((0.641, milieu - 0.055), (0.514, milieu - 0.055))
    ax.text(0.578, milieu + 0.082, "requête HTTP", ha="center", fontsize=8.8,
            color=GRIS, style="italic")
    ax.text(0.578, milieu - 0.105, "réponse JSON", ha="center", fontsize=8.8,
            color=GRIS, style="italic")
    _fleche((0.722, bas["fastapi"]), (0.722, 0.365))
    _fleche((0.907, bas["fastapi"]), (0.907, 0.365))

    # --- MLflow, hors chaîne ---------------------------------------------- #
    lignes_mlflow = ["A servi à choisir le modèle, à suivre les essais et à",
                     "le promouvoir — puis à l'exporter en artefact.",
                     "Le conteneur de production ne l'embarque pas : le trait",
                     "en pointillé ne porte aucune requête."]
    _boite(ax, 0.015, 0.105, 0.495, "MLflow — traçabilité, hors production",
           lignes_mlflow, GRIS_CLAIR, "#f7fafc")
    ax.add_patch(FancyArrowPatch(
        (0.512, 0.232), (0.641, 0.232), arrowstyle="-", linewidth=1.3,
        linestyle=(0, (5, 4)), color=GRIS_CLAIR, zorder=1))

    figure.text(0.015, 0.055,
                "La seule chose qui traverse la frontière du milieu est du "
                "texte : une question en HTTP, une réponse en JSON. "
                "L'interface ne sait pas ce qu'est un modèle, et le service "
                "ne sait pas qu'une interface existe.",
                fontsize=9, color=GRIS, style="italic", va="top")

    enregistrer(figure, "app_architecture.png")


def figure_chaine_mlops() -> None:
    """Du poste au navigateur : ce qui produit les modèles, et ce qui les sert.

    `app_architecture.png` répond à « qui parle à qui pendant un appel ».
    Celle-ci répond à l'autre question : **comment le code et les modèles
    arrivent en production**. Les réunir en un seul schéma mêlerait un flux de
    requêtes — quelques millisecondes, plusieurs fois par minute — à un flux
    de livraison, qui se compte en commits.

    **La frontière qui compte est celle du milieu** : rien ne passe du poste à
    la production sans traverser un commit. Déplacer un alias dans MLflow ne
    déploie rien, et c'est le fait le plus contre-intuitif de la chaîne. Le
    schéma le montre en n'offrant aucune flèche qui contourne GitHub.

    Métriques locales plutôt que `_boite` : trois bandes empilées demandent
    des blocs bien plus plats que les schémas à une rangée, et la figure est
    haute pour que le texte reste lisible malgré des hauteurs relatives
    réduites.
    """
    ligne, entete, marge = 0.035, 0.058, 0.022

    def bloc(x, y, largeur, titre, lignes, couleur, fond="#ffffff") -> None:
        haut = entete + len(lignes) * ligne + marge
        ax.add_patch(FancyBboxPatch(
            (x, y), largeur, haut, boxstyle="round,pad=0.008",
            linewidth=1.6, edgecolor=couleur, facecolor=fond, zorder=2))
        ax.text(x + largeur / 2, y + haut - 0.018, titre, ha="center",
                va="top", fontsize=10.5, fontweight="bold", color=couleur,
                zorder=3)
        for index, texte in enumerate(lignes):
            ax.text(x + 0.016, y + haut - entete - index * ligne, texte,
                    ha="left", va="top", fontsize=9.2, color=GRIS, zorder=3)

    def fleche(depart, arrivee, style="-|>") -> None:
        ax.add_patch(FancyArrowPatch(
            depart, arrivee, arrowstyle=style, mutation_scale=15,
            linewidth=1.4, color=GRIS_CLAIR, zorder=1))

    def bande(y, numero, titre) -> None:
        # Fond blanc opaque et zorder élevé : les deux descentes vers la
        # production sont diagonales et passeraient sinon **au travers** du
        # titre de la bande qu'elles rejoignent.
        ax.text(0.015, y, f"{numero}  ·  {titre}", ha="left", va="bottom",
                fontsize=11.5, fontweight="bold", color=GRIS, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=3))

    figure, ax = plt.subplots(figsize=(14, 8.6))
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    # --- 1. Le poste : tout ce qui n'atteint jamais la production --------- #
    bande(0.960, "1", "Sur le poste — la chaîne MLOps, rejouable par commande")
    for x, titre, lignes in (
        (0.015, "Données sources", ["999 769 parcelles simulées",
                                    "+ panel FAO observé"]),
        (0.265, "Préparation et découpage", ["Deux chaînes, deux découpages :",
                                             "aléatoire · chronologique"]),
        (0.515, "Entraînement et sélection", ["Optuna, 619 runs tracés,",
                                              "quatre modèles comparés"]),
        (0.765, "MLflow — registres", ["Deux registres, alias champion,",
                                       "puis export scellé par empreinte"]),
    ):
        bloc(x, 0.790, 0.220, titre, lignes, GRIS, "#f7fafc")
    for depart in (0.235, 0.485, 0.735):
        fleche((depart, 0.865), (depart + 0.030, 0.865))

    # --- La seule frontière : un commit ---------------------------------- #
    # Elle vise le **dépôt**, pas la CI : un commit arrive dans le dépôt, qui
    # déclenche la CI ensuite — ce second lien est déjà dessiné plus bas.
    fleche((0.165, 0.790), (0.165, 0.640))
    ax.text(0.190, 0.715, "commit de l'artefact exporté — un alias déplacé "
            "ne déploie rien", ha="left", va="center", fontsize=9.4,
            color=GRIS, style="italic")

    # --- 2. GitHub : ce qui vérifie, puis ce qui livre -------------------- #
    bande(0.655, "2", "GitHub — intégration continue, puis livraison")
    for x, largeur, titre, lignes in (
        (0.015, 0.300, "Dépôt", ["Code, artefacts, rapports",
                                 "`main` protégée par la CI"]),
        (0.355, 0.300, "CI — ce qui vérifie", ["302 tests, couverture ≥ 90 %,",
                                               "image et conteneur éprouvés"]),
        (0.705, 0.280, "Docker Hub", ["`latest` + SHA du commit :",
                                      "l'image éprouvée, non rebâtie"]),
    ):
        bloc(x, 0.485, largeur, titre, lignes, BLUE, "#f7fafc")
    fleche((0.315, 0.560), (0.355, 0.560))
    fleche((0.655, 0.560), (0.705, 0.560))

    # --- 3. En ligne : deux hébergeurs, aucune machine à installer -------- #
    bande(0.295, "3", "En ligne — ce que l'agriculteur atteint sans rien "
                      "installer")
    for x, largeur, titre, lignes in (
        (0.015, 0.220, "Navigateur", ["Un lien, aucune installation,",
                                      "aucun compte à créer"]),
        (0.285, 0.300, "Streamlit Community Cloud", [
            "Interface, trois modes",
            "Redéployée à chaque poussée"]),
        (0.645, 0.340, "Render — FastAPI", [
            "Neuf routes, deux modèles servis",
            "Image tirée, jamais reconstruite"]),
    ):
        bloc(x, 0.125, largeur, titre, lignes, ORANGE, "#fffaf0")
    fleche((0.235, 0.200), (0.285, 0.200))
    fleche((0.585, 0.218), (0.645, 0.218))
    fleche((0.645, 0.172), (0.585, 0.172))
    ax.text(0.615, 0.247, "HTTP", ha="center", fontsize=8.8, color=GRIS,
            style="italic")

    # Les deux descentes vers la production, qui ne partent pas du même bloc.
    fleche((0.165, 0.485), (0.400, 0.278))
    ax.text(0.185, 0.370, "webhook", ha="left", fontsize=9.2, color=GRIS,
            style="italic")
    fleche((0.845, 0.485), (0.815, 0.278))
    ax.text(0.860, 0.370, "deploy hook", ha="left", fontsize=9.2, color=GRIS,
            style="italic")

    figure.text(0.015, 0.055,
                "Les deux hébergements sont indépendants : l'un tire une "
                "image, l'autre lit le dépôt. Ce qui les relie n'est pas un "
                "réseau partagé mais une adresse — et c'est ce qui permet de "
                "remplacer l'un sans toucher à l'autre.",
                fontsize=9, color=GRIS, style="italic", va="top")

    enregistrer(figure, "mlops_architecture.png")


def main() -> int:
    """Charge le modèle une fois, puis trace les six figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110

    if not DOSSIER.exists():
        raise SystemExit(f"Dossier de figures introuvable : {DOSSIER}")

    print("=== Figures des rapports des étapes 3 et 4 ===")
    modele = charger_modele()
    print(f"  modèle : {modele.served.registry} v{modele.served.version} "
          f"({modele.served.source})")

    figure_parcours()
    figure_intervalle(modele)
    figure_rendement_vs_marge(modele)
    figure_prix()
    figure_architecture()
    figure_chaine_mlops()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

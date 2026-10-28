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

#: Hypothèses de démonstration. Le prix vient de FAOSTAT via `config.py` ; le
#: rendement attendu et les charges sont **fournis par l'agriculteur** dans le
#: contrat de l'API, et n'ont donc aucune source. Annoncés comme tels sur la
#: figure.
BUDGETS_DEMONSTRATION = {
    "Maize": {"rendement_attendu": 9.0, "charges": 1100.0},
    "Wheat": {"rendement_attendu": 6.5, "charges": 850.0},
}


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

    **Les deux axes partent de zéro, et les deux panneaux ont six
    emplacements.** Zoomer à gauche ferait apparaître des écarts
    spectaculaires là où il n'y en a pas — le piège que ce projet documente
    depuis l'étape 2. Et les quatre emplacements vides à droite *sont* le
    message : sans budget, pas de marge.
    """
    predits = rendements(modele)
    marges = {
        culture: (PRIX_REFERENCE_FAOSTAT[culture]["prix_eur_t"]
                  * demo["rendement_attendu"] - demo["charges"])
        for culture, demo in BUDGETS_DEMONSTRATION.items()
    }
    etendue = max(predits.values()) - min(predits.values())

    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # --- Panneau gauche : six rendements indiscernables ------------------- #
    ordre = sorted(predits, key=predits.get, reverse=True)
    gauche.barh([EN_FRANCAIS[c] for c in ordre], [predits[c] for c in ordre],
                color=BLUE, alpha=0.85, height=0.58)
    gauche.set_xlim(0, max(predits.values()) * 1.30)
    gauche.set_ylim(len(CULTURES) - 0.45, -0.55)
    gauche.set_xlabel("rendement attendu (t/ha)", fontsize=10, color=GRIS)
    gauche.set_title("Le modèle ne départage pas les cultures", fontsize=12.5,
                     fontweight="bold", color="#1a202c", loc="left", pad=12)
    for index, culture in enumerate(ordre):
        gauche.text(predits[culture] + 0.1, index, fr(predits[culture], 2),
                    va="center", fontsize=9.5, color=GRIS)

    # --- Panneau droit : mêmes emplacements, mêmes barres ----------------- #
    classees = sorted(marges, key=marges.get, reverse=True)
    etiquettes = ([EN_FRANCAIS[c] for c in classees]
                  + [EN_FRANCAIS[c] for c in CULTURES if c not in marges])
    droite.barh([EN_FRANCAIS[c] for c in classees],
                [marges[c] for c in classees],
                color=ORANGE, alpha=0.9, height=0.58)
    droite.set_xlim(0, max(marges.values()) * 1.32)
    droite.set_ylim(len(CULTURES) - 0.45, -0.55)
    droite.set_yticks(range(len(etiquettes)))
    droite.set_yticklabels(
        etiquettes,
        # Les cultures sans budget restent nommées, en gris : les effacer
        # laisserait croire que la comparaison porte sur deux cultures seules.
        color=GRIS)
    for etiquette in droite.get_yticklabels()[len(classees):]:
        etiquette.set_color(GRIS_CLAIR)
    droite.set_xlabel("marge brute simulée (€/ha)", fontsize=10, color=GRIS)
    droite.set_title("Ce sont vos chiffres économiques qui tranchent",
                     fontsize=12.5, fontweight="bold", color="#1a202c",
                     loc="left", pad=12)
    for index, culture in enumerate(classees):
        droite.text(marges[culture] + max(marges.values()) * 0.015, index,
                    f"{fr(marges[culture])} €", va="center", fontsize=9.5,
                    color=GRIS)

    # L'espace vide porte son explication, là où le lecteur le remarque.
    droite.text(max(marges.values()) * 0.04, len(classees) + 0.75,
                "Aucun budget fourni pour ces quatre cultures :\n"
                "elles ne sont pas comparées. Rien ne dit\n"
                "qu'elles feraient moins bien.",
                fontsize=9.5, color=GRIS, va="top",
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fffaf0",
                      "edgecolor": ORANGE, "linewidth": 0.9})

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
        "Prix issus de FAOSTAT. Rendements attendus et charges : hypothèses "
        "de démonstration fournies par l'appelant — aucune source du projet "
        "ne donne de charges.\n"
        "La marge la plus élevée n'est pas un avis agronomique : aucune "
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


def main() -> int:
    """Charge le modèle une fois, puis trace les quatre figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110

    if not DOSSIER.exists():
        raise SystemExit(f"Dossier de figures introuvable : {DOSSIER}")

    print("=== Figures du rapport de l'étape 3 ===")
    modele = charger_modele()
    print(f"  modèle : {modele.served.registry} v{modele.served.version} "
          f"({modele.served.source})")

    figure_parcours()
    figure_intervalle(modele)
    figure_rendement_vs_marge(modele)
    figure_prix()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

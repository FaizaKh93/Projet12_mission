"""Produit les deux rapports PDF du projet, depuis leurs sources Markdown.

    reports/rapport_technique.pdf   introduction + les quatre rapports d'étape
    rapport_metier.pdf              le document court, pour un lecteur non technique

Les sources restent en Markdown dans `reports/` : elles se relisent sur GitHub,
se comparent dans une proposition de fusion, et ce script les met en page. Le
PDF est donc **dérivé**, jamais édité à la main — comme les `requirements.txt`
le sont de `uv.lock`.

**Le convertisseur refuse ce qu'il ne connaît pas.** Il ne traite que la syntaxe
réellement employée par les rapports — titres jusqu'au niveau 3, tableaux,
images, citations, puces, listes numérotées, règles horizontales, et les styles
en ligne. Toute autre construction lève `SyntaxeInconnue` plutôt que d'être
ignorée en silence : un paragraphe qui disparaîtrait du PDF sans prévenir serait
pire qu'une erreur.

Exemples :

    uv run python scripts/build_reports.py              # les deux
    uv run python scripts/build_reports.py --technique  # le dossier seul
    uv run python scripts/build_reports.py --metier     # le rapport métier seul
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from PIL import Image as ImagePIL
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from agritech.config import PROJECT_ROOT

DOSSIER = PROJECT_ROOT / "reports"
LARGEUR_UTILE = A4[0] - 4 * cm

#: L'intitulé de la mission, porté par la couverture des deux documents.
TITRE_PROJET = ("Conception d'un système de recommandations "
                "pour une agriculture optimisée par les données")
AUTRICE = "Faïza KHELLADI"

#: Les sources du dossier technique, dans l'ordre de lecture. La seconde
#: chaîne suit la première : elle en reprend les chiffres pour les comparer.
DOSSIER_TECHNIQUE = [
    "introduction.md",
    "data_preparation_summary.md",
    "model_experimentation_summary.md",
    "model_experimentation_panel_summary.md",
    "prediction_api_summary.md",
    "streamlit_app_summary.md",
    "conclusion.md",
]

ARDOISE = colors.HexColor("#1f2d3d")
GRIS = colors.HexColor("#5a6b7b")
GRIS_CLAIR = colors.HexColor("#eef2f6")
BLEU = colors.HexColor("#2d6ca2")


class SyntaxeInconnue(ValueError):
    """Une construction Markdown que ce convertisseur ne sait pas rendre."""


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    commun = dict(fontName="Helvetica", textColor=ARDOISE)

    return {
        "corps": ParagraphStyle("corps", base, fontSize=9.5, leading=14,
                                spaceAfter=7, alignment=TA_JUSTIFY, **commun),
        "h1": ParagraphStyle("h1", base, fontName="Helvetica-Bold", fontSize=17,
                             leading=21, spaceBefore=0, spaceAfter=14,
                             textColor=ARDOISE),
        "h2": ParagraphStyle("h2", base, fontName="Helvetica-Bold", fontSize=12.5,
                             leading=16, spaceBefore=16, spaceAfter=7,
                             textColor=BLEU, keepWithNext=1),
        "h3": ParagraphStyle("h3", base, fontName="Helvetica-Bold", fontSize=10.5,
                             leading=14, spaceBefore=11, spaceAfter=5,
                             textColor=ARDOISE, keepWithNext=1),
        "legende": ParagraphStyle("legende", base, fontSize=8, leading=11.5,
                                  spaceBefore=4, spaceAfter=11,
                                  alignment=TA_CENTER, fontName="Helvetica-Oblique",
                                  textColor=GRIS),
        "citation": ParagraphStyle("citation", base, fontSize=9, leading=13,
                                   alignment=TA_JUSTIFY, **commun),
        "puce": ParagraphStyle("puce", base, fontSize=9.5, leading=13.5,
                               spaceAfter=3, leftIndent=14, bulletIndent=4,
                               alignment=TA_JUSTIFY, **commun),
        "cellule": ParagraphStyle("cellule", base, fontSize=8.5, leading=11.5,
                                  **commun),
        "entete": ParagraphStyle("entete", base, fontSize=8.5, leading=11.5,
                                 fontName="Helvetica-Bold", textColor=colors.white),
        "couverture_titre": ParagraphStyle("ct", base, fontName="Helvetica-Bold",
                                           fontSize=20, leading=27,
                                           alignment=TA_CENTER, textColor=ARDOISE),
        "couverture_sous": ParagraphStyle("cs", base, fontSize=11, leading=17,
                                          alignment=TA_CENTER, textColor=GRIS),
        "couverture_document": ParagraphStyle("cd", base, fontName="Helvetica-Bold",
                                              fontSize=13, leading=18,
                                              alignment=TA_CENTER, textColor=ARDOISE),
    }


# --------------------------------------------------------------------------- #
# Styles en ligne
# --------------------------------------------------------------------------- #

#: Caractères absents de l'encodage des polices standard, et leur équivalent.
#: Sans cette table, ils disparaîtraient du PDF sans le moindre avertissement.
SUBSTITUTIONS = {
    "−": "-",    # signe moins mathématique
    "→": "->",   # flèche vers la droite
    "–": "-",    # tiret demi-cadratin
    " ": " ",  # espace fine insécable -> espace insécable
}

_CODE = re.compile(r"`([^`]+)`")
_LIEN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_GRAS = re.compile(r"\*\*([^*]+)\*\*")
_ITALIQUE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")


def en_ligne(texte: str) -> str:
    """Traduit les styles Markdown vers le mini-HTML de ReportLab.

    Les segments de code et les liens sont mis de côté avant tout traitement :
    un astérisque à l'intérieur d'un `code` ne doit pas devenir de l'italique.
    """
    for absent, equivalent in SUBSTITUTIONS.items():
        texte = texte.replace(absent, equivalent)

    # Les polices standard encodent en cp1252. Tout ce qui en sort serait
    # remplacé sans bruit : mieux vaut s'arrêter et nommer le caractère.
    try:
        texte.encode("cp1252")
    except UnicodeEncodeError as absent:
        fautif = texte[absent.start:absent.end]
        raise SyntaxeInconnue(
            f"Caractère non rendu par les polices standard : {fautif!r} "
            f"(U+{ord(fautif[0]):04X}) dans « {texte[:60]}… ». "
            f"L'ajouter à SUBSTITUTIONS."
        ) from absent

    reserve: list[str] = []

    def garder(remplacement: str) -> str:
        reserve.append(remplacement)
        return f"\x00{len(reserve) - 1}\x01"

    texte = texte.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    texte = _CODE.sub(
        lambda m: garder(f'<font face="Courier" size="8.5">{m.group(1)}</font>'), texte)
    texte = _LIEN.sub(
        lambda m: garder(f'<a href="{m.group(2)}" color="#2d6ca2">{m.group(1)}</a>'),
        texte)
    texte = _GRAS.sub(r"<b>\1</b>", texte)
    texte = _ITALIQUE.sub(r"<i>\1</i>", texte)

    for i, remplacement in enumerate(reserve):
        texte = texte.replace(f"\x00{i}\x01", remplacement)
    return texte


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #


@dataclass
class Convertisseur:
    """Transforme un Markdown en liste d'éléments ReportLab."""

    st: dict[str, ParagraphStyle]
    images: Path

    def convertir(self, source: Path) -> list:
        lignes = source.read_text(encoding="utf-8").splitlines()
        elements: list = []
        i = 0

        while i < len(lignes):
            ligne = lignes[i].rstrip()

            if not ligne:
                i += 1
            elif ligne.startswith("|"):
                bloc, i = self._bloc(lignes, i, lambda l: l.startswith("|"))
                elements.append(self._tableau(bloc))
            elif ligne.startswith("> "):
                bloc, i = self._bloc(lignes, i, lambda l: l.startswith("> "))
                elements.append(self._citation(bloc))
            elif ligne.startswith("- "):
                bloc, i = self._bloc(lignes, i, lambda l: l.startswith("- "))
                elements += self._puces(bloc)
            elif re.match(r"^\d+\. ", ligne):
                bloc, i = self._bloc(lignes, i, lambda l: bool(re.match(r"^\d+\. ", l)))
                elements += self._puces(bloc, numerotee=True)
            else:
                elements += self._une_ligne(ligne, source)
                i += 1

        return self._souder_legendes(elements)

    @staticmethod
    def _souder_legendes(elements: list) -> list:
        """Une figure et sa légende ne se séparent pas sur deux pages.

        Le `Spacer` qui précède l'image reste dehors : c'est lui qui doit
        disparaître si le groupe bascule en haut de la page suivante.
        """
        soudes: list = []
        i = 0
        while i < len(elements):
            suivant = elements[i + 1] if i + 1 < len(elements) else None
            if (isinstance(elements[i], Image)
                    and isinstance(suivant, Paragraph)
                    and suivant.style.name == "legende"):
                soudes.append(KeepTogether([elements[i], suivant]))
                i += 2
            else:
                soudes.append(elements[i])
                i += 1
        return soudes

    @staticmethod
    def _bloc(lignes: list[str], debut: int, appartient) -> tuple[list[str], int]:
        """Rassemble les lignes consécutives qui appartiennent au même bloc."""
        fin = debut
        while fin < len(lignes) and appartient(lignes[fin].rstrip()):
            fin += 1
        return [l.rstrip() for l in lignes[debut:fin]], fin

    def _une_ligne(self, ligne: str, source: Path) -> list:
        if ligne.startswith("#### "):
            raise SyntaxeInconnue(f"{source.name} : titre de niveau 4 non géré — {ligne}")
        if ligne.startswith("### "):
            return [Paragraph(en_ligne(ligne[4:]), self.st["h3"])]
        if ligne.startswith("## "):
            # `keepWithNext` colle le titre au paragraphe suivant, mais laisse
            # les deux seuls en bas de page quand la figure ne suit pas. Sous
            # cinq centimètres de place restante, la section commence à la page
            # d'après.
            return [CondPageBreak(5 * cm),
                    Paragraph(en_ligne(ligne[3:]), self.st["h2"])]
        if ligne.startswith("# "):
            return [Paragraph(en_ligne(ligne[2:]), self.st["h1"])]
        if ligne.startswith("```"):
            raise SyntaxeInconnue(f"{source.name} : bloc de code non géré")
        if re.fullmatch(r"-{3,}", ligne):
            return [Spacer(1, 4),
                    HRFlowable(width="100%", thickness=0.6, color=GRIS_CLAIR),
                    Spacer(1, 8)]
        if ligne.startswith("!["):
            return self._image(ligne, source)
        # Une ligne entièrement en italique est une légende de figure
        if ligne.startswith("*") and ligne.endswith("*") and not ligne.startswith("**"):
            return [Paragraph(en_ligne(ligne), self.st["legende"])]
        return [Paragraph(en_ligne(ligne), self.st["corps"])]

    def _image(self, ligne: str, source: Path) -> list:
        trouve = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", ligne)
        if not trouve:
            raise SyntaxeInconnue(f"{source.name} : image mal formée — {ligne}")

        chemin = (self.images / Path(trouve.group(2)).name)
        if not chemin.exists():
            raise FileNotFoundError(f"{source.name} cite une figure absente : {chemin}")

        with ImagePIL.open(chemin) as fichier:
            largeur_px, hauteur_px = fichier.size

        # Une image beaucoup plus large que haute rend son texte minuscule
        # si on la borne comme une figure ordinaire : elle prend alors
        # toute la largeur utile.
        plafond = LARGEUR_UTILE if largeur_px >= 3 * hauteur_px else 15 * cm
        largeur = min(LARGEUR_UTILE, plafond)
        hauteur = largeur * hauteur_px / largeur_px
        # Une figure très haute mangerait la page : on la borne et on recalcule.
        if hauteur > 16 * cm:
            hauteur = 16 * cm
            largeur = hauteur * largeur_px / hauteur_px

        return [Spacer(1, 6), Image(str(chemin), largeur, hauteur, hAlign="CENTER")]

    def _tableau(self, bloc: list[str]) -> Table:
        lignes = [[c.strip() for c in l.strip("|").split("|")] for l in bloc]
        # La deuxième ligne d'un tableau Markdown n'est qu'un séparateur
        if len(lignes) > 1 and all(re.fullmatch(r":?-{2,}:?", c) for c in lignes[1] if c):
            lignes.pop(1)

        colonnes = max(len(l) for l in lignes)
        lignes = [l + [""] * (colonnes - len(l)) for l in lignes]

        # Largeurs proportionnelles au contenu, bornées pour éviter qu'une
        # colonne de deux mots n'occupe la moitié de la page.
        poids = [max(3, min(60, max(len(l[c]) for l in lignes))) for c in range(colonnes)]
        total = sum(poids)
        largeurs = [LARGEUR_UTILE * p / total for p in poids]

        donnees = [
            [Paragraph(en_ligne(cellule), self.st["entete" if i == 0 else "cellule"])
             for cellule in ligne]
            for i, ligne in enumerate(lignes)
        ]

        tableau = Table(donnees, colWidths=largeurs, repeatRows=1, hAlign="LEFT")
        tableau.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BLEU),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS_CLAIR]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d4de")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return KeepTogether([Spacer(1, 4), tableau, Spacer(1, 10)])

    def _citation(self, bloc: list[str]) -> Table:
        texte = " ".join(l[2:].strip() for l in bloc)
        cellule = Paragraph(en_ligne(texte), self.st["citation"])
        tableau = Table([[cellule]], colWidths=[LARGEUR_UTILE], hAlign="LEFT")
        tableau.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), GRIS_CLAIR),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, BLEU),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return KeepTogether([Spacer(1, 4), tableau, Spacer(1, 9)])

    def _puces(self, bloc: list[str], numerotee: bool = False) -> list:
        elements = []
        for ligne in bloc:
            if numerotee:
                numero, texte = ligne.split(". ", 1)
                puce = f"{numero}."
            else:
                texte, puce = ligne[2:], "•"
            elements.append(
                Paragraph(en_ligne(texte), self.st["puce"], bulletText=puce))
        return elements + [Spacer(1, 5)]


# --------------------------------------------------------------------------- #
# Mise en page
# --------------------------------------------------------------------------- #


class Document(BaseDocTemplate):
    """Un document A4 numéroté, qui sait construire sa table des matières."""

    def __init__(self, chemin: Path, titre: str, **kwargs):
        super().__init__(str(chemin), pagesize=A4, title=titre,
                         author="Agritech Answers", **kwargs)
        cadre = Frame(2 * cm, 2 * cm, LARGEUR_UTILE, A4[1] - 4 * cm, id="corps")
        self.addPageTemplates([PageTemplate(id="page", frames=[cadre],
                                            onPage=self._pied)])
        self._titre = titre
        self._compteur = 0

    def _pied(self, canevas, document):
        canevas.saveState()
        canevas.setFont("Helvetica", 7.5)
        canevas.setFillColor(GRIS)
        if document.page > 1:
            canevas.drawString(2 * cm, 1.2 * cm, self._titre)
            canevas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(document.page))
        canevas.restoreState()

    def beforeDocument(self):
        """Remet les signets à zéro au début de chaque passe.

        `multiBuild` rejoue le document jusqu'à ce que les numéros de page se
        stabilisent. Sans cette remise à zéro, les clés de signet changeraient
        d'une passe à l'autre et la table des matières ne convergerait jamais.
        """
        self._compteur = 0

    def afterFlowable(self, flowable):
        """Alimente la table des matières à mesure que les titres se posent.

        Chaque titre reçoit un signet nommé. Il sert deux fois : la table des
        matières en fait un lien cliquable, et le lecteur PDF en construit son
        volet de navigation. Sans ce repère, le sommaire n'afficherait que des
        numéros de page — vite insuffisant sur un document de quarante pages.
        """
        if not isinstance(flowable, Paragraph):
            return
        niveau = {"h1": 0, "h2": 1}.get(flowable.style.name)
        if niveau is None:
            return

        texte = flowable.getPlainText()
        cle = f"titre-{self._compteur}"
        self._compteur += 1

        self.canv.bookmarkPage(cle)
        self.canv.addOutlineEntry(texte, cle.encode("utf-8"), level=niveau,
                                  closed=niveau > 0)
        self.notify("TOCEntry", (niveau, texte, self.page, cle))


def couverture(st: dict, document: str, sous_titre: str) -> list:
    return [
        Spacer(1, 5 * cm),
        Paragraph("Agritech Answers", st["couverture_sous"]),
        Spacer(1, 0.6 * cm),
        Paragraph(TITRE_PROJET, st["couverture_titre"]),
        Spacer(1, 1 * cm),
        HRFlowable(width="40%", thickness=1.2, color=BLEU, hAlign="CENTER"),
        Spacer(1, 1 * cm),
        Paragraph(document, st["couverture_document"]),
        Spacer(1, 0.3 * cm),
        Paragraph(sous_titre, st["couverture_sous"]),
        Spacer(1, 4 * cm),
        Paragraph(AUTRICE, st["couverture_document"]),
        Paragraph(date.today().strftime("%d/%m/%Y"), st["couverture_sous"]),
        PageBreak(),
    ]


def table_des_matieres(st: dict) -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("toc1", fontName="Helvetica-Bold", fontSize=10.5,
                       leading=18, textColor=ARDOISE),
        ParagraphStyle("toc2", fontName="Helvetica", fontSize=9, leading=14,
                       leftIndent=16, textColor=GRIS),
    ]
    return [Paragraph("Sommaire", st["h1"]), Spacer(1, 6), toc, PageBreak()]


# --------------------------------------------------------------------------- #
# Les deux documents
# --------------------------------------------------------------------------- #


def construire_technique(st: dict) -> Path:
    conv = Convertisseur(st, DOSSIER / "figures")
    chemin = DOSSIER / "rapport_technique.pdf"

    histoire = couverture(
        st, "Rapport technique",
        "Prototype d'estimation de rendement et de comparaison de marges "
        "simulées<br/>Préparation des données · Modélisation · API de prédiction · Interface")
    histoire += table_des_matieres(st)

    for i, nom in enumerate(DOSSIER_TECHNIQUE):
        if i:
            histoire.append(PageBreak())
        histoire += conv.convertir(DOSSIER / nom)

    Document(chemin, "Agritech Answers — rapport technique").multiBuild(histoire)
    return chemin


def construire_metier(st: dict) -> Path:
    conv = Convertisseur(st, DOSSIER / "figures")
    chemin = PROJECT_ROOT / "rapport_metier.pdf"

    histoire = couverture(
        st, "Rapport métier",
        "Prototype d'estimation de rendement et de comparaison de marges "
        "simulées<br/>Résultats, variables clés et recommandations")
    histoire += conv.convertir(DOSSIER / "rapport_metier.md")

    Document(chemin, "Agritech Answers — rapport métier").build(histoire)
    return chemin


def analyser_arguments():
    parseur = argparse.ArgumentParser(
        description="Produit les rapports PDF depuis leurs sources Markdown.")
    parseur.add_argument("--technique", action="store_true",
                         help="ne produire que le dossier technique")
    parseur.add_argument("--metier", action="store_true",
                         help="ne produire que le rapport métier")
    return parseur.parse_args()


def main() -> int:
    arguments = analyser_arguments()
    les_deux = not (arguments.technique or arguments.metier)
    st = styles()

    for faire, construire in [(arguments.technique or les_deux, construire_technique),
                              (arguments.metier or les_deux, construire_metier)]:
        if not faire:
            continue
        try:
            chemin = construire(st)
        except (SyntaxeInconnue, FileNotFoundError) as erreur:
            print(erreur, file=sys.stderr)
            return 1
        except PermissionError as verrou:
            # Sous Windows, un PDF ouvert dans un lecteur ne peut pas être
            # réécrit. C'est le cas le plus fréquent, et une trace d'exécution
            # ne dirait pas quoi faire.
            print(f"Le fichier est verrouillé : {verrou.filename}", file=sys.stderr)
            print("  Il est probablement ouvert dans un lecteur PDF. "
                  "Le fermer, puis relancer.", file=sys.stderr)
            return 1
        poids = chemin.stat().st_size / 1024
        print(f"  {chemin.relative_to(PROJECT_ROOT).as_posix():<34} {poids:>8,.0f} Ko")

    return 0


if __name__ == "__main__":
    sys.exit(main())

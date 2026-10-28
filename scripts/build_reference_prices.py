"""Re-dérive les prix de référence depuis un export FAOSTAT, et les vérifie.

**Pourquoi ce script existe.** Les six prix exposés par `/reference/prices`
sont écrits en dur dans `config.py`. Des nombres écrits à la main sont des
affirmations ; ce script les rend **re-dérivables**, et peut constater qu'ils
n'ont pas dérivé.

L'export FAOSTAT n'est pas versionné — 204 Mo décompressés — et n'a donc pas
sa place dans le dépôt. Il se télécharge sur
`https://bulks-faostat.fao.org/production/Prices_E_All_Data_(Normalized).zip`,
et le chemin du dossier décompressé se passe en argument.

    # Vérifier que config.py n'a pas dérivé de la source
    uv run python scripts/build_reference_prices.py <dossier> --verifier

    # Produire le bloc à recopier dans config.py
    uv run python scripts/build_reference_prices.py <dossier>

**La qualité des valeurs est rapportée, pas filtrée.** FAOSTAT marque chaque
prix d'un drapeau : officiel, estimé, imputé, repris d'un tiers. Les 294
valeurs retenues sont aujourd'hui toutes officielles. Si cela cessait, le
script le dirait au lieu de les écarter sans que personne ne l'ait décidé.

**Un piège, rencontré en écrivant ce script.** `Prices_E_ItemCodes.csv`
orthographie certains produits avec un point-virgule — « Seed cotton; unginned »
— là où le fichier de données met une virgule. Un filtre construit sur le
fichier de codes ne renvoie **aucune ligne, sans lever d'erreur** : le coton
disparaîtrait silencieusement du résultat. Les noms sont donc écrits tels
qu'ils figurent dans les données, et le script refuse de continuer si l'un
d'eux ne ramène rien.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agritech.config import PRIX_REFERENCE_FAOSTAT, PRIX_REFERENCE_SOURCE

#: Élément FAOSTAT retenu : le seul prix comparable entre pays, les prix en
#: monnaie locale ne l'étant pas.
ELEMENT_PRIX_USD = 5532

#: Les codes de zone au-delà de ce seuil désignent des agrégats — Monde,
#: continents, groupes économiques. FAOSTAT n'en publie pas pour les prix à la
#: production, mais le script le vérifie plutôt que de le supposer : une
#: moyenne mondiale glissée parmi les pays fausserait la médiane.
PREMIER_CODE_AGREGAT = 5000

#: Drapeau de qualité FAOSTAT : `A` = valeur officielle déclarée par le pays.
#: Les autres valent `E` (estimée), `I` (imputée par une agence réceptrice),
#: `X` (reprise d'un organisme tiers) et `B` (rupture de série).
#:
#: Au 2026-10-28, les 294 valeurs retenues portent toutes `A`. Ce n'est pas
#: acquis pour les années suivantes : le script **le signale** plutôt que de
#: filtrer en silence. Écarter les valeurs non officielles changerait la
#: médiane sans que personne ne l'ait décidé.
DRAPEAU_OFFICIEL = "A"

#: Du nom de culture du projet vers le produit FAOSTAT, **orthographié comme
#: dans les données**. Le coton graine non égrené est ce qui sort de la ferme ;
#: la fibre égrenée est un produit industriel, vendu près de trois fois plus
#: cher, et n'a rien à faire dans un budget de producteur.
ITEMS = {
    "Barley": "Barley",
    "Cotton": "Seed cotton, unginned",
    "Maize": "Maize (corn)",
    "Rice": "Rice",
    "Soybean": "Soya beans",
    "Wheat": "Wheat",
}

COLONNES = ["Area Code", "Area", "Item", "Element Code", "Year", "Unit",
            "Value", "Flag"]


def charger(dossier: Path, annee: int) -> pd.DataFrame:
    """Les prix à la production de l'année, agrégats exclus.

    Seules sept colonnes sont lues : le fichier normalisé dépasse 1,3 million
    de lignes, et en charger la totalité coûterait plusieurs gigaoctets.
    """
    fichier = dossier / "Prices_E_All_Data_(Normalized).csv"
    if not fichier.exists():
        raise SystemExit(
            f"Fichier introuvable : {fichier}\n"
            "  Télécharger et décompresser Prices_E_All_Data_(Normalized).zip "
            "depuis bulks-faostat.fao.org, puis passer le dossier en argument.")

    brut = pd.read_csv(fichier, encoding="latin-1", usecols=COLONNES,
                       low_memory=False)
    print(f"  {len(brut):,} lignes lues")

    jeu = brut[(brut["Element Code"] == ELEMENT_PRIX_USD)
               & (brut["Year"] == annee)
               & (brut["Item"].isin(ITEMS.values()))
               & (brut["Value"].notna())]

    agregats = jeu[jeu["Area Code"] >= PREMIER_CODE_AGREGAT]
    if not agregats.empty:
        print(f"  {len(agregats)} ligne(s) d'agrégat écartée(s) : "
              f"{sorted(agregats['Area'].unique())}")
    jeu = jeu[jeu["Area Code"] < PREMIER_CODE_AGREGAT]

    unites = set(jeu["Unit"].unique())
    if unites != {"USD"}:
        raise SystemExit(f"Unités inattendues : {unites}. Attendu USD seul.")

    print(f"  {len(jeu):,} lignes retenues (élément {ELEMENT_PRIX_USD}, "
          f"{annee}, {len(ITEMS)} produits)")

    # Qualité des valeurs : signalée, jamais filtrée en douce.
    drapeaux = jeu["Flag"].fillna("(vide)").value_counts()
    non_officielles = int(drapeaux.drop(DRAPEAU_OFFICIEL, errors="ignore").sum())
    detail = ", ".join(f"{d}={n}" for d, n in drapeaux.items())
    print(f"  drapeaux de qualité : {detail}")
    if non_officielles:
        print(f"  ATTENTION — {non_officielles} valeur(s) non officielle(s) "
              f"entrent dans la médiane. Le dire dans PRIX_REFERENCE_SOURCE "
              f"ou restreindre l'extraction, mais ne pas le taire.")
    else:
        print(f"  toutes officielles (drapeau {DRAPEAU_OFFICIEL}) — "
              f"déclarées par les pays, ni estimées ni imputées")
    return jeu


def calculer(jeu: pd.DataFrame, taux: float) -> dict:
    """Médiane et quartiles par culture, convertis en euros.

    Médiane plutôt que moyenne : quelques pays à prix très élevés — souvent
    de petits marchés insulaires — tireraient une moyenne vers le haut sans
    rien dire du prix qu'un producteur rencontre.
    """
    resultat = {}
    for culture, item in ITEMS.items():
        valeurs = jeu[jeu["Item"] == item]["Value"] / taux
        if valeurs.empty:
            raise SystemExit(
                f"Aucune donnée pour « {item} » ({culture}).\n"
                "  Vérifier l'orthographe du produit : le fichier de codes "
                "et le fichier de données ne l'écrivent pas pareil.")
        resultat[culture] = {
            "prix_eur_t": round(valeurs.median()),
            "q1_eur_t": round(valeurs.quantile(0.25)),
            "q3_eur_t": round(valeurs.quantile(0.75)),
            "pays": len(valeurs),
            "item_faostat": item,
        }
    return resultat


def comparer(calcules: dict) -> int:
    """Confronte le calcul aux constantes de `config.py`.

    Rend le nombre d'écarts : zéro signifie que les valeurs servies par l'API
    sont bien celles que la source produit aujourd'hui.
    """
    print(f"\n{'Culture':<10} {'config.py':>28}   {'recalculé':>28}")
    print("-" * 72)
    ecarts = 0
    for culture in ITEMS:
        inscrit = PRIX_REFERENCE_FAOSTAT.get(culture, {})
        calcule = calcules[culture]
        concorde = inscrit == calcule
        ecarts += not concorde
        for source, valeurs in (("config.py", inscrit), ("recalculé", calcule)):
            if source == "config.py":
                print(f"{culture:<10} "
                      f"{valeurs.get('prix_eur_t', '?'):>6} €/t "
                      f"[{valeurs.get('q1_eur_t', '?')}-"
                      f"{valeurs.get('q3_eur_t', '?')}] "
                      f"{valeurs.get('pays', '?')}p", end="   ")
            else:
                print(f"{valeurs['prix_eur_t']:>6} €/t "
                      f"[{valeurs['q1_eur_t']}-{valeurs['q3_eur_t']}] "
                      f"{valeurs['pays']}p"
                      f"   {'OK' if concorde else '<-- ÉCART'}")
    return ecarts


def imprimer_bloc(calcules: dict) -> None:
    """Le bloc Python à recopier dans `config.py`."""
    print("\nPRIX_REFERENCE_FAOSTAT = {")
    for culture, v in calcules.items():
        print(f'    "{culture}": {{"prix_eur_t": {v["prix_eur_t"]}, '
              f'"q1_eur_t": {v["q1_eur_t"]}, "q3_eur_t": {v["q3_eur_t"]},')
        print(f'                {" " * len(culture)}"pays": {v["pays"]}, '
              f'"item_faostat": "{v["item_faostat"]}"}},')
    print("}")


def analyser_arguments():
    """Les options de la ligne de commande."""
    parseur = argparse.ArgumentParser(
        description="Re-dérive les prix de référence depuis un export FAOSTAT.")
    parseur.add_argument(
        "dossier", type=Path,
        help="dossier contenant Prices_E_All_Data_(Normalized).csv")
    parseur.add_argument(
        "--verifier", action="store_true",
        help="compare aux constantes de config.py et sort en erreur si elles "
             "ont dérivé — ce que la CI exécuterait")
    parseur.add_argument(
        "--annee", type=int, default=PRIX_REFERENCE_SOURCE["annee"],
        help=f"année à extraire (défaut : {PRIX_REFERENCE_SOURCE['annee']})")
    return parseur.parse_args()


def main() -> int:
    """Charge, calcule, puis compare ou imprime."""
    arguments = analyser_arguments()
    taux = PRIX_REFERENCE_SOURCE["taux_usd_par_eur"]

    print(f"=== Source : FAOSTAT {arguments.annee}, "
          f"taux {taux} USD/EUR ===")
    calcules = calculer(charger(arguments.dossier, arguments.annee), taux)

    if not arguments.verifier:
        imprimer_bloc(calcules)
        return 0

    ecarts = comparer(calcules)
    if ecarts:
        print(f"\n{ecarts} culture(s) ont dérivé. Mettre config.py à jour, "
              f"et penser à `extrait_le` dans PRIX_REFERENCE_SOURCE.")
        return 1
    print(f"\nLes {len(ITEMS)} cultures concordent avec config.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

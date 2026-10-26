"""Métriques d'évaluation : prédiction, économie, recommandation.

**Bibliothèque, comme `training`** : rien ici ne s'exécute seul, et aucune
fonction n'appelle MLflow. Elles calculent ; `training` journalise.

Trois niveaux, volontairement séparés parce qu'ils ne répondent pas à la même
question :

1. **prédiction** — RMSE, MAE, R² : le rendement prédit est-il juste ? En t/ha ;
2. **économie** — les mêmes erreurs traduites en €/ha, sous les hypothèses de
   prix de `config`. Ce sont des **hypothèses explicites**, pas des prix de
   marché : aucun des deux jeux de données ne contient de prix ni de coût ;
3. **recommandation** — l'écart de rendement prédit entre cultures, à
   conditions de parcelle identiques. C'est ce niveau qui dit si conseiller une
   culture a un sens, et c'est le risque métier réel du projet.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from agritech.config import (
    COUT_SOUSESTIMATION,
    COUT_SURESTIMATION,
    PRIX_TONNE_EUR,
    TARGET,
)


# --------------------------------------------------------------------------- #
# 1. Prédiction — la qualité du rendement prédit
# --------------------------------------------------------------------------- #


def metriques_prediction(y_vrai, y_predit) -> dict[str, float]:
    """Les trois métriques classiques de régression, en tonnes par hectare.

    - **RMSE** : pénalise les grandes erreurs plus que les petites (elles sont
      élevées au carré). C'est la métrique qu'Optuna minimise.
    - **MAE** : erreur moyenne en valeur absolue. Plus lisible pour le métier,
      car elle s'exprime dans l'unité de la cible sans déformation.
    - **R²** : part de la variance expliquée. 0 = aussi bon que de prédire la
      moyenne, 1 = parfait.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_vrai, y_predit))),
        "mae": float(mean_absolute_error(y_vrai, y_predit)),
        "r2": float(r2_score(y_vrai, y_predit)),
    }


# --------------------------------------------------------------------------- #
# 2. Économie — les mêmes erreurs, en euros
# --------------------------------------------------------------------------- #


def metriques_economiques(y_vrai, y_predit,
                          prix_tonne: float = PRIX_TONNE_EUR) -> dict[str, float]:
    """Traduit l'erreur de rendement en euros par hectare.

    Trois indicateurs :

    - **mae_eur_ha** — l'erreur moyenne, en euros. « Nous nous trompons en
      moyenne de X € par hectare. »
    - **cout_asymetrique_eur_ha** — surestimer et sous-estimer ne coûtent pas
      la même chose. Prédire trop haut pousse à sur-investir en intrants et en
      stockage : perte sèche. Prédire trop bas ne fait manquer qu'une
      opportunité. Les deux pondérations viennent de `config`.
    - **biais_eur_ha** — l'erreur *signée* moyenne. Si elle s'éloigne de zéro,
      le modèle se trompe systématiquement dans le même sens, ce que ni le MAE
      ni le RMSE ne montrent.
    """
    # residu > 0 : le modèle a prédit plus que la réalité (surestimation)
    residu = np.asarray(y_predit, dtype=float) - np.asarray(y_vrai, dtype=float)

    cout = np.where(residu > 0,
                    COUT_SURESTIMATION * residu,       # surestimation
                    COUT_SOUSESTIMATION * -residu)     # sous-estimation

    return {
        "mae_eur_ha": float(np.abs(residu).mean() * prix_tonne),
        "cout_asymetrique_eur_ha": float(cout.mean() * prix_tonne),
        "biais_eur_ha": float(residu.mean() * prix_tonne),
    }


def sensibilite_au_prix(y_vrai, y_predit, prix_candidats) -> pd.DataFrame:
    """Rejoue les métriques économiques sur une plage de prix.

    Le prix étant une hypothèse et non une donnée, la bonne parade n'est pas de
    chercher « le bon prix » mais de montrer que la **conclusion ne dépend pas**
    du prix retenu. C'est ce tableau qui le démontre.
    """
    return pd.DataFrame(
        [{"prix_eur_t": prix, **metriques_economiques(y_vrai, y_predit, prix)}
         for prix in prix_candidats]
    ).set_index("prix_eur_t")


# --------------------------------------------------------------------------- #
# 3. Recommandation — le modèle sait-il départager les cultures ?
# --------------------------------------------------------------------------- #


def metriques_par_culture(donnees: pd.DataFrame, y_predit) -> pd.DataFrame:
    """Performance ventilée culture par culture.

    Une erreur globale faible peut masquer une culture systématiquement mal
    traitée : ce tableau sert à le vérifier avant de retenir un modèle.
    """
    table = donnees[["Crop", TARGET]].copy()
    # Affectation positionnelle : `y_predit` est dans l'ordre de `donnees`
    table["prediction"] = np.asarray(y_predit, dtype=float)

    lignes = [
        {"Crop": culture, "n": len(groupe),
         **metriques_prediction(groupe[TARGET], groupe["prediction"])}
        for culture, groupe in table.groupby("Crop", observed=True)
    ]
    return pd.DataFrame(lignes).set_index("Crop").sort_values("rmse")


def sensibilite_culture(pipeline, donnees: pd.DataFrame, features: list[str],
                        cultures: list[str] | None = None,
                        prix_tonne: float = PRIX_TONNE_EUR) -> dict:
    """Contrefactuel : mêmes conditions de parcelle, seule la culture change.

    On ne peut pas observer dans les données ce qu'aurait donné une autre
    culture sur la même parcelle — une ligne, une culture. On demande donc au
    modèle de le simuler : on recopie chaque parcelle six fois en ne changeant
    que `Crop`, et on regarde l'écart entre la meilleure et la pire.

    Les variables **déterminées par la culture** — les indicateurs FAO — sont
    mises à jour en même temps que `Crop`. Sans cela le contrefactuel serait
    incohérent, et l'effet culture sous-estimé.

    C'est la mesure qui décide si une recommandation de culture est
    actionnable. Si l'écart est petit devant l'erreur du modèle, le classement
    est du bruit — et aucun modèle n'y changera rien, puisque l'information
    n'est pas dans les données.

    `classement_constant=True` signale que la même culture arrive en tête
    partout : le « conseil » se réduit alors à une constante.
    """
    cultures = cultures or sorted(donnees["Crop"].unique())

    # Certaines variables sont des FONCTIONS de la culture : les indicateurs
    # FAO ne prennent qu'une valeur par culture. Changer `Crop` sans les mettre
    # à jour fabriquerait des parcelles impossibles — « culture = Rice » avec
    # les indicateurs du maïs — et sous-estimerait l'effet culture, puisque
    # ces colonnes en portent une part.
    candidates = [c for c in features if c != "Crop"]
    constantes = [
        c for c in candidates
        if donnees.groupby("Crop", observed=True)[c].nunique(dropna=False).max() <= 1
    ]
    table_culture = (donnees[["Crop", *constantes]].drop_duplicates("Crop")
                     .set_index("Crop")) if constantes else None

    # Une prédiction par culture, sur les mêmes parcelles
    predictions = {}
    for culture in cultures:
        variante = donnees.copy()
        variante["Crop"] = culture
        for colonne in constantes:
            variante[colonne] = table_culture.loc[culture, colonne]
        predictions[culture] = pipeline.predict(variante[features])

    matrice = pd.DataFrame(predictions, index=donnees.index)
    ecart = matrice.max(axis=1) - matrice.min(axis=1)   # par parcelle
    meilleure = matrice.idxmax(axis=1)

    return {
        "rendement_moyen_par_culture": matrice.mean().sort_values(ascending=False),
        # Part des parcelles où chaque culture arrive en tête. 100 % sur une
        # seule culture = classement identique partout, donc sans valeur.
        "part_en_tete": 100 * meilleure.value_counts(normalize=True),
        "ecart_moyen_t_ha": float(ecart.mean()),
        "ecart_median_t_ha": float(ecart.median()),
        # Le même écart en euros : à comparer au `mae_eur_ha` du modèle
        "gain_max_eur_ha": float(ecart.mean() * prix_tonne),
        "classement_constant": bool(meilleure.nunique() == 1),
    }


# --------------------------------------------------------------------------- #
# 4. Contraintes de déploiement
# --------------------------------------------------------------------------- #


def latence_prediction(pipeline, donnees: pd.DataFrame, features: list[str],
                       repetitions: int = 50, taille_lot: int = 1_000
                       ) -> dict[str, float]:
    """Temps de réponse d'une prédiction, en millisecondes.

    Deux mesures, parce que l'API fera les deux : une ligne à la fois quand un
    agriculteur pose sa question, et un lot pour un traitement groupé.

    Un tour de chauffe précède la mesure : le premier appel paie des
    allocations mémoire qui ne se reproduisent pas ensuite.
    """
    une_ligne = donnees[features].head(1)
    lot = donnees[features].head(taille_lot)

    pipeline.predict(une_ligne)   # chauffe, non mesurée

    debut = time.perf_counter()
    for _ in range(repetitions):
        pipeline.predict(une_ligne)
    unitaire_ms = (time.perf_counter() - debut) / repetitions * 1_000

    debut = time.perf_counter()
    pipeline.predict(lot)
    lot_ms = (time.perf_counter() - debut) * 1_000

    return {
        "latence_unitaire_ms": float(unitaire_ms),
        "latence_lot_ms": float(lot_ms),
        "taille_lot": float(len(lot)),   # peut être < taille_lot si peu de données
    }


# --------------------------------------------------------------------------- #
# 5. Raccourci pour MLflow
# --------------------------------------------------------------------------- #


def evaluer(pipeline, donnees: pd.DataFrame, features: list[str],
            prefixe: str = "val") -> dict[str, float]:
    """Toutes les métriques scalaires d'un modèle sur un jeu, prêtes à journaliser.

    Le préfixe (`val_`, `test_`) évite que les métriques de deux jeux se
    recouvrent dans un même run MLflow.
    """
    prediction = pipeline.predict(donnees[features])
    mesures = {
        **metriques_prediction(donnees[TARGET], prediction),
        **metriques_economiques(donnees[TARGET], prediction),
    }
    return {f"{prefixe}_{nom}": valeur for nom, valeur in mesures.items()}

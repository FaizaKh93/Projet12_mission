"""Endpoints HTTP, et rien d'autre.

Le calcul vit dans `service.py`, le contrat dans `schemas.py`. Ce module se
borne à nommer les routes et à choisir un code de retour : c'est ce qui permet
de tester tout le métier sans client HTTP.

Le modèle est chargé **une fois**, dans le `lifespan`, et non à chaque
requête. Un chargement par requête coûterait quelques centaines de
millisecondes à chaque appel et, pire, masquerait une panne de registre
derrière une simple lenteur.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel

from api.journal import AVERTISSEMENT, CAPACITE, JOURNAL, journaliser, lire
from api.schemas import (
    EntreeJournal,
    JournalReponse,
    PredictionReponse,
    PredictionRequete,
    PrixReferenceReponse,
    RecommandationReponse,
    RecommandationRequete,
    ServedModel,
)
from api.service import (
    ModeleServi,
    charger_modele,
    construire_prediction,
    construire_recommandation,
    info_modele,
    prix_reference,
)

#: Page d'accueil Swagger. **Délibérément courte.** Chaque réserve est portée
#: par le schéma qu'elle concerne — `MargeCulture`, `IntervallePrediction`,
#: `QualiteDonnees` — où elle est lue au moment utile. Ne reviennent ici que
#: les trois mises en garde qu'il faut avoir lues *avant* le premier appel.
DESCRIPTION = """
* **`POST /predict`** — le rendement d'une culture, avec son intervalle de
  prédiction à 95 %.
* **`POST /recommend`** — les six cultures classées par rendement, puis par
  marge brute dès que vous fournissez deux budgets complets.

### Trois choses à savoir avant d'appeler

**Les données d'entraînement sont simulées.** Les effets chiffrés ne sont pas
transposables à des parcelles réelles. Chaque réponse le rappelle dans son
bloc `data_quality`.

**Les chiffres économiques viennent de vous.** Aucun jeu de données du projet
ne contient de prix ni de charges : les budgets sont fournis par l'appelant,
et chaque marge porte `"source": "user"`. `GET /reference/prices` propose des
prix sourcés pour amorcer la saisie — les charges, non.

**L'API annonce ce qu'elle ignore.** Classement sans pouvoir discriminant,
saisie hors du domaine observé, cultures non évaluées faute de budget : la
réserve voyage dans la réponse, pas seulement ici. Et une marge sur une
campagne n'est pas un conseil d'assolement, qui se raisonne en rotation.

*Le détail de chaque réserve figure sur le schéma qui la porte.*
"""

ETIQUETTES = [
    {"name": "prédiction",
     "description": "Le rendement d'une culture dans un contexte donné."},
    {"name": "recommandation",
     "description": "Comparaison des cultures, agronomique et économique."},
    {"name": "service",
     "description": "Supervision et provenance du modèle servi."},
    {"name": "référence",
     "description": "Données économiques externes, sourcées. Aucun calcul, "
                    "aucune dépendance au modèle."},
]

#: Entrée du contrôle de démarrage : les valeurs d'exemple du contrat, au
#: milieu du domaine d'entraînement.
SONDE = PredictionRequete(rainfall_mm=500.0, temperature_celsius=25.0,
                          fertilizer_used=True, irrigation_used=False,
                          crop="Wheat")


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    """Charge et éprouve le modèle au démarrage.

    Un échec ici empêche le service de démarrer, ce qui est voulu : un
    artefact illisible doit se voir dans les journaux du conteneur au
    déploiement, pas dans la requête d'un utilisateur.
    """
    modele = charger_modele()
    construire_prediction(modele, SONDE)
    app.state.modele = modele
    yield
    # Rien à libérer : le pipeline n'ouvre ni fichier ni connexion.


app = FastAPI(
    title="Agritech Answers — API de rendement",
    summary="Rendement attendu par culture, et comparaison économique.",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=ETIQUETTES,
    lifespan=cycle_de_vie,
)

# Enregistré par appel plutôt que par décorateur : `journal.py` est de
# l'infrastructure et ne doit rien savoir de cette application.
app.middleware("http")(journaliser)


def _modele(requete: Request) -> ModeleServi:
    """Le modèle chargé au démarrage, injecté dans chaque route."""
    return requete.app.state.modele


Modele = Annotated[ModeleServi, Depends(_modele)]


def _refuser(erreur: ValueError) -> HTTPException:
    """Une entrée que le transformateur refuse vient de l'appelant.

    Le contrat Pydantic écarte déjà les modalités inconnues. Ce filet
    rattrape une divergence entre les constantes de `config.py` et l'encodeur
    réellement ajusté, et la rend lisible au lieu de produire un 500.
    """
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erreur))


class EtatService(BaseModel):
    """Réponse de `/health` — opérationnelle, hors contrat métier.

    Elle vit ici et non dans `schemas.py` : une sonde de supervision n'est pas
    une promesse faite à l'agriculteur.
    """

    status: Literal["ok"]
    served_model: ServedModel


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.post("/predict", tags=["prédiction"],
          summary="Rendement attendu d'une culture")
def predict(requete: PredictionRequete, modele: Modele) -> PredictionReponse:
    """Prédit le rendement d'une culture dans le contexte fourni.

    Les trois champs catégoriels facultatifs peuvent être omis : le modèle les
    traite comme une information absente, et la réponse le signale.
    """
    try:
        return construire_prediction(modele, requete)
    except ValueError as erreur:
        raise _refuser(erreur) from erreur


@app.post("/recommend", tags=["recommandation"],
          summary="Classement des cultures, par rendement puis par marge")
def recommend(requete: RecommandationRequete,
              modele: Modele) -> RecommandationReponse:
    """Compare les six cultures dans le contexte fourni.

    Le classement par rendement est toujours rendu, avec la note qui en donne
    la portée. Le classement économique n'apparaît qu'à partir de deux budgets
    complets, et ne porte que sur les cultures effectivement budgétées.
    """
    try:
        return construire_recommandation(modele, requete)
    except ValueError as erreur:
        raise _refuser(erreur) from erreur


@app.get("/health", tags=["service"],
         summary="Le service répond et son modèle est chargé")
def health(modele: Modele) -> EtatService:
    """Sonde de supervision. Le modèle étant chargé au démarrage, un 200 ici
    signifie qu'il est en mémoire et qu'il a produit une prédiction."""
    return EtatService(status="ok", served_model=modele.served)


@app.get("/logs", tags=["service"],
         summary="Les derniers appels à /predict et /recommend")
def logs(limit: Annotated[int, Query(ge=1, le=CAPACITE)] = 20) -> JournalReponse:
    """Les derniers appels observés, le plus récent d'abord.

    Les appels **rejetés** y figurent aussi : un 422 renseigne sur ce que les
    appelants saisissent mal, ce qu'un journal des seuls succès tairait.

    Deux limites, répétées dans `caveat` parce qu'elles comptent. Le tampon
    vit dans la mémoire du processus : il disparaît au redémarrage, et chaque
    worker tient le sien. Et cette route n'est protégée par rien, alors
    qu'elle expose les budgets transmis à `/recommend` — les prix et les
    charges de ses utilisateurs.
    """
    entrees = lire(limit)
    return JournalReponse(
        returned=len(entrees), stored=len(JOURNAL), capacity=CAPACITE,
        caveat=AVERTISSEMENT,
        entries=[EntreeJournal(**entree) for entree in entrees])


@app.get("/reference/prices", tags=["référence"],
         summary="Prix à la production de référence, par culture")
def reference_prices() -> PrixReferenceReponse:
    """Les six prix FAOSTAT 2024, avec leur dispersion et leur provenance.

    **Cette route ne calcule rien et ne dépend pas du modèle.** Elle répond
    même si l'artefact venait à manquer : ce sont des données externes, pas un
    résultat d'inférence. Elle sert à amorcer le champ `price_eur_t` des
    budgets de `/recommend` plutôt que de laisser l'appelant deviner.

    Les quartiles sont rendus avec la médiane, délibérément. Lire 267 €/t pour
    le maïs sans voir que les pays déclarants s'étalent de 186 à 432 €/t
    donnerait à ce chiffre une précision qu'il n'a pas.

    Elle ne remplace pas un budget : les charges opérationnelles restent à la
    charge de l'appelant, et sans elles aucune marge ne se calcule.
    """
    return prix_reference()


@app.get("/model/info", tags=["service"],
         summary="Provenance du modèle et valeurs acceptées")
def model_info(modele: Modele) -> dict:
    """Ce que le modèle servi accepte, et d'où il vient.

    La liste des cultures et des modalités **fait foi ici** : c'est celle que
    le contrat dérive et qu'un test confronte à l'encodeur ajusté.
    """
    return info_modele(modele)

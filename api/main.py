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

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from api.journal import (
    AVERTISSEMENT,
    CAPACITE,
    JOURNAL,
    evenement,
    journaliser,
    lire,
)
from api.schemas import (
    ClassementNationalReponse,
    ClassementNationalRequete,
    EntreeJournal,
    JournalReponse,
    PredictionNationaleReponse,
    PredictionNationaleRequete,
    PredictionReponse,
    PredictionRequete,
    PrixReferenceReponse,
    RecommandationReponse,
    RecommandationRequete,
    SeriesNationalesReponse,
    ServedModel,
)
from api.service import (
    ModeleServi,
    _derivation,
    charger_modele,
    charger_modele_national,
    construire_classement_national,
    construire_prediction,
    construire_prediction_nationale,
    construire_recommandation,
    info_modele,
    prix_reference,
    series_nationales,
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
* **`POST /national/predict`** — le rendement **national** d'une culture dans
  un pays, pour l'année qui suit le dernier relevé FAO.
* **`POST /national/ranking`** — les cultures d'un pays classées par rendement
  national estimé.
* **`GET /national/series`** — les couples (pays, culture) que ce service sait
  prédire, et l'année prédite.

### Deux modèles, deux questions

Les routes sans préfixe servent un modèle de **parcelle** sur données
**simulées** : « ce champ, dans ces conditions ». Les routes `/national/*`
servent le **panel FAO**, sur rendements observés mais à l'échelle d'un pays et
d'une année : « ce pays, cette culture, l'an prochain ». Aucune ne remplace
l'autre, et leurs erreurs ne se comparent pas.

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
    {"name": "national",
     "description": "Rendements nationaux annuels, sur le panel FAO. Une "
                    "autre unité d'observation, un autre modèle, un autre "
                    "horizon — jamais une parcelle."},
    {"name": "service",
     "description": "Supervision et provenance des modèles servis."},
    {"name": "référence",
     "description": "Données économiques externes, sourcées. Aucun calcul, "
                    "aucune dépendance au modèle."},
]

#: Entrée du contrôle de démarrage : les valeurs d'exemple du contrat, au
#: milieu du domaine d'entraînement.
SONDE = PredictionRequete(rainfall_mm=500.0, temperature_celsius=25.0,
                          fertilizer_used=True, irrigation_used=False,
                          crop="Wheat")


def _sonde_nationale(modele: ModeleServi) -> PredictionNationaleRequete:
    """Le premier couple de la table embarquée, quel qu'il soit.

    Un couple écrit en dur ferait échouer le démarrage le jour où ce pays
    cesserait de déclarer cette culture — une panne de service pour une
    raison qui n'en est pas une. La table dit elle-même ce qu'elle sait
    prédire ; la sonde lui demande simplement son premier élément.
    """
    aire, culture = _derivation(modele).couples_servables_[0]
    return PredictionNationaleRequete(area=aire, item=culture)


def _charger(nom, charger, eprouver) -> ModeleServi:
    """Charge un artefact, l'éprouve, et dit sur stdout ce qui s'est passé.

    L'échec est tracé **puis relancé** : la trace sert au diagnostic, pas à
    rattraper l'erreur. Un artefact illisible doit continuer d'empêcher le
    démarrage — c'est tout l'intérêt de le charger ici.
    """
    try:
        modele = charger()
        eprouver(modele)
    except Exception as erreur:
        evenement("chargement_modele_echoue", niveau=logging.ERROR, model=nom,
                  error_type=type(erreur).__name__, error=str(erreur))
        raise

    servi = modele.served
    evenement("modele_charge", model=nom, registry=servi.registry,
              version=servi.version, source=servi.source)
    return modele


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    """Charge et éprouve les **deux** modèles au démarrage.

    Un échec ici empêche le service de démarrer, ce qui est voulu : un
    artefact illisible doit se voir dans les journaux du conteneur au
    déploiement, pas dans la requête d'un utilisateur.

    Les deux sont exigés, sans mode dégradé. L'image embarque quatre fichiers —
    deux artefacts et leurs provenances — et son épreuve de conteneur les
    vérifie : un artefact manquant signale une image mal construite, pas une
    fonctionnalité désactivée. Démarrer à moitié reviendrait à rendre des 503
    pour une erreur de `COPY`.

    **Chaque chargement laisse une trace sur stdout**, réussi ou non. C'est la
    seule que l'hébergeur conserve : un conteneur qui refuse de démarrer ne
    répond à aucune route, donc ni `/health` ni `/logs` ne diront jamais
    pourquoi. La provenance y figure aussi — registre, version, artefact
    embarqué ou registre MLflow —, ce qui permet de lire après coup *quel*
    modèle servait au moment d'un incident.
    """
    app.state.modele = _charger("parcelle", charger_modele,
                                lambda m: construire_prediction(m, SONDE))
    app.state.modele_national = _charger(
        "national", charger_modele_national,
        lambda m: construire_prediction_nationale(m, _sonde_nationale(m)))

    yield
    # Rien à libérer : les pipelines n'ouvrent ni fichier ni connexion.


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
    """Le modèle de parcelle chargé au démarrage, injecté dans chaque route."""
    return requete.app.state.modele


def _modele_national(requete: Request) -> ModeleServi:
    """Le modèle de panel, injecté dans les routes `/national/*`."""
    return requete.app.state.modele_national


Modele = Annotated[ModeleServi, Depends(_modele)]
ModeleNational = Annotated[ModeleServi, Depends(_modele_national)]


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
    served_model_national: ServedModel = Field(
        ..., description="Le second artefact, celui du panel FAO. Deux "
                         "champs et non une liste : les deux modèles ne sont "
                         "pas interchangeables, et un client qui ne connaît "
                         "que la chaîne des parcelles continue de lire "
                         "`served_model`.")


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
def health(modele: Modele, national: ModeleNational) -> EtatService:
    """Sonde de supervision. Les modèles étant chargés au démarrage, un 200
    ici signifie que les deux sont en mémoire et ont produit une prédiction."""
    return EtatService(status="ok", served_model=modele.served,
                       served_model_national=national.served)


@app.get("/logs", tags=["service"],
         summary="Les derniers appels métier reçus par le service")
def logs(limit: Annotated[int, Query(ge=1, le=CAPACITE)] = 20) -> JournalReponse:
    """Les derniers appels observés, le plus récent d'abord.

    Les quatre routes métier y figurent — les deux de parcelle et les deux du
    service national. Les catalogues en sont exclus : une interface appelle
    `/national/series` à chaque chargement de page, et ces appels ne disent
    rien de ce qu'un utilisateur cherche à savoir.

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


@app.post("/national/predict", tags=["national"],
          summary="Rendement national attendu d'une culture dans un pays")
def national_predict(requete: PredictionNationaleRequete,
                     modele: ModeleNational) -> PredictionNationaleReponse:
    """Prédit le rendement national d'une culture pour l'année à venir.

    L'appelant fournit un pays et une culture, **rien d'autre** : les trois
    derniers rendements connus de cette série sont cherchés dans la table
    embarquée, et la réponse les rend pour que le chiffre se lise avec son
    historique.

    L'année prédite est celle qui suit le dernier relevé du panel. Ce n'est
    pas la campagne en cours, et `target_year` l'annonce dans chaque réponse.

    Un couple dont l'historique s'interrompt avant l'année de référence est
    refusé en 422 : `GET /national/series` donne ceux qui sont servis.
    """
    try:
        return construire_prediction_nationale(modele, requete)
    except ValueError as erreur:
        raise _refuser(erreur) from erreur


@app.post("/national/ranking", tags=["national"],
          summary="Cultures d'un pays, classées par rendement estimé")
def national_ranking(requete: ClassementNationalRequete,
                     modele: ModeleNational) -> ClassementNationalReponse:
    """Classe les cultures du pays par rendement national estimé.

    **Un classement par rendement, pas par rentabilité** : il manque six prix
    sur dix pour ces cultures, et `economic_note` le dit avec la réponse.

    Seules les cultures que le pays a effectivement cultivées entrent dans la
    comparaison. Sans ce garde-fou, la France verrait le plantain en deuxième
    position — le modèle lui prêterait un historique qui n'est pas le sien.
    """
    try:
        return construire_classement_national(modele, requete)
    except ValueError as erreur:
        raise _refuser(erreur) from erreur


@app.get("/national/series", tags=["national"],
         summary="Couples (pays, culture) que le service sait prédire")
def national_series(modele: ModeleNational) -> SeriesNationalesReponse:
    """Le vocabulaire du service national, et l'année qu'il prédit.

    **Cette liste fait foi** : c'est la table embarquée dans l'artefact, la
    seule à savoir quelles séries ont un historique exploitable. Une interface
    qui recopierait des pays et des cultures proposerait tôt ou tard un couple
    que le modèle ne connaît pas.
    """
    return series_nationales(modele)


@app.get("/model/info", tags=["service"],
         summary="Provenance du modèle et valeurs acceptées")
def model_info(modele: Modele) -> dict:
    """Ce que le modèle servi accepte, et d'où il vient.

    La liste des cultures et des modalités **fait foi ici** : c'est celle que
    le contrat dérive et qu'un test confronte à l'encodeur ajusté.
    """
    return info_modele(modele)

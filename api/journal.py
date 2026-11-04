"""Journal des appels : un tampon borné en mémoire, et une trace sur stdout.

**Deux destinations, et elles ne portent pas la même chose.** Le tampon garde
les corps complets pour `/logs`, et disparaît au redémarrage. La sortie
standard ne reçoit que les **métadonnées** — route, statut, durée,
identifiant — mais l'hébergeur les ramasse et les conserve : sept jours sur
l'offre gratuite de Render. C'est la seule des deux qui survit à une mise en
veille, et une mise en veille survient toutes les quinze minutes d'inactivité.

La répartition n'est pas un détail de confort. Les corps de `/recommend`
contiennent les prix et les charges de l'appelant : les déverser sur stdout
les afficherait dans un tableau de bord pendant une semaine. Ils restent donc
dans le tampon, et seul `request_id` fait le lien entre une ligne de log et
l'appel complet.

**Ce n'est pas un journal d'audit.** Le tampon vit dans la mémoire d'un
worker, disparaît au redémarrage, et `uvicorn --workers N` en crée N
exemplaires indépendants qui s'ignorent. Il sert à voir ce que l'API reçoit et
renvoie pendant une démonstration ou une mise au point — pas à prouver ce qui
s'est passé.

**Les appels rejetés sont journalisés aussi, et c'est le point.** Un 422 dit
quelle saisie l'utilisateur rate ; en supervision c'est souvent plus
instructif qu'un succès. C'est la raison de l'interception en middleware :
un handler ne voit jamais une requête invalide, ni un corps JSON malformé.

**Aucune authentification.** Les corps de `/recommend` contiennent les prix et
les charges de l'appelant. Exposer cette route publiquement reviendrait à
publier les données économiques de ses utilisateurs.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4

from starlette.requests import Request
from starlette.responses import Response

#: Capacité du tampon : `deque` évince le plus ancien de lui-même.
CAPACITE = 100

#: Seules ces routes sont journalisées : les quatre appels **métier**, ceux où
#: un utilisateur demande une réponse sur sa situation. `/health` noierait le
#: journal sous les sondes de supervision, `/logs` s'auto-alimenterait à chaque
#: consultation, et les routes de référence — `/model/info`, `/reference/prices`,
#: `/national/series` — ne portent aucune saisie : ce sont des catalogues, et
#: une interface les appelle à chaque chargement de page.
CHEMINS_JOURNALISES = ("/predict", "/recommend",
                       "/national/predict", "/national/ranking")

JOURNAL: deque[dict] = deque(maxlen=CAPACITE)

AVERTISSEMENT = (
    "Journal en mémoire du processus : perdu au redémarrage, et propre à "
    "chaque worker — `uvicorn --workers N` en tient N séparés. Ce n'est pas "
    "un journal d'audit. Cette route rend les budgets transmis à /recommend — "
    "les prix et les charges de ses appelants : ne l'ouvrir qu'à qui peut les "
    "lire."
)


def _sortie_standard(nom: str = "agritech.api") -> logging.Logger:
    """Un logger qui écrit une ligne brute sur la sortie standard.

    Aucun formateur bavard : le message **est** déjà un objet JSON, et le
    préfixer d'un horodatage texte le rendrait illisible aux outils qui le
    relisent. L'hôte — Render, Docker, Kubernetes — ramasse stdout et date
    lui-même chaque ligne.

    `propagate = False` est nécessaire, pas décoratif : sans lui, uvicorn
    réémettrait la même ligne dans son propre format, et chaque appel
    apparaîtrait deux fois dans le tableau de bord.
    """
    logger = logging.getLogger(nom)
    if not logger.handlers:
        sortie = logging.StreamHandler(sys.stdout)
        sortie.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sortie)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = _sortie_standard()


def _maintenant() -> str:
    """L'instant courant en UTC, à la seconde."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evenement(nom: str, niveau: int = logging.INFO, **champs) -> None:
    """Émet une ligne JSON sur stdout : un fait, daté et nommé.

    Volontairement générique — ce module est de l'infrastructure et ne sait
    rien de l'application qui l'utilise. C'est l'appelant qui nomme ses
    événements et choisit ses champs, comme il nomme déjà ses routes.

    `default=str` évite qu'une valeur non sérialisable fasse échouer un
    appel : un journal qui casse le service qu'il observe serait pire que pas
    de journal du tout.
    """
    LOGGER.log(niveau, json.dumps(
        {"timestamp": _maintenant(), "level": logging.getLevelName(niveau),
         "event": nom, **champs},
        ensure_ascii=False, default=str))


def _charge(octets: bytes):
    """Le corps décodé, ou une trace de ce qui n'a pas pu l'être.

    Un corps illisible est précisément ce qu'on veut voir dans un journal :
    il est conservé en texte tronqué plutôt qu'écarté.
    """
    if not octets:
        return None
    try:
        return json.loads(octets)
    except (ValueError, UnicodeDecodeError):
        return {"_non_json": octets[:500].decode("utf-8", "replace")}


async def journaliser(requete: Request, suivant):
    """Middleware HTTP : enregistre date, entrée, sortie, statut et durée."""
    if requete.url.path not in CHEMINS_JOURNALISES:
        return await suivant(requete)

    corps = await requete.body()

    # Lire le corps ici épuise le canal de réception, et le handler en aval
    # attendrait indéfiniment un contenu déjà consommé. On le rejoue.
    async def rejouer():
        """Rend le corps déjà lu, en un seul message ASGI."""
        return {"type": "http.request", "body": corps, "more_body": False}

    requete._receive = rejouer

    debut = time.perf_counter()
    reponse = await suivant(requete)
    duree_ms = (time.perf_counter() - debut) * 1000

    if not hasattr(reponse, "body_iterator"):
        return reponse

    sortie = b"".join([section async for section in reponse.body_iterator])

    # Court mais suffisant : il ne sert qu'à relier une ligne de log à son
    # entrée de tampon, dans une fenêtre de cent appels.
    identifiant = uuid4().hex[:12]
    JOURNAL.append({
        "timestamp": _maintenant(),
        "request_id": identifiant,
        "endpoint": requete.url.path,
        "status_code": reponse.status_code,
        "duration_ms": round(duree_ms, 1),
        "request": _charge(corps),
        "response": _charge(sortie),
    })

    # Les métadonnées seulement : les corps restent dans le tampon.
    evenement(
        "appel_api",
        niveau=logging.INFO if reponse.status_code < 400 else logging.WARNING,
        request_id=identifiant, method=requete.method,
        route=requete.url.path, status=reponse.status_code,
        duration_ms=round(duree_ms, 1))

    # Le flux ayant été consommé pour être lu, il faut le rendre entier.
    return Response(content=sortie, status_code=reponse.status_code,
                    headers=dict(reponse.headers),
                    media_type=reponse.media_type)


def lire(limite: int) -> list[dict]:
    """Les `limite` entrées les plus récentes, la dernière d'abord."""
    return list(JOURNAL)[-limite:][::-1]

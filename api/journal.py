"""Journal des appels : un tampon borné, en mémoire du processus.

**Ce n'est pas un journal d'audit.** Il vit dans la mémoire d'un worker,
disparaît au redémarrage, et `uvicorn --workers N` en crée N exemplaires
indépendants qui s'ignorent. Il sert à voir ce que l'API reçoit et renvoie
pendant une démonstration ou une mise au point — pas à prouver ce qui s'est
passé.

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
import time
from collections import deque
from datetime import datetime, timezone

from starlette.requests import Request
from starlette.responses import Response

#: Capacité du tampon : `deque` évince le plus ancien de lui-même.
CAPACITE = 100

#: Seules ces routes sont journalisées. `/health` noierait le journal sous les
#: sondes de supervision, et `/logs` s'auto-alimenterait à chaque consultation.
CHEMINS_JOURNALISES = ("/predict", "/recommend")

JOURNAL: deque[dict] = deque(maxlen=CAPACITE)

AVERTISSEMENT = (
    "Journal en mémoire du processus : perdu au redémarrage, et propre à "
    "chaque worker — `uvicorn --workers N` en tient N séparés. Ce n'est pas "
    "un journal d'audit. Aucune authentification ne protège cette route, qui "
    "expose les budgets transmis à /recommend : ne pas l'ouvrir publiquement."
)


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

    JOURNAL.append({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "endpoint": requete.url.path,
        "status_code": reponse.status_code,
        "duration_ms": round(duree_ms, 1),
        "request": _charge(corps),
        "response": _charge(sortie),
    })

    # Le flux ayant été consommé pour être lu, il faut le rendre entier.
    return Response(content=sortie, status_code=reponse.status_code,
                    headers=dict(reponse.headers),
                    media_type=reponse.media_type)


def lire(limite: int) -> list[dict]:
    """Les `limite` entrées les plus récentes, la dernière d'abord."""
    return list(JOURNAL)[-limite:][::-1]

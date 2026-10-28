"""Éprouve l'image de service dans un conteneur réel.

**Ce que la suite pytest ne peut pas dire.** Les 93 tests tournent dans
l'environnement de développement : ils ont MLflow, PyArrow et les données sous
la main. Ils ne prouvent donc rien sur l'autonomie du conteneur, qui n'a ni
registre, ni `mlflow.db`, ni `mlartifacts/`. C'est l'objet de ce script.

Trois familles de contrôles :

1. **Autonomie** — le conteneur démarre sans aucun montage, et l'image ne
   contient ni MLflow, ni PyArrow, ni uv.
2. **Service** — les routes répondent, et la version servie est celle que la
   provenance embarquée annonce.
3. **Refus** — un artefact qui ne correspond pas à sa provenance doit faire
   **échouer le démarrage**, pas produire des prédictions mal estampillées.
   C'est le contrôle qui justifie le sceau : sans lui, le reste n'est qu'une
   déclaration.

    uv run python scripts/smoke_docker.py
    uv run python scripts/smoke_docker.py --construire --image agritech-api:v2
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Nom du conteneur de test. Fixe, pour qu'un essai interrompu se nettoie au
#: suivant plutôt que de laisser des conteneurs orphelins s'accumuler.
CONTENEUR = "agritech-api-smoke"

#: Contexte au milieu du domaine d'entraînement : aucun avertissement attendu.
CONTEXTE = {"rainfall_mm": 500.0, "temperature_celsius": 25.0,
            "fertilizer_used": True, "irrigation_used": False}

#: Deux budgets suffisent à exercer le classement économique et à vérifier que
#: les quatre cultures sans budget sont annoncées comme non évaluées.
BUDGETS = {
    "Wheat": {"expected_yield_t_ha": 6.0, "variable_costs_eur_ha": 800.0,
              "price_eur_t": 225.0},
    "Maize": {"expected_yield_t_ha": 9.0, "variable_costs_eur_ha": 1100.0,
              "price_eur_t": 267.0},
}

#: Paquets qui n'ont rien à faire dans une image de service.
ABSENTS_ATTENDUS = ("mlflow", "pyarrow", "lightgbm", "optuna", "shap",
                    "jupyterlab", "pytest", "uv")

RACINE = Path(__file__).resolve().parents[1]
echecs: list[str] = []


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #


def titre(texte: str) -> None:
    """Un intertitre lisible dans la sortie."""
    print(f"\n{'=' * 74}\n{texte}\n{'=' * 74}")


def verifier(libelle: str, condition: bool, detail: str = "") -> bool:
    """Journalise un contrôle et retient les échecs pour le code de sortie."""
    print(f"  {'OK  ' if condition else 'ECHEC'}  {libelle}"
          f"{'  — ' + detail if detail else ''}")
    if not condition:
        echecs.append(libelle)
    return condition


def docker(*arguments: str, capture: bool = True) -> subprocess.CompletedProcess:
    """Appelle `docker` sans passer par un interpréteur de commandes.

    Les arguments sont transmis tels quels : sous Windows, un chemin passé au
    travers d'un shell POSIX serait réécrit avant d'atteindre le démon.
    """
    return subprocess.run(["docker", *arguments], capture_output=capture,
                          text=True, encoding="utf-8", errors="replace")


def port_libre() -> int:
    """Un port que le système attribue, pour ne pas entrer en conflit."""
    with socket.socket() as prise:
        prise.bind(("127.0.0.1", 0))
        return prise.getsockname()[1]


def appeler(port: int, chemin: str, charge=None, timeout: int = 20):
    """GET, ou POST si une charge utile est fournie."""
    donnees = json.dumps(charge).encode() if charge is not None else None
    requete = urllib.request.Request(
        f"http://127.0.0.1:{port}{chemin}", data=donnees,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(requete, timeout=timeout) as reponse:
        return reponse.status, json.loads(reponse.read())


def attendre_sante(port: int, limite: int = 120) -> float:
    """Sonde `/health` jusqu'à ce que le modèle soit chargé et vérifié."""
    debut = time.perf_counter()
    while time.perf_counter() - debut < limite:
        if docker("inspect", "-f", "{{.State.Running}}",
                  CONTENEUR).stdout.strip() != "true":
            raise RuntimeError(
                "le conteneur s'est arrêté :\n"
                + docker("logs", CONTENEUR).stdout[-2000:])
        try:
            if appeler(port, "/health", timeout=3)[0] == 200:
                return time.perf_counter() - debut
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.5)
    raise RuntimeError("`/health` n'a pas répondu dans le délai imparti.")


def nettoyer() -> None:
    """Supprime le conteneur de test, qu'il tourne ou non."""
    docker("rm", "-f", CONTENEUR)


# --------------------------------------------------------------------------- #
# 1. Autonomie
# --------------------------------------------------------------------------- #


def eprouver_autonomie(image: str) -> None:
    """L'image ne contient que ce qu'il faut pour servir."""
    titre("1. Autonomie de l'image")

    programme = (
        "import importlib.util as u, json, sys, os\n"
        "absents = {}\n"
        f"for m in {ABSENTS_ATTENDUS!r}:\n"
        "    try: absents[m] = u.find_spec(m) is not None\n"
        "    except Exception: absents[m] = False\n"
        "import pandas, sklearn, numpy\n"
        "from agritech.config import PROJECT_ROOT, API_MODEL_PATH\n"
        "print(json.dumps({'presents': [m for m, p in absents.items() if p],\n"
        "                  'python': sys.version.split()[0],\n"
        "                  'pandas': pandas.__version__,\n"
        "                  'sklearn': sklearn.__version__,\n"
        "                  'numpy': numpy.__version__,\n"
        "                  'racine': str(PROJECT_ROOT),\n"
        "                  'artefact': API_MODEL_PATH.exists(),\n"
        "                  'uid': os.getuid()}))"
    )
    acheve = docker("run", "--rm", "--entrypoint", "python", image,
                    "-c", programme)
    if acheve.returncode != 0:
        verifier("l'image s'inspecte", False, acheve.stderr[-300:])
        return

    etat = json.loads(acheve.stdout.strip().splitlines()[-1])
    verifier("aucun paquet d'entraînement embarqué", not etat["presents"],
             ", ".join(etat["presents"]) or "ni mlflow, ni pyarrow, ni uv")
    verifier("PROJECT_ROOT résout sur /app", etat["racine"] == "/app",
             etat["racine"])
    verifier("l'artefact est à sa place", etat["artefact"])
    verifier("le service ne tourne pas en root", etat["uid"] != 0,
             f"uid={etat['uid']}")
    print(f"         python {etat['python']}  pandas {etat['pandas']}  "
          f"sklearn {etat['sklearn']}  numpy {etat['numpy']}")


# --------------------------------------------------------------------------- #
# 2. Service
# --------------------------------------------------------------------------- #


def eprouver_service(image: str) -> None:
    """Le conteneur démarre sans montage et répond sur toutes ses routes."""
    titre("2. Service — aucun montage, aucun registre")

    port = port_libre()
    nettoyer()
    # Volontairement sans `-v` : ni mlflow.db, ni mlartifacts, ni data. Si le
    # conteneur en avait besoin, il échouerait ici.
    lancement = docker("run", "-d", "--name", CONTENEUR,
                       "-p", f"127.0.0.1:{port}:8000", image)
    if lancement.returncode != 0:
        verifier("le conteneur démarre", False, lancement.stderr[-300:])
        return

    try:
        duree = attendre_sante(port)
        print(f"  démarrage en {duree:.1f} s, port {port}, sans aucun montage")

        _, sante = appeler(port, "/health")
        servi = sante["served_model"]
        verifier("/health répond 200 et le modèle est chargé",
                 sante["status"] == "ok")
        verifier("le modèle vient de l'artefact embarqué",
                 servi["source"] == "bundled-joblib", servi["source"])

        _, info = appeler(port, "/model/info")
        provenance = info["provenance"]
        verifier("la version servie est celle de la provenance",
                 servi["version"] == provenance["version"],
                 f"v{servi['version']}")
        verifier("la provenance porte l'empreinte vérifiée au démarrage",
                 bool(provenance.get("artefact_sha256")),
                 str(provenance.get("artefact_sha256"))[:12] + "…")
        print(f"         registre {servi['registry']} v{servi['version']}  "
              f"alias {provenance.get('alias')!r}")

        statut, predit = appeler(port, "/predict", {**CONTEXTE, "crop": "Wheat"})
        verifier("/predict rend un rendement et son intervalle",
                 statut == 200 and "prediction_interval_95" in predit,
                 f"Wheat {predit.get('yield_t_ha')} t/ha")

        statut, reco = appeler(port, "/recommend",
                               {**CONTEXTE, "budgets": BUDGETS})
        economique = reco["economic_ranking"]
        verifier("/recommend classe les six rendements", statut == 200
                 and len(reco["yield_ranking"]) == 6)
        verifier("le classement économique reste borné aux budgets fournis",
                 sorted(economique["assessed_crops"]) == ["Maize", "Wheat"]
                 and len(economique["unassessed_crops"]) == 4,
                 f"évaluées {economique['assessed_crops']}")
        verifier("le pouvoir discriminant du rendement est annoncé",
                 "ranking_discriminative" in reco,
                 f"écart {reco['yield_spread_t_ha']} t/ha, "
                 f"discriminant={reco['ranking_discriminative']}")

        # Un refus doit être journalisé : c'est la raison d'être du middleware.
        try:
            appeler(port, "/predict", {**CONTEXTE, "crop": "Quinoa"})
            verifier("une culture inconnue est rejetée", False, "acceptée")
        except urllib.error.HTTPError as erreur:
            verifier("une culture inconnue est rejetée", erreur.code == 422,
                     f"HTTP {erreur.code}")

        # Données externes : la route doit répondre sans toucher au modèle, et
        # ses prix doivent pouvoir alimenter un budget de /recommend.
        statut, reference = appeler(port, "/reference/prices")
        cultures_prix = {p["crop"] for p in reference["prices"]}
        verifier("/reference/prices couvre les six cultures",
                 statut == 200 and len(cultures_prix) == 6,
                 f"{len(cultures_prix)} cultures, source "
                 f"{reference['source']['source']} {reference['source']['year']}")
        verifier("les prix portent leur dispersion entre pays",
                 all(p["q1_eur_t"] <= p["price_eur_t"] <= p["q3_eur_t"]
                     for p in reference["prices"]))
        verifier("la réserve dit que les charges manquent",
                 "charges" in reference["caveat"])

        statut, journal = appeler(port, "/logs?limit=10")
        verifier("/logs a enregistré les appels, refus compris",
                 statut == 200 and journal["stored"] >= 3,
                 f"{journal['returned']}/{journal['stored']} entrées")
        verifier("le refus figure au journal",
                 any(e["status_code"] == 422 for e in journal["entries"]))
    finally:
        nettoyer()


# --------------------------------------------------------------------------- #
# 3. Refus
# --------------------------------------------------------------------------- #


def eprouver_refus(image: str) -> None:
    """Un artefact qui ne correspond pas à sa provenance bloque le démarrage.

    Le conteneur est relancé avec un `.joblib` substitué par un montage. Si
    l'API démarrait quand même, elle servirait un modèle en annonçant la
    provenance d'un autre — précisément ce que le sceau existe pour empêcher.
    """
    titre("3. Refus — un artefact substitué doit bloquer le démarrage")

    nettoyer()
    with tempfile.TemporaryDirectory() as dossier:
        imposteur = Path(dossier) / "modele_api.joblib"
        # Un contenu arbitraire : le sceau est vérifié **avant** le chargement,
        # donc ce fichier n'a pas besoin d'être un pickle valide.
        imposteur.write_bytes(b"ceci n'est pas l'artefact promu")

        acheve = docker(
            "run", "--rm", "--name", CONTENEUR,
            "-v", f"{imposteur}:/app/models/modele_api.joblib:ro", image)
        sortie = acheve.stdout + acheve.stderr

        verifier("le démarrage échoue", acheve.returncode != 0,
                 f"code {acheve.returncode}")
        verifier("le message nomme l'artefact fautif",
                 "n'est pas l'artefact décrit" in sortie)
        verifier("les deux empreintes sont affichées",
                 "attendue" in sortie and "constatée" in sortie)

        if echecs and "le démarrage échoue" in echecs:
            print("\n  --- sortie du conteneur ---")
            print("  " + "\n  ".join(sortie.strip().splitlines()[-12:]))


# --------------------------------------------------------------------------- #
# Point d'entrée
# --------------------------------------------------------------------------- #


def analyser_arguments():
    """Les options de la ligne de commande."""
    parseur = argparse.ArgumentParser(
        description="Éprouve l'image de service dans un conteneur réel.")
    parseur.add_argument("--image", default="agritech-api:latest",
                         help="image à éprouver (défaut : agritech-api:latest)")
    parseur.add_argument("--construire", action="store_true",
                         help="construit l'image avant de l'éprouver")
    return parseur.parse_args()


def main() -> int:
    """Construit si demandé, éprouve, puis rend un code de sortie parlant."""
    arguments = analyser_arguments()

    if docker("info", "--format", "{{.ServerVersion}}").returncode != 0:
        print("Le démon Docker est injoignable. Démarrer Docker Desktop.")
        return 2

    if arguments.construire:
        titre(f"Construction de {arguments.image}")
        if docker("build", "-t", arguments.image, str(RACINE),
                  capture=False).returncode != 0:
            return 2

    try:
        eprouver_autonomie(arguments.image)
        eprouver_service(arguments.image)
        eprouver_refus(arguments.image)
    finally:
        nettoyer()

    titre("Bilan")
    if echecs:
        print(f"  {len(echecs)} contrôle(s) en échec :")
        for echec in echecs:
            print(f"    - {echec}")
        return 1
    print("  Tous les contrôles passent. L'image sert l'artefact promu, "
          "sans registre ni données.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

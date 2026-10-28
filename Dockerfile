# syntax=docker/dockerfile:1
#
# Image de service de l'API Agritech Answers.
#
# **Ce qu'elle n'embarque pas, et c'est l'essentiel.** Ni MLflow, ni PyArrow,
# ni les données d'entraînement, ni `mlflow.db`. Le conteneur charge l'artefact
# embarqué et vérifie son empreinte au démarrage ; il n'a aucun registre à
# consulter. L'absence de PyArrow a été éprouvée avant d'écrire ce fichier :
# le pickle n'y fait aucune référence, et pandas 3.0 retombe sur son dtype
# `str` natif.
#
# **Une seule source de vérité pour les versions : `uv.lock`.** `uv sync
# --locked` échoue si le lock ne décrit plus `pyproject.toml` — un contrôle
# qu'un `requirements.txt` exporté ne sait pas faire. Ce dernier reste produit
# pour le livrable OpenClassrooms, n'est jamais édité à la main, et
# n'intervient pas dans cette construction.
#
#     docker build -t agritech-api .
#     docker run --rm -p 8000:8000 agritech-api

# Version corrective épinglée, et non `3.12-slim` : un pickle scikit-learn se
# recharge sous l'environnement qui l'a produit, pas sous « à peu près ».
# C'est la version du poste de développement et celle qu'exige `uv.lock`
# (`requires-python = "==3.12.*"`).
ARG IMAGE_PYTHON=python:3.12.13-slim-bookworm


# --------------------------------------------------------------------------- #
# Étage 1 — uv résout et installe. Rien de cet étage n'atteint l'image finale
#            sinon l'environnement virtuel lui-même.
# --------------------------------------------------------------------------- #
FROM ${IMAGE_PYTHON} AS constructeur

# uv épinglé à la version qui a produit `uv.lock`. Laisser flotter le tag
# reviendrait à confier la résolution à un outil dont on ignore la version,
# dans le fichier censé la figer.
COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# `--no-install-project` n'est pas une optimisation, c'est une **nécessité**.
# `agritech/config.py` calcule `PROJECT_ROOT = Path(__file__).parents[2]` :
# depuis `/app/src/agritech/` cela donne `/app`, et `models/` est trouvé ;
# depuis `site-packages/agritech/` cela donnerait `/usr/local/lib/python3.12`,
# et le conteneur ne retrouverait jamais son modèle. Le paquet reste donc dans
# l'arborescence source, et seules les dépendances entrent dans le venv.
#
# `--no-default-groups --group api` écarte `train`, `notebooks` et `dev` :
# 25 paquets au lieu de 179, sans MLflow, PyArrow, LightGBM ni JupyterLab.
#
# Le montage en liaison évite de copier les fichiers dans une couche : seule
# l'arborescence du venv est conservée, et la couche reste valide tant que ni
# le lock ni le manifeste ne bougent.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-default-groups --group api


# --------------------------------------------------------------------------- #
# Étage 2 — l'image de service. Ni uv, ni cache de construction.
# --------------------------------------------------------------------------- #
FROM ${IMAGE_PYTHON}

# Un compte dédié : un serveur HTTP exposé n'a aucune raison d'être root.
RUN useradd --create-home --uid 1000 agritech

WORKDIR /app

# Le venv est copié au **même chemin absolu** qu'à la construction : les
# shebangs et `pyvenv.cfg` qu'il contient y sont inscrits en dur.
COPY --from=constructeur --chown=agritech:agritech /app/.venv /app/.venv

# Le code, puis l'artefact. Dans cet ordre : le modèle change plus souvent que
# le code du service, donc il occupe la dernière couche.
COPY --chown=agritech:agritech src/ /app/src/
COPY --chown=agritech:agritech api/ /app/api/
COPY --chown=agritech:agritech models/modele_api.joblib models/modele_api.json /app/models/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src:/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER agritech
EXPOSE 8000

# La sonde interroge la vraie route : `/health` ne répond 200 qu'une fois le
# modèle chargé, son empreinte vérifiée et une prédiction de contrôle faite.
# `start-period` couvre ce démarrage, mesuré à 2,5 s hors conteneur.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request as r, sys; \
sys.exit(0 if r.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# `--host 0.0.0.0` : sur la boucle locale, le port publié ne joindrait rien.
# Un seul worker — le journal de `/logs` vit en mémoire de processus, plusieurs
# workers en tiendraient autant d'exemplaires indépendants.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1"]

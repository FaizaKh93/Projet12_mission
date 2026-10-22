# Agritech Answers — prédiction de rendement et recommandation de culture

Prototype MLOps : une API de prédiction du rendement agricole et de recommandation
de culture, interrogée par une interface Streamlit.

## Structure

```
data/OC_InputData/     jeux de données sources (non versionnés)
notebooks/             analyse exploratoire et préparation des données
src/agritech/          code du projet (configuration, chargement, profilage)
```

## Démarrage

```bash
uv sync                # crée .venv et installe les dépendances
uv run jupyter lab
```

## Jeux de données

- **Agriculture Crop Yield** — 1 000 000 parcelles simulées, cible `Yield_tons_per_hectare`.
- **Crop Yield Prediction** — données FAO pays × année × culture, plus ses quatre fichiers sources.

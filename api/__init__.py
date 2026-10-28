"""Application de service : l'API de prédiction de rendement.

Distincte du paquet `agritech`, qui est la **bibliothèque** — préparation,
entraînement, transformateur de dérivation. Même séparation qu'entre
`src/agritech/training.py` et `scripts/train.py` : ici vit l'application, et
elle n'est importée par rien d'autre qu'elle-même et ses tests.
"""

# Projet MLOps - GermanCredit

Ce projet est une pipeline MLOps professionnelle pour l'analyse et la modélisation du dataset GermanCredit.

## Structure du projet

- `data/` : Données brutes et traitées
- `notebooks/` : Notebooks d'exploration et de prototypage
- `src/` : Code source (préprocessing, modèles, etc.)
- `tests/` : Tests unitaires

## Démarrage rapide

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Télécharger les données :
   ```bash
   python src/data/download_data.py
   ```
3. Lancer l'exploration :
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ``` 


source .venv/bin/activatepython -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
dvc init
dvc repro
- Git gère les versions de votre code et des pointeurs vers vos données.
- DVC gère les versions de vos données et modèles et les lie à votre dépôt Git.
#!/bin/bash

# Script de lancement rapide pour Linux/WSL
echo " Lancement du projet MLOps German Credit"
echo "========================================="

# Vérifier si on est dans le bon répertoire
if [ ! -f "params.yaml" ]; then
    echo " Veuillez exécuter ce script depuis le répertoire racine du projet"
    exit 1
fi

# Activer l'environnement virtuel si disponible
if [ -d ".venv" ]; then
    echo " Activation de l'environnement virtuel..."
    source .venv/bin/activate
fi

# Installer les dépendances
echo " Installation des dépendances..."
pip install -r requirements.txt

# Exécuter le pipeline complet
echo " Exécution du pipeline ML..."
python3 launch_project.py

echo "Script terminé"

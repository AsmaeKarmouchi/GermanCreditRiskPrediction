@echo off
REM Script de lancement rapide pour Windows

echo  Lancement du projet MLOps German Credit
echo =========================================

REM Vérifier si on est dans le bon répertoire
if not exist "params.yaml" (
    echo  Veuillez exécuter ce script depuis le répertoire racine du projet
    pause
    exit /b 1
)

REM Activer l'environnement virtuel si disponible
if exist ".venv\Scripts\activate.bat" (
    echo 🔧 Activation de l'environnement virtuel...
    call .venv\Scripts\activate.bat
)

REM Installer les dépendances
echo  Installation des dépendances...
pip install -r requirements.txt

REM Exécuter le pipeline complet
echo  Exécution du pipeline ML...
python launch_project.py

echo Script terminé
pause

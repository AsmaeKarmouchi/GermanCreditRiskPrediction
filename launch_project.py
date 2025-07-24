#!/usr/bin/env python3
"""
Script de lancement complet du pipeline MLOps
"""
import subprocess
import sys
import time
import threading
from pathlib import Path
import webbrowser
from datetime import datetime
import shutil

def get_python_command():#bref j'avais problème avec python3
    """Détecter la commande Python disponible"""
    for cmd in ['python3', 'python']:
        if shutil.which(cmd):
            return cmd
    return 'python3'  # Fallback

def run_command(command, description, background=False):
    """Exécuter une commande avec affichage du statut"""
    print(f"\n {description}...")
    print(f"   Commande: {command}")
    
    if background:
        # Lancer en arrière-plan
        process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    else:
        # Lancer et attendre la fin
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"    {description} - Terminé avec succès")
            if result.stdout:
                print(f"    Sortie: {result.stdout[:200]}...")
        else:
            print(f"    {description} - Erreur")
            if result.stderr:
                print(f"    Erreur: {result.stderr}")
        
        return result

def check_dependencies():
    """Vérifier et installer les dépendances"""
    print("🔍 Vérification des dépendances...")
    
    try:
        import streamlit
        import plotly
        import flask
        import mlflow
        print(" Toutes les dépendances principales sont installées")
        return True
    except ImportError as e:
        print(f" Dépendance manquante: {e}")
        print(" Installation des dépendances...")
        
        result = run_command(
            "pip install -r requirements.txt",
            "Installation des dépendances"
        )
        
        return result.returncode == 0

def run_ml_pipeline():
    """Exécuter le pipeline ML complet"""
    print("\n" + "="*60)
    print(" LANCEMENT DU PIPELINE MLOPS")
    print("="*60)
    
    python_cmd = get_python_command()
    
    # Étape 1: Feature Engineering
    result1 = run_command(
        f"{python_cmd} src/features/advanced_features.py",
        "Feature Engineering Avancé"
    )
    
    if result1.returncode != 0:
        print(" Attention: Erreur lors du feature engineering")
        # Continuer quand même si les données sont déjà traitées
    
    # Étape 2: Entraînement des modèles
    result2 = run_command(
        f"{python_cmd} src/models/advanced_training.py",
        "Entraînement des Modèles Avancés"
    )
    
    if result2.returncode != 0:
        print(" Attention: Erreur lors de l'entraînement")
    
    # Étape 3: Évaluation
    result3 = run_command(
        f"{python_cmd} src/models/advanced_evaluation.py",
        "Évaluation des Modèles"
    )
    
    if result3.returncode != 0:
        print("Attention: Erreur lors de l'évaluation")
    
    return all(r.returncode == 0 for r in [result1, result2, result3])

def launch_services():
    """Lancer les services (API et Dashboard)"""
    print("\n" + "="*60)
    print(" LANCEMENT DES SERVICES")
    print("="*60)
    
    python_cmd = get_python_command()
    
    # Lancer l'API en arrière-plan
    print("\n Lancement de l'API de prédiction...")
    api_process = run_command(
        f"{python_cmd} api/prediction_api.py",
        "API de Prédiction",
        background=True
    )
    
    # Attendre un peu pour que l'API démarre
    time.sleep(3)
    
    # Lancer le dashboard
    print("\n Lancement du dashboard Streamlit...")
    dashboard_process = run_command(
        "streamlit run dashboard/ml_dashboard.py --server.port 8501 --server.headless true",
        "Dashboard Streamlit",
        background=True
    )
    
    return api_process, dashboard_process

def open_browser_tabs():
    """Ouvrir les onglets du navigateur"""
    print("\n Ouverture des interfaces web...")
    
    time.sleep(5)  # Attendre que les services démarrent
    
    try:
        # Ouvrir le dashboard
        webbrowser.open('http://localhost:8501')
        print(" Dashboard ouvert: http://localhost:8501")
        
        # Ouvrir l'API docs (health check)
        webbrowser.open('http://localhost:5000/health')
        print(" API health check ouvert: http://localhost:5000/health")
        
    except Exception as e:
        print(f" Impossible d'ouvrir automatiquement le navigateur: {e}")
        print("\n Ouvrez manuellement ces URLs:")
        print("   - Dashboard: http://localhost:8501")
        print("   - API Health: http://localhost:5000/health")

def show_usage_instructions():
    """Afficher les instructions d'utilisation"""
    print("\n" + "="*60)
    print("GUIDE D'UTILISATION")
    print("="*60)
    
    print("\n DASHBOARD STREAMLIT (http://localhost:8501)")
    print("   • Vue d'ensemble: Métriques principales et alertes")
    print("   • Performance Modèle: Évaluation détaillée")
    print("   • Monitoring Temps Réel: Suivi en direct")
    print("   • Analyse des Données: Upload et analyse de fichiers")
    print("   • Configuration: Paramètres du système")
    
    print("\n API DE PRÉDICTION (http://localhost:5000)")
    print("   • /health - Vérification de santé")
    print("   • /predict - Prédiction unique")
    print("   • /predict/batch - Prédictions en lot")
    print("   • /monitoring/stats - Statistiques")
    print("   • /model/info - Informations du modèle")
    
    print("\n EXEMPLES D'UTILISATION DE L'API:")
    print("   # Test de santé")
    print("   curl http://localhost:5000/health")
    print("")
    print("   # Prédiction")
    print('   curl -X POST http://localhost:5000/predict \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"duration": 24, "amount": 5000, "age": 35}\'')
    
    print("\n FICHIERS GÉNÉRÉS:")
    print("   • reports/figures/ - Graphiques d'évaluation")
    print("   • reports/metrics/ - Métriques JSON")
    print("   • models/best_model.joblib - Meilleur modèle")
    print("   • logs/ - Logs d'exécution")

def main():
    """Fonction principale"""
    print(" PIPELINE MLOPS - GERMAN CREDIT RISK PREDICTION")
    print(f" Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Vérifier les dépendances
    if not check_dependencies():
        print(" Impossible de continuer sans les dépendances")
        sys.exit(1)
    
    # Demander confirmation pour le pipeline complet
    response = input("\n Voulez-vous exécuter le pipeline ML complet ? (y/n): ").lower()
    
    if response == 'y':
        # Exécuter le pipeline ML
        pipeline_success = run_ml_pipeline()
        
        if not pipeline_success:
            print(" Le pipeline a rencontré des erreurs, mais on continue...")
    else:
        print(" Pipeline ML ignoré - lancement direct des services")
    
    # Lancer les services
    api_process, dashboard_process = launch_services()
    
    # Ouvrir le navigateur dans un thread séparé
    browser_thread = threading.Thread(target=open_browser_tabs)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Afficher les instructions
    show_usage_instructions()
    
    print("\n" + "="*60)
    print(" SYSTÈME LANCÉ AVEC SUCCÈS")
    print("="*60)
    print(" Les services sont en cours d'exécution...")
    print("  Appuyez sur Ctrl+C pour arrêter tous les services")
    
    try:
        # Attendre indéfiniment
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n Arrêt des services...")
        
        # Arrêter les processus
        if api_process:
            api_process.terminate()
            print(" API arrêtée")
        
        if dashboard_process:
            dashboard_process.terminate()
            print(" Dashboard arrêté")
        
        print(" Pipeline MLOps arrêté proprement")

if __name__ == "__main__":
    main()

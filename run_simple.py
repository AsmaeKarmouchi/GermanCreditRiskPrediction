#!/usr/bin/env python3
"""
Script de lancement simple du pipeline MLOps
Exécute directement les modules Python sans subprocess
"""
import sys
import os
from pathlib import Path
import time
import webbrowser
import threading

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def run_pipeline_step(module_path, description):
    """Exécuter une étape du pipeline"""
    print(f"\n🔄 {description}...")
    print(f"   Module: {module_path}")
    
    try:
        # Changer le répertoire de travail
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Importer et exécuter le module
        if module_path == "src.features.advanced_features":
            from src.features.advanced_features import advanced_feature_engineering
            from config.config import config
            advanced_feature_engineering(
                config.data.raw_data_path,
                config.data.processed_data_path
            )
            
        elif module_path == "src.models.advanced_training":
            from src.models.advanced_training import train_advanced_models
            train_advanced_models()
            
        elif module_path == "src.models.advanced_evaluation":
            from src.models.advanced_evaluation import evaluate_saved_model
            from config.config import config
            model_path = f"{config.model.model_dir}/best_model.joblib"
            if Path(model_path).exists():
                evaluate_saved_model(model_path, config.data.processed_data_path)
            else:
                print("⚠️ Modèle non trouvé, ignoré")
        
        print(f"✅ {description} - Terminé avec succès")
        return True
        
    except Exception as e:
        print(f"❌ {description} - Erreur: {e}")
        return False
    finally:
        # Restaurer le répertoire de travail
        os.chdir(original_cwd)

def check_data_exists():
    """Vérifier si les données existent"""
    raw_data = Path("data/raw/GermanCredit.csv")
    if raw_data.exists():
        print("✅ Données brutes trouvées")
        return True
    else:
        print("📥 Téléchargement des données...")
        try:
            from src.data.download import download_german_credit
            download_german_credit()
            return True
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            return False

def launch_api_service():
    """Lancer l'API en tant que service"""
    try:
        print("🔄 Démarrage de l'API...")
        from api.prediction_api import app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"❌ Erreur API: {e}")

def launch_dashboard_service():
    """Lancer le dashboard"""
    import subprocess
    try:
        print("🔄 Démarrage du dashboard...")
        subprocess.run([
            "streamlit", "run", "dashboard/ml_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
    except Exception as e:
        print(f"❌ Erreur Dashboard: {e}")

def open_browser():
    """Ouvrir les onglets du navigateur"""
    time.sleep(5)
    try:
        webbrowser.open('http://localhost:8501')
        print("✅ Dashboard ouvert: http://localhost:8501")
        time.sleep(2)
        webbrowser.open('http://localhost:5000/health')
        print("✅ API health check ouvert: http://localhost:5000/health")
    except Exception as e:
        print(f"⚠️ Ouverture navigateur échouée: {e}")
        print("📋 Ouvrez manuellement:")
        print("   - Dashboard: http://localhost:8501")
        print("   - API: http://localhost:5000/health")

def main():
    """Fonction principale"""
    print("🎯 PIPELINE MLOPS - GERMAN CREDIT RISK PREDICTION")
    print("=" * 60)
    
    # Vérifier les données
    if not check_data_exists():
        print("❌ Impossible de continuer sans données")
        return
    
    # Demander confirmation
    response = input("\n❓ Exécuter le pipeline ML complet ? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🚀 EXÉCUTION DU PIPELINE ML")
        print("=" * 60)
        
        # Étapes du pipeline
        steps = [
            ("src.features.advanced_features", "Feature Engineering Avancé"),
            ("src.models.advanced_training", "Entraînement des Modèles"),
            ("src.models.advanced_evaluation", "Évaluation des Modèles")
        ]
        
        results = []
        for module_path, description in steps:
            success = run_pipeline_step(module_path, description)
            results.append(success)
        
        if all(results):
            print("\n✅ Pipeline ML terminé avec succès !")
        else:
            print("\n⚠️ Pipeline terminé avec des erreurs")
    
    # Lancer les services
    print("\n🌐 LANCEMENT DES SERVICES")
    print("=" * 60)
    
    choice = input("\nChoisissez une option:\n1. API seulement\n2. Dashboard seulement\n3. Les deux\n4. Aucun\nChoix (1-4): ").strip()
    
    if choice in ['1', '3']:
        # Lancer l'API dans un thread
        api_thread = threading.Thread(target=launch_api_service, daemon=True)
        api_thread.start()
        print("🚀 API démarrée en arrière-plan sur http://localhost:5000")
    
    if choice in ['2', '3']:
        # Lancer le navigateur dans un thread
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Lancer le dashboard (bloquant)
        print("🚀 Lancement du dashboard...")
        launch_dashboard_service()
    
    elif choice == '1':
        # API seulement
        print("🔄 API en cours d'exécution...")
        print("🌐 Accès: http://localhost:5000")
        print("⏹️ Appuyez sur Ctrl+C pour arrêter")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 API arrêtée")
    
    elif choice == '4':
        print("✅ Pipeline terminé sans lancement de services")
        print("\n📋 Pour lancer manuellement:")
        print("   API: python3 api/prediction_api.py")
        print("   Dashboard: streamlit run dashboard/ml_dashboard.py")

if __name__ == "__main__":
    main()

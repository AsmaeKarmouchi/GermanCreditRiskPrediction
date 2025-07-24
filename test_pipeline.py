# Version améliorée avec corrections d'imports
import sys
from pathlib import Path

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.config import config
    from utils.logging_utils import get_logger
except ImportError:
    # Fallback si les modules ne sont pas disponibles
    import logging
    config = None
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

def main():
    """Script principal pour tester le nouveau pipeline"""
    
   
    print("=" * 50)
    
    # Test 1: Configuration
    print("\n1. Test de la configuration...")
    try:
        if config:
            
            print(f"   - Données: {config.data.raw_data_path}")
            print(f"   - Modèles: {config.model.model_dir}")
        else:
            print("onfiguration non disponible (mode fallback)")
    except Exception as e:
        print(f"Erreur configuration: {e}")
    
    # Test 2: Structure des dossiers
    print("\n2. Vérification de la structure...")
    
    required_dirs = [
        "config", "utils", "api", "dashboard", "src/features", 
        "src/models", "tests", "reports/figures", "reports/metrics"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f" {dir_path}")
        else:
            print(f" {dir_path} - manquant")
    
    # Test 3: Fichiers clés
    print("\n3. Vérification des fichiers clés...")
    
    key_files = [
        "config/config.py",
        "utils/logging_utils.py", 
        "src/features/advanced_features.py",
        "src/models/advanced_training.py",
        "src/models/advanced_evaluation.py",
        "api/prediction_api.py",
        "dashboard/ml_dashboard.py",
        "tests/test_ml_pipeline.py"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"{file_path}")
        else:
            print(f"{file_path} - manquant")
    
    # Test 4: Test d'imports
    print("\n4. Test des imports principaux...")
    
    try:
        from src.features.advanced_features import AdvancedPreprocessor
        print(" AdvancedPreprocessor")
    except ImportError as e:
        print(f" AdvancedPreprocessor: {e}")
    
    try:
        from src.models.advanced_training import AdvancedModelTrainer
        print(" AdvancedModelTrainer")
    except ImportError as e:
        print(f" AdvancedModelTrainer: {e}")
    
    try:
        from src.models.advanced_evaluation import ModelEvaluator
        print(" ModelEvaluator")
    except ImportError as e:
        print(f" ModelEvaluator: {e}")
    
    # Recommandations
    print("\n" + "=" * 50)
    print(" PROCHAINES ÉTAPES RECOMMANDÉES:")
    print("=" * 50)
    print("1. Installer les nouvelles dépendances:")
    print("   pip install -r requirements.txt")
    print("\n2. Tester le pipeline feature engineering:")
    print("   python src/features/advanced_features.py")
    print("\n3. Entraîner les modèles avancés:")
    print("   python src/models/advanced_training.py")
    print("\n4. Lancer l'évaluation:")
    print("   python src/models/advanced_evaluation.py")
    print("\n5. Tester l'API:")
    print("   python api/prediction_api.py")
    print("\n6. Lancer le dashboard:")
    print("   streamlit run dashboard/ml_dashboard.py")
    print("\n7. Exécuter les tests:")
    print("   python tests/test_ml_pipeline.py")
    
    print("\n AMÉLIORATIONS APPORTÉES:")
    print("• Feature engineering automatisé avec création de variables métier")
    print("• Ensemble learning avec 4+ algorithmes et sélection automatique")
    print("• Hyperparameter tuning avec RandomizedSearchCV")
    print("• Métriques métier (coût, profit, seuil optimal)")
    print("• API RESTful avec monitoring en temps réel")
    print("• Dashboard interactif pour le monitoring")
    print("• Tests complets et pipeline reproductible")
    print("• Configuration centralisée et logging structuré")
    print("• Architecture modulaire prête pour la production")

if __name__ == "__main__":
    main()

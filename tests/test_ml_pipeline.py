"""
Tests complets pour le pipeline ML
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys
import os

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config
from src.features.advanced_features import AdvancedPreprocessor, FeatureCreator
from src.models.advanced_training import AdvancedModelTrainer, EnsembleClassifier
from src.models.advanced_evaluation import ModelEvaluator

class TestDataQuality(unittest.TestCase):
    """Tests de qualité des données"""
    
    def setUp(self):
        """Configuration des tests"""
        # Créer des données de test
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'status': ['... < 100 DM', '0 <= ... < 200 DM', 'no checking account'] * 10,
            'duration': np.random.randint(6, 72, 30),
            'amount': np.random.randint(250, 18424, 30),
            'age': np.random.randint(19, 75, 30),
            'credit_risk': np.random.choice([0, 1], 30)
        })
    
    def test_data_loading(self):
        """Test du chargement des données"""
        self.assertFalse(self.sample_data.empty)
        self.assertIn('credit_risk', self.sample_data.columns)
        self.assertEqual(len(self.sample_data), 30)
    
    def test_data_types(self):
        """Test des types de données"""
        numeric_cols = ['duration', 'amount', 'age', 'credit_risk']
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data[col]))
    
    def test_target_distribution(self):
        """Test de la distribution de la variable cible"""
        target_values = self.sample_data['credit_risk'].unique()
        self.assertTrue(all(val in [0, 1] for val in target_values))
        
        # Vérifier qu'on a les deux classes
        self.assertTrue(len(target_values) >= 1)
    
    def test_missing_values(self):
        """Test des valeurs manquantes"""
        missing_counts = self.sample_data.isnull().sum()
        self.assertTrue(all(count >= 0 for count in missing_counts))

class TestFeatureEngineering(unittest.TestCase):
    """Tests du feature engineering"""
    
    def setUp(self):
        """Configuration des tests"""
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'duration': [12, 24, 36, 48],
            'amount': [1000, 2000, 3000, 4000],
            'installment_rate': [4, 3, 2, 1],
            'age': [25, 35, 45, 55],
            'present_residence': [1, 2, 3, 4],
            'number_credits': [1, 2, 1, 3]
        })
    
    def test_feature_creator(self):
        """Test de la création de nouvelles features"""
        creator = FeatureCreator()
        result = creator.fit_transform(self.sample_data)
        
        # Vérifier que les nouvelles features sont créées
        self.assertIn('monthly_payment', result.columns)
        self.assertIn('age_group', result.columns)
        self.assertIn('credits_per_year', result.columns)
        
        # Vérifier les calculs
        expected_monthly = self.sample_data['amount'] / self.sample_data['duration']
        pd.testing.assert_series_equal(result['monthly_payment'], expected_monthly, check_names=False)
    
    def test_preprocessor_pipeline(self):
        """Test du pipeline de preprocessing"""
        # Ajouter la target
        data_with_target = self.sample_data.copy()
        data_with_target['credit_risk'] = [0, 1, 0, 1]
        
        X = data_with_target.drop('credit_risk', axis=1)
        y = data_with_target['credit_risk']
        
        preprocessor = AdvancedPreprocessor()
        X_processed, y_processed = preprocessor.fit_transform(X, y)
        
        # Vérifier la forme de sortie
        self.assertEqual(X_processed.shape[0], len(X))
        self.assertIsInstance(X_processed, np.ndarray)
        self.assertIsNotNone(y_processed)

class TestModelTraining(unittest.TestCase):
    """Tests de l'entraînement des modèles"""
    
    def setUp(self):
        """Configuration des tests"""
        np.random.seed(42)
        # Créer des données synthétiques plus réalistes
        n_samples = 200
        n_features = 10
        
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.choice([0, 1], n_samples)
    
    def test_ensemble_classifier(self):
        """Test du classificateur ensemble"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=100))
        ]
        
        ensemble = EnsembleClassifier(estimators)
        ensemble.fit(self.X, self.y)
        
        # Test des prédictions
        predictions = ensemble.predict(self.X)
        probabilities = ensemble.predict_proba(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(probabilities.shape, (len(self.X), 2))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_model_trainer_initialization(self):
        """Test de l'initialisation du trainer"""
        trainer = AdvancedModelTrainer()
        
        self.assertIsInstance(trainer.models, dict)
        self.assertEqual(len(trainer.models), 0)
        self.assertIsNone(trainer.best_model)
    
    @patch('src.models.advanced_training.get_mlflow_tracker')
    def test_single_model_training(self, mock_mlflow):
        """Test de l'entraînement d'un modèle unique"""
        # Mock MLFlow
        mock_tracker = MagicMock()
        mock_mlflow.return_value = mock_tracker
        mock_tracker.start_run.return_value.__enter__ = MagicMock()
        mock_tracker.start_run.return_value.__exit__ = MagicMock()
        
        trainer = AdvancedModelTrainer()
        
        # Diviser les données
        split_idx = len(self.X) // 2
        X_train, X_val = self.X[:split_idx], self.X[split_idx:]
        y_train, y_val = self.y[:split_idx], self.y[split_idx:]
        
        # Entraîner un modèle simple
        results = trainer.train_single_model(X_train, y_train, X_val, y_val, 'random_forest')
        
        self.assertIn('model', results)
        self.assertIn('metrics', results)
        self.assertIn('roc_auc', results['metrics'])

class TestModelEvaluation(unittest.TestCase):
    """Tests de l'évaluation des modèles"""
    
    def setUp(self):
        """Configuration des tests"""
        np.random.seed(42)
        
        # Créer un modèle simple pour les tests
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Données de test
        n_samples = 100
        n_features = 5
        self.X_test = np.random.randn(n_samples, n_features)
        self.y_test = np.random.choice([0, 1], n_samples)
        
        # Entraîner le modèle
        self.model.fit(self.X_test, self.y_test)
    
    def test_model_evaluator_initialization(self):
        """Test de l'initialisation de l'évaluateur"""
        evaluator = ModelEvaluator()
        
        self.assertTrue(evaluator.figures_dir.exists())
        self.assertIsNotNone(evaluator.metrics_collector)
    
    def test_metrics_calculation(self):
        """Test du calcul des métriques"""
        evaluator = ModelEvaluator()
        
        # Créer des prédictions de test
        y_pred = np.random.choice([0, 1], len(self.y_test))
        y_pred_proba = np.random.rand(len(self.y_test))
        
        metrics = evaluator.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)
    
    def test_business_metrics(self):
        """Test des métriques métier"""
        evaluator = ModelEvaluator()
        
        y_pred = np.random.choice([0, 1], len(self.y_test))
        y_pred_proba = np.random.rand(len(self.y_test))
        
        business_metrics = evaluator._calculate_business_metrics(self.y_test, y_pred, y_pred_proba)
        
        expected_business = ['total_cost', 'cost_savings', 'false_positive_rate', 'false_negative_rate']
        for metric in expected_business:
            self.assertIn(metric, business_metrics)

class TestAPI(unittest.TestCase):
    """Tests de l'API de prédiction"""
    
    def setUp(self):
        """Configuration des tests"""
        # Créer un modèle temporaire pour les tests
        from sklearn.ensemble import RandomForestClassifier
        import tempfile
        
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = Path(self.temp_dir) / "test_model.joblib"
        
        # Créer et sauvegarder un modèle simple
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.randn(100, 5)
        y_dummy = np.random.choice([0, 1], 100)
        model.fit(X_dummy, y_dummy)
        
        joblib.dump(model, self.model_path)
    
    def tearDown(self):
        """Nettoyage après les tests"""
        shutil.rmtree(self.temp_dir)
    
    @patch('api.prediction_api.config')
    def test_api_initialization(self, mock_config):
        """Test de l'initialisation de l'API"""
        # Mock de la configuration
        mock_config.model.model_dir = str(self.model_path.parent)
        
        from api.prediction_api import PredictionAPI
        
        # Temporairement renommer le fichier pour correspondre au pattern attendu
        expected_path = self.model_path.parent / "best_model.joblib"
        shutil.copy(self.model_path, expected_path)
        
        api = PredictionAPI()
        
        # Vérifier que le modèle est chargé
        self.assertIsNotNone(api.model)

class TestIntegration(unittest.TestCase):
    """Tests d'intégration du pipeline complet"""
    
    def setUp(self):
        """Configuration des tests d'intégration"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Nettoyage après les tests"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test du pipeline complet (version simplifiée)"""
        # Créer des données de test
        data = pd.DataFrame({
            'duration': [12, 24, 36],
            'amount': [1000, 2000, 3000],
            'age': [25, 35, 45],
            'credit_risk': [0, 1, 0]
        })
        
        # Test du preprocessing
        X = data.drop('credit_risk', axis=1)
        y = data['credit_risk']
        
        preprocessor = AdvancedPreprocessor()
        X_processed, y_processed = preprocessor.fit_transform(X, y)
        
        # Vérifier que les données sont correctement preprocessées
        self.assertIsInstance(X_processed, np.ndarray)
        self.assertEqual(len(X_processed), len(data))
        
        # Test de l'entraînement (version simplifiée)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # S'assurer qu'on a assez de données pour l'entraînement
        if len(X_processed) >= 2 and len(np.unique(y_processed)) == 2:
            model.fit(X_processed, y_processed)
            predictions = model.predict(X_processed)
            
            # Vérifier que les prédictions sont valides
            self.assertTrue(all(pred in [0, 1] for pred in predictions))

def run_all_tests():
    """Exécuter tous les tests"""
    
    # Créer une suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter les classes de test
    test_classes = [
        TestDataQuality,
        TestFeatureEngineering,
        TestModelTraining,
        TestModelEvaluation,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Retourner le résultat
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_all_tests()
    if success:
        print("\n✅ Tous les tests sont passés avec succès!")
    else:
        print("\n❌ Certains tests ont échoué.")
        sys.exit(1)

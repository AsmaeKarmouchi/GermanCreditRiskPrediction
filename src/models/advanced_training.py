"""
Modèles ML avancés avec hyperparameter tuning et validation croisée
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import config
from utils.logging_utils import get_logger, get_mlflow_tracker, get_metrics_collector

logger = get_logger(__name__)

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Classificateur ensemble personnalisé"""
    
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], voting='soft'):
        self.estimators = estimators
        self.voting = voting
        self.fitted_estimators_ = []
        self.classes_ = None
    
    def fit(self, X, y):
        """Entraîner tous les estimateurs"""
        self.classes_ = np.unique(y)
        self.fitted_estimators_ = []
        
        for name, estimator in self.estimators:
            logger.info(f"Entraînement de {name}")
            fitted_est = estimator.fit(X, y)
            self.fitted_estimators_.append((name, fitted_est))
        
        return self
    
    def predict_proba(self, X):
        """Prédiction des probabilités par vote"""
        if self.voting == 'soft':
            probas = np.array([est.predict_proba(X) for _, est in self.fitted_estimators_])
            return np.mean(probas, axis=0)
        else:
            raise NotImplementedError("Voting 'hard' non implémenté pour predict_proba")
    
    def predict(self, X):
        """Prédiction par vote majoritaire"""
        if self.voting == 'soft':
            probas = self.predict_proba(X)
            return self.classes_[np.argmax(probas, axis=1)]
        else:
            predictions = np.array([est.predict(X) for _, est in self.fitted_estimators_])
            return np.array([np.bincount(predictions[:, i]).argmax() for i in range(X.shape[0])])

class AdvancedModelTrainer:
    """Entraîneur de modèles avec tuning automatique"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.mlflow_tracker = get_mlflow_tracker()
        self.metrics_collector = get_metrics_collector()
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Configuration des modèles et hyperparamètres"""
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=config.model.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=config.model.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=config.model.random_state, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'svm': {
                'model': SVC(random_state=config.model.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
    
    def train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          model_name: str) -> Dict[str, Any]:
        """Entraîner un modèle unique avec tuning"""
        
        logger.info(f"Entraînement du modèle: {model_name}")
        
        config_models = self.get_model_configs()
        model_config = config_models[model_name]
        
        # Grid Search avec validation croisée
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.model.random_state)
        
        grid_search = RandomizedSearchCV(
            estimator=model_config['model'],
            param_distributions=model_config['params'],
            n_iter=20,  # Limite pour éviter de trop longs temps d'exécution
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=config.model.random_state
        )
        
        # Entraînement
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Calibration des probabilités
        calibrated_model = CalibratedClassifierCV(best_model, cv=3)
        calibrated_model.fit(X_train, y_train)
        
        # Évaluation
        train_score = calibrated_model.score(X_train, y_train)
        val_score = calibrated_model.score(X_val, y_val)
        
        y_pred = calibrated_model.predict(X_val)
        y_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
        
        metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Logging MLFlow
        with self.mlflow_tracker.start_run(run_name=f"{model_name}_tuning"):
            self.mlflow_tracker.log_params(grid_search.best_params_)
            self.mlflow_tracker.log_metrics({
                'train_accuracy': train_score,
                'val_accuracy': val_score,
                **metrics
            })
            self.mlflow_tracker.log_model(calibrated_model, model_name)
        
        results = {
            'model': calibrated_model,
            'best_params': grid_search.best_params_,
            'train_score': train_score,
            'val_score': val_score,
            'metrics': metrics,
            'cv_score': grid_search.best_score_
        }
        
        self.models[model_name] = results
        
        logger.info(f"{model_name} - Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculer les métriques complètes"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Entraîner un ensemble des meilleurs modèles"""
        
        logger.info("Entraînement de l'ensemble")
        
        # Sélectionner les 3 meilleurs modèles
        top_models = sorted(self.models.items(), 
                           key=lambda x: x[1]['metrics']['roc_auc'], 
                           reverse=True)[:3]
        
        estimators = [(name, results['model']) for name, results in top_models]
        
        # Créer l'ensemble
        ensemble = EnsembleClassifier(estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Évaluation
        y_pred = ensemble.predict(X_val)
        y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
        
        metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Logging MLFlow
        with self.mlflow_tracker.start_run(run_name="ensemble_model"):
            ensemble_params = {f"model_{i}": name for i, (name, _) in enumerate(estimators)}
            self.mlflow_tracker.log_params(ensemble_params)
            self.mlflow_tracker.log_metrics(metrics)
            self.mlflow_tracker.log_model(ensemble, "ensemble")
        
        results = {
            'model': ensemble,
            'metrics': metrics,
            'component_models': [name for name, _ in estimators]
        }
        
        self.models['ensemble'] = results
        
        logger.info(f"Ensemble - Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> str:
        """Entraîner tous les modèles et retourner le meilleur"""
        
        logger.info("Début de l'entraînement de tous les modèles")
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.model.validation_size,
            random_state=config.model.random_state, stratify=y
        )
        
        logger.info(f"Taille train: {X_train.shape}, Taille validation: {X_val.shape}")
        
        # Entraîner tous les modèles individuels
        model_configs = self.get_model_configs()
        for model_name in model_configs.keys():
            try:
                self.train_single_model(X_train, y_train, X_val, y_val, model_name)
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {model_name}: {e}")
        
        # Entraîner l'ensemble si on a au moins 2 modèles
        if len(self.models) >= 2:
            self.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Sélectionner le meilleur modèle
        best_model_name = max(self.models.keys(), 
                             key=lambda x: self.models[x]['metrics']['roc_auc'])
        
        self.best_model = self.models[best_model_name]['model']
        self.best_score = self.models[best_model_name]['metrics']['roc_auc']
        
        logger.info(f"Meilleur modèle: {best_model_name} (ROC-AUC: {self.best_score:.4f})")
        
        # Sauvegarder les métriques
        all_metrics = {name: results['metrics'] for name, results in self.models.items()}
        self.metrics_collector.save_metrics(all_metrics, "model_comparison")
        
        return best_model_name
    
    def save_best_model(self, filepath: str):
        """Sauvegarder le meilleur modèle"""
        if self.best_model is None:
            raise ValueError("Aucun modèle entraîné")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, filepath)
        logger.info(f"Meilleur modèle sauvegardé: {filepath}")
    
    def generate_model_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """Générer un rapport complet des modèles"""
        
        report_path = "reports/model_evaluation_report.txt"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT D'ÉVALUATION DES MODÈLES ===\n\n")
            
            for model_name, results in self.models.items():
                f.write(f"\n--- {model_name.upper()} ---\n")
                f.write(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}\n")
                f.write(f"Accuracy: {results['metrics']['accuracy']:.4f}\n")
                f.write(f"F1-Score: {results['metrics']['f1_score']:.4f}\n")
                f.write(f"Precision: {results['metrics']['precision']:.4f}\n")
                f.write(f"Recall: {results['metrics']['recall']:.4f}\n")
                
                if 'best_params' in results:
                    f.write(f"Meilleurs paramètres: {results['best_params']}\n")
                
                # Test sur les données de test
                if self.models[model_name]['model'] is not None:
                    y_pred = results['model'].predict(X_test)
                    test_accuracy = accuracy_score(y_test, y_pred)
                    f.write(f"Accuracy sur test: {test_accuracy:.4f}\n")
        
        logger.info(f"Rapport généré: {report_path}")
        return report_path

def train_advanced_models() -> str:
    """Pipeline principal d'entraînement des modèles"""
    
    logger.info("Début de l'entraînement des modèles avancés")
    
    # Charger les données preprocessées
    df = pd.read_csv(config.data.processed_data_path)
    
    # Séparer features et target
    target_col = config.data.target_column
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    logger.info(f"Données chargées: {X.shape}")
    logger.info(f"Distribution des classes: {np.bincount(y)}")
    
    # Split train/test final
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.test_size,
        random_state=config.model.random_state, stratify=y
    )
    
    # Entraîner les modèles
    trainer = AdvancedModelTrainer()
    best_model_name = trainer.train_all_models(X_train, y_train)
    
    # Sauvegarder le meilleur modèle
    model_path = f"{config.model.model_dir}/best_model.joblib"
    trainer.save_best_model(model_path)
    
    # Générer le rapport
    trainer.generate_model_report(X_test, y_test)
    
    logger.info("Entraînement terminé")
    
    return best_model_name

if __name__ == "__main__":
    train_advanced_models()

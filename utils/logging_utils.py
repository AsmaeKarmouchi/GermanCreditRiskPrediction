"""
Utilitaires pour le logging et le monitoring
"""
import logging
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import pandas as pd
from config.config import config

class Logger:
    """Gestionnaire de logs centralisé"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.logging.log_level))
        
        # Éviter la duplication des handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Configurer les handlers de logging"""
        formatter = logging.Formatter(config.logging.log_format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = Path(config.logging.log_dir) / f"{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

class MLFlowTracker:
    """Gestionnaire pour MLFlow tracking"""
    
    def __init__(self):
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        self.experiment_name = config.mlflow.experiment_name
        self._setup_experiment()
    
    def _setup_experiment(self):
        """Configurer l'expérience MLFlow"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(
                    name=self.experiment_name,
                    artifact_location=config.mlflow.artifact_location
                )
        except Exception as e:
            print(f"Erreur lors de la configuration MLFlow: {e}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Démarrer une nouvelle run MLFlow"""
        mlflow.set_experiment(self.experiment_name)
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Logger les paramètres"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Logger les métriques"""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, model_name: str):
        """Logger le modèle"""
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifacts(self, artifacts_path: str):
        """Logger les artefacts"""
        mlflow.log_artifacts(artifacts_path)

class MetricsCollector:
    """Collecteur de métriques pour le monitoring"""
    
    def __init__(self):
        self.metrics_dir = Path("reports/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str):
        """Sauvegarder les métriques dans un fichier JSON"""
        timestamp = datetime.now().isoformat()
        metrics_with_timestamp = {
            "timestamp": timestamp,
            "metrics": metrics
        }
        
        filepath = self.metrics_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(metrics_with_timestamp, f, indent=2)
    
    def load_latest_metrics(self, pattern: str) -> Optional[Dict[str, Any]]:
        """Charger les dernières métriques correspondant au pattern"""
        files = list(self.metrics_dir.glob(f"{pattern}_*.json"))
        if not files:
            return None
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def compare_metrics(self, current_metrics: Dict[str, float], 
                       baseline_pattern: str) -> Dict[str, Dict[str, float]]:
        """Comparer les métriques actuelles avec la baseline"""
        baseline = self.load_latest_metrics(baseline_pattern)
        if not baseline:
            return {"comparison": "No baseline found"}
        
        baseline_metrics = baseline.get("metrics", {})
        comparison = {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline_metrics:
                baseline_value = baseline_metrics[metric_name]
                difference = current_value - baseline_value
                percentage_change = (difference / baseline_value) * 100 if baseline_value != 0 else 0
                
                comparison[metric_name] = {
                    "current": current_value,
                    "baseline": baseline_value,
                    "difference": difference,
                    "percentage_change": percentage_change
                }
        
        return comparison

def get_logger(name: str) -> logging.Logger:
    """Factory function pour obtenir un logger"""
    return Logger(name).logger

def get_mlflow_tracker() -> MLFlowTracker:
    """Factory function pour obtenir un tracker MLFlow"""
    return MLFlowTracker()

def get_metrics_collector() -> MetricsCollector:
    """Factory function pour obtenir un collecteur de métriques"""
    return MetricsCollector()

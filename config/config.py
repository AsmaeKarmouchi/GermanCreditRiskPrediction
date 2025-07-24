"""
Configuration centralisée pour le projet MLOps GermanCredit
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class DataConfig:
    """Configuration des données"""
    raw_data_path: str = "data/raw/GermanCredit.csv"
    processed_data_path: str = "data/processed/GermanCredit_processed.csv"
    feature_store_path: str = "data/features/"
    target_column: str = "credit_risk"
    
@dataclass
class ModelConfig:
    """Configuration des modèles"""
    model_dir: str = "models/"
    experiment_dir: str = "experiments/"
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    
@dataclass
class LoggingConfig:
    """Configuration des logs"""
    log_dir: str = "logs/"
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class MLFlowConfig:
    """Configuration MLFlow"""
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "german_credit_risk"
    artifact_location: str = "artifacts/"

class Config:
    """Configuration principale"""
    
    def __init__(self, config_path: str = "params.yaml"):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / config_path
        
        # Charger la configuration depuis params.yaml
        self._load_config()
        
        # Initialiser les sous-configurations
        self.data = DataConfig()
        self.model = ModelConfig()
        self.logging = LoggingConfig()
        self.mlflow = MLFlowConfig()
        
        # Créer les répertoires nécessaires
        self._create_directories()
    
    def _load_config(self):
        """Charger la configuration depuis le fichier YAML"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.params = yaml.safe_load(f)
        else:
            self.params = {}
    
    def _create_directories(self):
        """Créer les répertoires nécessaires"""
        dirs_to_create = [
            self.data.feature_store_path,
            self.model.model_dir,
            self.model.experiment_dir,
            self.logging.log_dir,
            self.mlflow.artifact_location,
            "reports/figures",
            "reports/metrics"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_model_params(self) -> Dict[str, Any]:
        """Récupérer les paramètres du modèle"""
        return self.params.get('train', {})
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """Récupérer les paramètres de préprocessing"""
        return self.params.get('preprocessing', {})

# Instance globale de configuration
config = Config()

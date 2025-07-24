"""
Feature Engineering et Data Preprocessing avancés
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Tuple, Optional
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config.config import config
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class OutlierHandler(BaseEstimator, TransformerMixin):
    """Gestionnaire d'outliers avec plusieurs méthodes"""
    
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        """Calculer les bornes pour chaque colonne numérique"""
        X = pd.DataFrame(X)
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.bounds_[col] = {
                    'lower': Q1 - self.threshold * IQR,
                    'upper': Q3 + self.threshold * IQR
                }
            elif self.method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                self.bounds_[col] = {
                    'lower': mean - self.threshold * std,
                    'upper': mean + self.threshold * std
                }
        return self
    
    def transform(self, X):
        """Transformer les outliers"""
        X = pd.DataFrame(X).copy()
        for col, bounds in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=bounds['lower'], upper=bounds['upper'])
        return X

class FeatureCreator(BaseEstimator, TransformerMixin):
    """Créateur de nouvelles features métier"""
    
    def __init__(self):
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Créer de nouvelles features"""
        X = pd.DataFrame(X).copy()
        
        # Features financières
        if 'amount' in X.columns and 'duration' in X.columns:
            X['monthly_payment'] = X['amount'] / X['duration']
            X['amount_to_duration_ratio'] = X['amount'] / (X['duration'] + 1)
        
        if 'amount' in X.columns and 'installment_rate' in X.columns:
            X['total_installment_cost'] = X['amount'] * X['installment_rate'] / 100
        
        # Features démographiques
        if 'age' in X.columns:
            X['age_group'] = pd.cut(X['age'], bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['young', 'young_adult', 'adult', 'middle_aged', 'senior'])
            X['age_group'] = X['age_group'].astype(str)
        
        # Features de risque
        if 'number_credits' in X.columns and 'age' in X.columns:
            X['credits_per_year'] = X['number_credits'] / (X['age'] - 18 + 1)
        
        if 'present_residence' in X.columns:
            X['residence_stability'] = X['present_residence'] >= 3
        
        return X

class AdvancedPreprocessor:
    """Préprocesseur avancé avec pipeline complète"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names_ = None
        self.target_encoder = LabelEncoder()
        
    def create_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Créer le pipeline de préprocessing"""
        
        # Identifier les types de colonnes
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        logger.info(f"Features numériques: {len(numeric_features)}")
        logger.info(f"Features catégorielles: {len(categorical_features)}")
        
        # Pipeline pour les features numériques
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_handler', OutlierHandler(method='iqr')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline pour les features catégorielles
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', LabelEncoder())  # Sera remplacé par une version compatible
        ])
        
        # Créer le transformeur composé
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
        
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Ajuster et transformer les données"""
        
        # Créer les nouvelles features
        feature_creator = FeatureCreator()
        X_enhanced = feature_creator.fit_transform(X)
        
        # Créer et ajuster le preprocessor
        self.preprocessor = self.create_pipeline(X_enhanced)
        
        # Gérer l'encodage des catégories manuellement pour éviter les problèmes avec ColumnTransformer
        X_processed = self._manual_preprocessing(X_enhanced)
        
        # Encoder la target si fournie
        y_processed = None
        if y is not None:
            y_processed = self.target_encoder.fit_transform(y)
        
        # Sauvegarder les noms des features
        self.feature_names_ = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        logger.info(f"Données préprocessées: {X_processed.shape}")
        
        return X_processed, y_processed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transformer de nouvelles données"""
        if self.preprocessor is None:
            raise ValueError("Le preprocessor doit d'abord être ajusté avec fit_transform()")
        
        # Créer les nouvelles features
        feature_creator = FeatureCreator()
        X_enhanced = feature_creator.transform(X)
        
        # Appliquer le preprocessing
        X_processed = self._manual_preprocessing(X_enhanced)
        
        return X_processed
    
    def _manual_preprocessing(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocessing manuel pour gérer les catégories"""
        X_copy = X.copy()
        
        # Traitement des valeurs manquantes
        numeric_cols = X_copy.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_copy.select_dtypes(include=['object', 'bool']).columns
        
        # Imputation numérique
        for col in numeric_cols:
            X_copy[col].fillna(X_copy[col].median(), inplace=True)
        
        # Imputation catégorielle
        for col in categorical_cols:
            X_copy[col].fillna('missing', inplace=True)
        
        # Encodage des variables catégorielles avec get_dummies
        X_encoded = pd.get_dummies(X_copy, drop_first=True)
        
        # Gestion des outliers pour les colonnes numériques
        outlier_handler = OutlierHandler()
        numeric_data = X_encoded.select_dtypes(include=['int64', 'float64'])
        if len(numeric_data.columns) > 0:
            numeric_processed = outlier_handler.fit_transform(numeric_data)
            
            # Standardisation
            scaler = StandardScaler()
            numeric_processed = scaler.fit_transform(numeric_processed)
            
            # Remplacer les colonnes numériques
            for i, col in enumerate(numeric_data.columns):
                X_encoded[col] = numeric_processed[:, i]
        
        return X_encoded.values
    
    def save(self, filepath: str):
        """Sauvegarder le preprocessor"""
        joblib.dump({
            'preprocessor': self.preprocessor,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names_
        }, filepath)
        logger.info(f"Preprocessor sauvegardé: {filepath}")
    
    def load(self, filepath: str):
        """Charger le preprocessor"""
        data = joblib.load(filepath)
        self.preprocessor = data['preprocessor']
        self.target_encoder = data['target_encoder']
        self.feature_names_ = data['feature_names']
        logger.info(f"Preprocessor chargé: {filepath}")

def advanced_feature_engineering(input_path: str, output_path: str) -> None:
    """Pipeline complète de feature engineering"""
    logger.info("Début du feature engineering avancé")
    
    # Charger les données
    df = pd.read_csv(input_path)
    logger.info(f"Données chargées: {df.shape}")
    
    # Séparer features et target
    target_col = config.data.target_column
    if target_col not in df.columns:
        raise ValueError(f"Colonne target '{target_col}' non trouvée")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Créer et appliquer le preprocessor
    preprocessor = AdvancedPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(X, y)
    
    # Créer le DataFrame final
    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed[target_col] = y_processed
    
    # Sauvegarder les données traitées
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    # Sauvegarder le preprocessor
    preprocessor_path = str(Path(output_path).parent / "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    
    logger.info(f"Feature engineering terminé: {output_path}")
    logger.info(f"Shape finale: {df_processed.shape}")

if __name__ == "__main__":
    advanced_feature_engineering(
        config.data.raw_data_path,
        config.data.processed_data_path
    )

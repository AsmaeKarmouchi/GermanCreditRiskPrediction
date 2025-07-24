"""
API de prédiction avec monitoring en temps réel
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from config.config import config
from utils.logging_utils import get_logger, get_metrics_collector
from src.models.advanced_evaluation import ModelEvaluator

# Configuration de l'application
app = Flask(__name__)
logger = get_logger(__name__)

class PredictionAPI:
    """API de prédiction avec monitoring"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metrics_collector = get_metrics_collector()
        self.prediction_history = []
        self.load_model()
    
    def load_model(self):
        """Charger le modèle et le preprocessor"""
        try:
            model_path = f"{config.model.model_dir}/best_model.joblib"
            preprocessor_path = "data/processed/preprocessor.joblib"
            
            if Path(model_path).exists():
                self.model = joblib.load(model_path)
                logger.info(f"Modèle chargé: {model_path}")
            else:
                logger.error(f"Modèle non trouvé: {model_path}")
                return False
            
            if Path(preprocessor_path).exists():
                data = joblib.load(preprocessor_path)
                self.preprocessor = data.get('preprocessor')
                logger.info(f"Preprocessor chargé: {preprocessor_path}")
            else:
                logger.warning(f"Preprocessor non trouvé: {preprocessor_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            return False
    
    def preprocess_input(self, data: Dict[str, Any]) -> np.ndarray:
        """Préprocesser les données d'entrée"""
        try:
            # Convertir en DataFrame
            df = pd.DataFrame([data])
            
            # Appliquer le preprocessing si disponible
            if self.preprocessor:
                processed_data = self.preprocessor.transform(df)
            else:
                # Preprocessing minimal si pas de preprocessor
                processed_data = pd.get_dummies(df, drop_first=True).values
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Erreur lors du preprocessing: {e}")
            raise
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Faire une prédiction"""
        try:
            # Préprocesser
            processed_data = self.preprocess_input(data)
            
            # Prédiction
            prediction = self.model.predict(processed_data)[0]
            probability = self.model.predict_proba(processed_data)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Préparer la réponse
            result = {
                'prediction': int(prediction),
                'risk_level': 'High' if prediction == 1 else 'Low',
                'timestamp': datetime.now().isoformat()
            }
            
            if probability is not None:
                result['probability'] = {
                    'low_risk': float(probability[0]),
                    'high_risk': float(probability[1])
                }
                result['confidence'] = float(max(probability))
            
            # Enregistrer pour le monitoring
            self._log_prediction(data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise
    
    def _log_prediction(self, input_data: Dict[str, Any], prediction: Dict[str, Any]):
        """Enregistrer la prédiction pour le monitoring"""
        log_entry = {
            'input': input_data,
            'output': prediction,
            'model_version': '1.0',  # À implémenter selon votre système de versioning
            'timestamp': datetime.now().isoformat()
        }
        
        self.prediction_history.append(log_entry)
        
        # Garder seulement les 1000 dernières prédictions en mémoire
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de monitoring"""
        if not self.prediction_history:
            return {'message': 'Aucune prédiction enregistrée'}
        
        # Calculer les statistiques
        total_predictions = len(self.prediction_history)
        high_risk_count = sum(1 for p in self.prediction_history if p['output']['prediction'] == 1)
        
        stats = {
            'total_predictions': total_predictions,
            'high_risk_predictions': high_risk_count,
            'low_risk_predictions': total_predictions - high_risk_count,
            'high_risk_percentage': (high_risk_count / total_predictions) * 100,
            'last_prediction': self.prediction_history[-1]['timestamp'] if self.prediction_history else None
        }
        
        # Ajouter les statistiques de confiance si disponibles
        confidences = [p['output'].get('confidence') for p in self.prediction_history if p['output'].get('confidence')]
        if confidences:
            stats['confidence_stats'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        
        return stats

# Instance globale de l'API
prediction_api = PredictionAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    try:
        if prediction_api.model is None:
            return jsonify({'status': 'unhealthy', 'reason': 'Model not loaded'}), 503
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de prédiction"""
    try:
        # Validation de l'entrée
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        
        # Faire la prédiction
        result = prediction_api.predict(data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Endpoint de prédiction par batch"""
    try:
        if not request.json or 'data' not in request.json:
            return jsonify({'error': 'No data array provided'}), 400
        
        data_list = request.json['data']
        results = []
        
        for data in data_list:
            try:
                result = prediction_api.predict(data)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint predict_batch: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/monitoring/stats', methods=['GET'])
def monitoring_stats():
    """Endpoint des statistiques de monitoring"""
    try:
        stats = prediction_api.get_monitoring_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint monitoring_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Informations sur le modèle"""
    try:
        info = {
            'model_type': type(prediction_api.model).__name__ if prediction_api.model else None,
            'model_loaded': prediction_api.model is not None,
            'preprocessor_loaded': prediction_api.preprocessor is not None,
            'features_expected': 'Variable selon le preprocessing',
            'output_format': {
                'prediction': 'int (0=Low Risk, 1=High Risk)',
                'risk_level': 'string (Low/High)',
                'probability': 'dict with low_risk and high_risk probabilities',
                'confidence': 'float (max probability)',
                'timestamp': 'ISO format timestamp'
            }
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Configuration pour le développement
    app.run(host='0.0.0.0', port=5000, debug=False)

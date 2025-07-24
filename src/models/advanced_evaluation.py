"""
Évaluation avancée et monitoring des modèles
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.calibration import calibration_curve
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config.config import config
from utils.logging_utils import get_logger, get_metrics_collector

logger = get_logger(__name__)

class ModelEvaluator:
    """Évaluateur avancé de modèles ML"""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.figures_dir = Path("reports/figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "model") -> Dict[str, Any]:
        """Évaluation complète d'un modèle"""
        
        logger.info(f"Évaluation du modèle: {model_name}")
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Métriques de base
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'specificity': self._calculate_specificity(y_test, y_pred),
        }
        
        # Métriques avec probabilités
        if y_pred_proba is not None:
            metrics.update({
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'average_precision': average_precision_score(y_test, y_pred_proba)
            })
        
        # Métriques métier
        business_metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
        metrics.update(business_metrics)
        
        # Génération des visualisations
        self._generate_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)
        
        # Analyse de stabilité
        stability_metrics = self._analyze_prediction_stability(y_pred_proba, model_name)
        metrics.update(stability_metrics)
        
        # Sauvegarder les métriques
        self.metrics_collector.save_metrics(metrics, f"evaluation_{model_name}")
        
        logger.info(f"Évaluation terminée - ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculer la spécificité (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: Optional[np.ndarray]) -> Dict[str, float]:
        """Calculer les métriques métier pour le risque de crédit"""
        
        # Coûts assumés pour la demo (à ajuster selon le contexte métier)
        cost_false_positive = 100  # Coût de refuser un bon client
        cost_false_negative = 1000  # Coût d'accepter un mauvais client
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Coût total
        total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
        
        # Économies par rapport à accepter tous les clients
        baseline_cost = sum(y_true) * cost_false_negative
        savings = baseline_cost - total_cost
        
        metrics = {
            'total_cost': total_cost,
            'cost_savings': savings,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0
        }
        
        # Métriques de profit si probabilités disponibles
        if y_pred_proba is not None:
            profit_metrics = self._calculate_profit_curve(y_true, y_pred_proba)
            metrics.update(profit_metrics)
        
        return metrics
    
    def _calculate_profit_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculer la courbe de profit optimale"""
        
        # Trier par probabilité décroissante
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_sorted = y_true[sorted_indices]
        
        # Calculer le profit cumulé pour différents seuils
        profits = []
        for i in range(len(y_sorted)):
            accepted = i + 1
            true_positives = np.sum(y_sorted[:i+1])
            false_positives = accepted - true_positives
            
            # Profit = (TP * gain_per_good_client) - (FP * loss_per_bad_client)
            profit = (true_positives * 300) - (false_positives * 1000)  # Valeurs exemple
            profits.append(profit)
        
        max_profit_idx = np.argmax(profits)
        optimal_threshold = y_pred_proba[sorted_indices[max_profit_idx]]
        
        return {
            'max_profit': profits[max_profit_idx],
            'optimal_threshold': optimal_threshold,
            'optimal_acceptance_rate': (max_profit_idx + 1) / len(y_true)
        }
    
    def _analyze_prediction_stability(self, y_pred_proba: Optional[np.ndarray],
                                    model_name: str) -> Dict[str, float]:
        """Analyser la stabilité des prédictions"""
        
        if y_pred_proba is None:
            return {}
        
        # Distribution des probabilités
        prob_stats = {
            'prob_mean': np.mean(y_pred_proba),
            'prob_std': np.std(y_pred_proba),
            'prob_skew': self._calculate_skewness(y_pred_proba),
            'prob_kurt': self._calculate_kurtosis(y_pred_proba),
            'prob_entropy': self._calculate_entropy(y_pred_proba)
        }
        
        # Concentration des prédictions
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(y_pred_proba, bins=bins)
        prob_concentration = np.max(hist) / len(y_pred_proba)
        prob_stats['prob_concentration'] = prob_concentration
        
        return prob_stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculer l'asymétrie (skewness)"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculer le kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, proba: np.ndarray) -> float:
        """Calculer l'entropie des probabilités"""
        # Discrétiser les probabilités
        bins = np.linspace(0, 1, 21)
        hist, _ = np.histogram(proba, bins=bins)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]  # Éviter log(0)
        return -np.sum(hist * np.log2(hist))
    
    def _generate_evaluation_plots(self, y_test: np.ndarray, y_pred: np.ndarray,
                                  y_pred_proba: Optional[np.ndarray], model_name: str):
        """Générer les visualisations d'évaluation"""
        
        plt.style.use('seaborn-v0_8')
        
        # Configuration de la figure
        if y_pred_proba is not None:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = axes.ravel()
        
        # 1. Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matrice de Confusion')
        axes[0].set_xlabel('Prédictions')
        axes[0].set_ylabel('Valeurs Réelles')
        
        # 2. Distribution des prédictions
        unique, counts = np.unique(y_pred, return_counts=True)
        axes[1].bar(unique, counts, alpha=0.7)
        axes[1].set_title('Distribution des Prédictions')
        axes[1].set_xlabel('Classe Prédite')
        axes[1].set_ylabel('Nombre')
        
        if y_pred_proba is not None:
            # 3. Courbe ROC
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            axes[2].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
            axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[2].set_xlabel('Taux de Faux Positifs')
            axes[2].set_ylabel('Taux de Vrais Positifs')
            axes[2].set_title('Courbe ROC')
            axes[2].legend()
            axes[2].grid(True)
            
            # 4. Courbe Précision-Rappel
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            axes[3].plot(recall, precision, label=f'PR (AP = {avg_precision:.3f})')
            axes[3].set_xlabel('Rappel')
            axes[3].set_ylabel('Précision')
            axes[3].set_title('Courbe Précision-Rappel')
            axes[3].legend()
            axes[3].grid(True)
            
            # 5. Distribution des probabilités
            axes[4].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Classe 0', density=True)
            axes[4].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Classe 1', density=True)
            axes[4].set_xlabel('Probabilité Prédite')
            axes[4].set_ylabel('Densité')
            axes[4].set_title('Distribution des Probabilités par Classe')
            axes[4].legend()
            
            # 6. Courbe de calibration
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
            axes[5].plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
            axes[5].plot([0, 1], [0, 1], "k:", label="Parfaitement calibré")
            axes[5].set_xlabel('Probabilité Moyenne Prédite')
            axes[5].set_ylabel('Fraction de Positifs')
            axes[5].set_title('Courbe de Calibration')
            axes[5].legend()
            axes[5].grid(True)
        
        plt.tight_layout()
        
        # Sauvegarder
        plot_path = self.figures_dir / f"evaluation_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphiques sauvegardés: {plot_path}")
    
    def compare_models(self, models_metrics: Dict[str, Dict[str, float]]) -> str:
        """Comparer plusieurs modèles"""
        
        logger.info("Génération du rapport de comparaison des modèles")
        
        # Créer un DataFrame pour la comparaison
        df_comparison = pd.DataFrame(models_metrics).T
        
        # Générer le graphique de comparaison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Métriques principales
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        if all(metric in df_comparison.columns for metric in main_metrics):
            df_comparison[main_metrics].plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Métriques Principales')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # ROC-AUC
        if 'roc_auc' in df_comparison.columns:
            df_comparison['roc_auc'].plot(kind='bar', ax=axes[0, 1], color='orange')
            axes[0, 1].set_title('ROC-AUC')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Métriques métier
        business_metrics = ['total_cost', 'cost_savings']
        available_business = [m for m in business_metrics if m in df_comparison.columns]
        if available_business:
            df_comparison[available_business].plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Métriques Métier')
            axes[1, 0].set_ylabel('Valeur')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Temps de traitement ou autre métrique
        if 'prob_mean' in df_comparison.columns:
            df_comparison['prob_mean'].plot(kind='bar', ax=axes[1, 1], color='green')
            axes[1, 1].set_title('Probabilité Moyenne')
            axes[1, 1].set_ylabel('Valeur')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique de comparaison
        comparison_plot_path = self.figures_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Générer le rapport textuel
        report_path = "reports/model_comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT DE COMPARAISON DES MODÈLES ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Tableau de comparaison
            f.write("RÉSUMÉ DES PERFORMANCES:\n")
            f.write(df_comparison.round(4).to_string())
            f.write("\n\n")
            
            # Classement par ROC-AUC
            if 'roc_auc' in df_comparison.columns:
                f.write("CLASSEMENT PAR ROC-AUC:\n")
                ranked = df_comparison.sort_values('roc_auc', ascending=False)
                for i, (model, metrics) in enumerate(ranked.iterrows(), 1):
                    f.write(f"{i}. {model}: {metrics['roc_auc']:.4f}\n")
                f.write("\n")
            
            # Recommandations
            f.write("RECOMMANDATIONS:\n")
            if 'roc_auc' in df_comparison.columns:
                best_model = df_comparison['roc_auc'].idxmax()
                f.write(f"- Meilleur modèle par ROC-AUC: {best_model}\n")
            
            if 'cost_savings' in df_comparison.columns:
                most_profitable = df_comparison['cost_savings'].idxmax()
                f.write(f"- Modèle le plus rentable: {most_profitable}\n")
        
        logger.info(f"Rapport de comparaison généré: {report_path}")
        
        return report_path

def evaluate_saved_model(model_path: str, test_data_path: str) -> Dict[str, Any]:
    """Évaluer un modèle sauvegardé"""
    
    logger.info(f"Évaluation du modèle: {model_path}")
    
    # Charger le modèle
    model = joblib.load(model_path)
    
    # Charger les données de test
    df_test = pd.read_csv(test_data_path)
    target_col = config.data.target_column
    
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].values
    
    # Évaluer
    evaluator = ModelEvaluator()
    model_name = Path(model_path).stem
    metrics = evaluator.evaluate_model(model, X_test, y_test, model_name)
    
    return metrics

if __name__ == "__main__":
    # Évaluer le meilleur modèle sauvegardé
    model_path = f"{config.model.model_dir}/best_model.joblib"
    if Path(model_path).exists():
        metrics = evaluate_saved_model(model_path, config.data.processed_data_path)
        print(f"Métriques d'évaluation: {metrics}")
    else:
        logger.error(f"Modèle non trouvé: {model_path}")

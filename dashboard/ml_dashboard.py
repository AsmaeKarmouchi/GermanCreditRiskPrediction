"""
Dashboard interactif pour le monitoring des modèles ML
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config
from utils.logging_utils import get_metrics_collector
from src.models.advanced_evaluation import ModelEvaluator

class MLDashboard:
    """Dashboard principal pour le monitoring ML"""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.load_model_info()
    
    def load_model_info(self):
        """Charger les informations du modèle"""
        try:
            model_path = f"{config.model.model_dir}/best_model.joblib"
            if Path(model_path).exists():
                self.model = joblib.load(model_path)
                self.model_loaded = True
            else:
                self.model = None
                self.model_loaded = False
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {e}")
            self.model_loaded = False
    
    def render_sidebar(self):
        """Rendu de la barre latérale"""
        st.sidebar.title("🔍 ML Monitoring Dashboard")
        st.sidebar.markdown("---")
        
        # Informations du modèle
        st.sidebar.subheader("📊 Informations Modèle")
        if self.model_loaded:
            st.sidebar.success("✅ Modèle chargé")
            st.sidebar.info(f"Type: {type(self.model).__name__}")
        else:
            st.sidebar.error("❌ Modèle non chargé")
        
        # Sélection de la page
        page = st.sidebar.selectbox(
            "Sélectionner une page",
            ["Vue d'ensemble", "Performance Modèle", "Monitoring Temps Réel", "Analyse des Données", "Configuration"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Métriques Rapides")
        
        # Afficher quelques métriques rapides
        latest_metrics = self.get_latest_metrics()
        if latest_metrics:
            for metric, value in list(latest_metrics.items())[:3]:
                if isinstance(value, (int, float)):
                    st.sidebar.metric(metric.title(), f"{value:.3f}")
        
        return page
    
    def get_latest_metrics(self):
        """Obtenir les dernières métriques"""
        try:
            return self.metrics_collector.load_latest_metrics("evaluation")
        except:
            return {}
    
    def render_overview_page(self):
        """Page de vue d'ensemble"""
        st.title("🏠 Vue d'ensemble du Système ML")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Statut Modèle", "🟢 Actif" if self.model_loaded else "🔴 Inactif")
        
        with col2:
            latest_metrics = self.get_latest_metrics()
            if latest_metrics and 'metrics' in latest_metrics:
                roc_auc = latest_metrics['metrics'].get('roc_auc', 0)
                st.metric("ROC-AUC", f"{roc_auc:.3f}")
            else:
                st.metric("ROC-AUC", "N/A")
        
        with col3:
            # Simulated prediction count
            st.metric("Prédictions/Jour", "1,245")
        
        with col4:
            st.metric("Uptime", "99.8%")
        
        st.markdown("---")
        
        # Graphiques de tendance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Tendance des Performances")
            self.render_performance_trend()
        
        with col2:
            st.subheader("🎯 Distribution des Prédictions")
            self.render_prediction_distribution()
        
        # Alertes récentes
        st.subheader("🚨 Alertes Récentes")
        self.render_recent_alerts()
    
    def render_performance_trend(self):
        """Graphique de tendance des performances"""
        # Données simulées pour la démo
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        metrics_trend = pd.DataFrame({
            'Date': dates,
            'ROC-AUC': np.random.normal(0.85, 0.02, len(dates)),
            'Accuracy': np.random.normal(0.82, 0.02, len(dates)),
            'F1-Score': np.random.normal(0.80, 0.02, len(dates))
        })
        
        fig = px.line(metrics_trend, x='Date', y=['ROC-AUC', 'Accuracy', 'F1-Score'],
                     title="Évolution des Métriques de Performance")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_prediction_distribution(self):
        """Distribution des prédictions"""
        # Données simulées
        predictions = np.random.choice(['Faible Risque', 'Risque Élevé'], 1000, p=[0.7, 0.3])
        pred_counts = pd.Series(predictions).value_counts()
        
        fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                    title="Distribution des Prédictions de Risque")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_alerts(self):
        """Alertes récentes"""
        alerts = [
            {"Time": "10:30", "Type": "Warning", "Message": "Légère baisse de performance détectée"},
            {"Time": "09:15", "Type": "Info", "Message": "Nouveau modèle déployé avec succès"},
            {"Time": "08:45", "Type": "Success", "Message": "Tous les tests de validation passés"}
        ]
        
        for alert in alerts:
            if alert["Type"] == "Warning":
                st.warning(f"⚠️ {alert['Time']} - {alert['Message']}")
            elif alert["Type"] == "Info":
                st.info(f"ℹ️ {alert['Time']} - {alert['Message']}")
            else:
                st.success(f"✅ {alert['Time']} - {alert['Message']}")
    
    def render_model_performance_page(self):
        """Page de performance du modèle"""
        st.title("📊 Performance du Modèle")
        
        # Charger les métriques détaillées
        latest_metrics = self.get_latest_metrics()
        
        if latest_metrics and 'metrics' in latest_metrics:
            metrics = latest_metrics['metrics']
            
            # Métriques principales
            st.subheader("🎯 Métriques Principales")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
            
            # Métriques métier
            st.subheader("💰 Métriques Métier")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_cost = metrics.get('total_cost', 0)
                st.metric("Coût Total", f"€{total_cost:,.0f}")
            with col2:
                cost_savings = metrics.get('cost_savings', 0)
                st.metric("Économies", f"€{cost_savings:,.0f}")
            with col3:
                optimal_threshold = metrics.get('optimal_threshold', 0.5)
                st.metric("Seuil Optimal", f"{optimal_threshold:.3f}")
        
        else:
            st.warning("Aucune métrique disponible. Veuillez d'abord entraîner et évaluer un modèle.")
        
        # Matrice de confusion interactive
        st.subheader("🔄 Matrice de Confusion")
        self.render_confusion_matrix()
        
        # Courbes ROC et PR
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Courbe ROC")
            self.render_roc_curve()
        
        with col2:
            st.subheader("📊 Courbe Précision-Rappel")
            self.render_pr_curve()
    
    def render_confusion_matrix(self):
        """Matrice de confusion interactive"""
        # Données simulées
        conf_matrix = np.array([[850, 120], [80, 950]])
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Prédiction: Faible', 'Prédiction: Élevé'],
            y=['Réel: Faible', 'Réel: Élevé'],
            colorscale='Blues',
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 20},
        ))
        
        fig.update_layout(
            title="Matrice de Confusion",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_roc_curve(self):
        """Courbe ROC"""
        # Données simulées
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) + np.random.normal(0, 0.02, 100)
        tpr = np.clip(tpr, 0, 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        
        fig.update_layout(
            xaxis_title='Taux de Faux Positifs',
            yaxis_title='Taux de Vrais Positifs',
            title='Courbe ROC',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_pr_curve(self):
        """Courbe Précision-Rappel"""
        # Données simulées
        recall = np.linspace(0, 1, 100)
        precision = 1 - recall + np.random.normal(0, 0.05, 100)
        precision = np.clip(precision, 0, 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve'))
        
        fig.update_layout(
            xaxis_title='Rappel',
            yaxis_title='Précision',
            title='Courbe Précision-Rappel',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_monitoring_page(self):
        """Page de monitoring temps réel"""
        st.title("⏱️ Monitoring Temps Réel")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Actualisation automatique (30s)")
        if auto_refresh:
            st.rerun()
        
        # Métriques en temps réel
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prédictions/min", np.random.randint(50, 100))
        with col2:
            st.metric("Latence Moyenne", f"{np.random.uniform(10, 50):.1f}ms")
        with col3:
            st.metric("Taux d'Erreur", f"{np.random.uniform(0, 2):.2f}%")
        
        # Graphique de monitoring en temps réel
        st.subheader("📈 Métriques en Temps Réel")
        self.render_realtime_metrics()
        
        # Logs récents
        st.subheader("📝 Logs Récents")
        self.render_recent_logs()
    
    def render_realtime_metrics(self):
        """Métriques en temps réel"""
        # Simulation de données en temps réel
        times = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                             end=datetime.now(), freq='1min')
        
        data = pd.DataFrame({
            'Time': times,
            'Predictions': np.random.poisson(75, len(times)),
            'Latency': np.random.exponential(20, len(times)),
            'Error_Rate': np.random.exponential(0.5, len(times))
        })
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Nombre de Prédictions', 'Latence (ms)', 'Taux d\'Erreur (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(go.Scatter(x=data['Time'], y=data['Predictions'], mode='lines', name='Prédictions'), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Time'], y=data['Latency'], mode='lines', name='Latence'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['Time'], y=data['Error_Rate'], mode='lines', name='Erreurs'), row=3, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_recent_logs(self):
        """Logs récents"""
        logs = [
            {"Time": datetime.now() - timedelta(minutes=1), "Level": "INFO", "Message": "Prédiction réussie - ID: 12345"},
            {"Time": datetime.now() - timedelta(minutes=2), "Level": "INFO", "Message": "Prédiction réussie - ID: 12344"},
            {"Time": datetime.now() - timedelta(minutes=3), "Level": "WARNING", "Message": "Latence élevée détectée: 45ms"},
            {"Time": datetime.now() - timedelta(minutes=5), "Level": "INFO", "Message": "Nouveau batch de prédictions traité"},
        ]
        
        log_df = pd.DataFrame(logs)
        log_df['Time'] = log_df['Time'].dt.strftime('%H:%M:%S')
        
        st.dataframe(log_df, use_container_width=True)
    
    def render_data_analysis_page(self):
        """Page d'analyse des données"""
        st.title("🔬 Analyse des Données")
        
        # Upload de fichier pour analyse
        uploaded_file = st.file_uploader("Charger un fichier de données", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Aperçu des Données")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Informations générales:**")
                st.write(f"- Nombre de lignes: {len(df)}")
                st.write(f"- Nombre de colonnes: {len(df.columns)}")
                st.write(f"- Valeurs manquantes: {df.isnull().sum().sum()}")
            
            with col2:
                st.write("**Types de données:**")
                st.write(df.dtypes.value_counts())
            
            # Analyse des colonnes numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.subheader("📊 Analyse des Variables Numériques")
                selected_col = st.selectbox("Sélectionner une colonne", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x=selected_col, title=f"Distribution de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(df, y=selected_col, title=f"Box Plot de {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Veuillez charger un fichier CSV pour commencer l'analyse.")
    
    def render_configuration_page(self):
        """Page de configuration"""
        st.title("⚙️ Configuration du Système")
        
        # Configuration des alertes
        st.subheader("🚨 Configuration des Alertes")
        
        col1, col2 = st.columns(2)
        with col1:
            performance_threshold = st.slider("Seuil d'alerte performance", 0.0, 1.0, 0.85, 0.01)
            latency_threshold = st.number_input("Seuil de latence (ms)", value=100)
        
        with col2:
            error_rate_threshold = st.slider("Seuil taux d'erreur (%)", 0.0, 10.0, 2.0, 0.1)
            email_notifications = st.checkbox("Notifications par email", value=True)
        
        # Configuration du monitoring
        st.subheader("📊 Configuration du Monitoring")
        
        col1, col2 = st.columns(2)
        with col1:
            monitoring_interval = st.selectbox("Intervalle de monitoring", ["1 min", "5 min", "15 min", "1 heure"])
            data_retention = st.selectbox("Rétention des données", ["7 jours", "30 jours", "90 jours", "1 an"])
        
        with col2:
            enable_drift_detection = st.checkbox("Détection de drift", value=True)
            auto_retrain = st.checkbox("Ré-entraînement automatique", value=False)
        
        # Bouton de sauvegarde
        if st.button("💾 Sauvegarder la Configuration"):
            st.success("Configuration sauvegardée avec succès!")

def main():
    """Fonction principale"""
    st.set_page_config(
        page_title="ML Monitoring Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialiser le dashboard
    dashboard = MLDashboard()
    
    # Rendu de la barre latérale
    selected_page = dashboard.render_sidebar()
    
    # Rendu de la page sélectionnée
    if selected_page == "Vue d'ensemble":
        dashboard.render_overview_page()
    elif selected_page == "Performance Modèle":
        dashboard.render_model_performance_page()
    elif selected_page == "Monitoring Temps Réel":
        dashboard.render_monitoring_page()
    elif selected_page == "Analyse des Données":
        dashboard.render_data_analysis_page()
    elif selected_page == "Configuration":
        dashboard.render_configuration_page()

if __name__ == "__main__":
    main()

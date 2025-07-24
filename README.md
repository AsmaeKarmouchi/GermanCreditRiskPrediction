# Projet MLOps  - GermanCredit Risk Prediction

##  Vue d'ensemble

Ce projet constitue une **pipeline MLOps   ** pour la prédiction du risque de crédit basée sur le dataset German Credit. Il implémente les meilleures pratiques de l'industrie avec monitoring avancé, API de production, et dashboard interactif.

##  Fonctionnalités Principales

###  MLOps & Engineering
- **Pipeline automatisée** avec DVC pour la reproductibilité
- **Feature engineering avancé** avec création automatique de variables métier
- **Hyperparameter tuning** avec recherche aléatoire et validation croisée
- **Ensemble learning** avec multiple algorithmes (RF, Gradient Boosting, SVM, Logistic Regression)
- **Configuration centralisée** avec gestion d'environnements
- **Logging structuré** avec niveaux multiples et persistence

###  Modélisation Avancée
- **Multiple algorithmes** avec sélection automatique du meilleur
- **Calibration des probabilités** pour des prédictions fiables
- **Métriques métier** (coût, profit, seuil optimal)
- **Validation robuste** avec stratified k-fold
- **Détection d'outliers** avec méthodes IQR et Z-score
- **Stabilité des prédictions** avec analyse statistique

###  Monitoring & Évaluation
- **Métriques complètes** : ROC-AUC, Precision, Recall, F1, métriques métier
- **Visualisations avancées** : courbes ROC/PR, matrices de confusion, calibration
- **Monitoring temps réel** avec collecte de métriques
- **Comparaison de modèles** automatique
- **Drift detection** (prêt à implémenter)

###  Production & Déploiement
- **API RESTful** avec Flask pour les prédictions en temps réel
- **Batch predictions** pour traitement en lot
- **Health checks** et monitoring API
- **Dashboard interactif** Streamlit pour le monitoring
- **Tests complets** avec unittest et couverture de code

##  Architecture

```
 GermanCreditRiskPrediction/
├──  api/                     # API de prédiction
│   └── prediction_api.py       # API Flask avec monitoring
├──  config/                  # Configuration centralisée
│   └── config.py               # Classes de configuration
├──  dashboard/               # Dashboard de monitoring
│   └── ml_dashboard.py         # Dashboard Streamlit interactif
├──  data/
│   ├──  raw/                 # Données brutes
│   ├──  processed/           # Données preprocessées
│   └──  features/            # Feature store
├──  models/                  # Modèles entraînés
├──  notebooks/               # Notebooks d'exploration
├──  reports/                 # Rapports et visualisations
│   ├──  figures/             # Graphiques générés
│   └──  metrics/             # Métriques sauvegardées
├── src/
│   ├──  data/                # Scripts de données
│   ├──  features/            # Feature engineering avancé
│   │   └── advanced_features.py
│   ├──  models/              # Modélisation avancée
│   │   ├── advanced_training.py    # Entraînement avec ensemble
│   │   └── advanced_evaluation.py # Évaluation complète
│   └──  visualization/       # Visualisations
├──  tests/                   # Tests complets
│   └── test_ml_pipeline.py     # Tests d'intégration
├──  utils/                   # Utilitaires
│   └── logging_utils.py        # Logging et MLFlow
├──  dvc.yaml                 # Pipeline DVC
├──  params.yaml              # Configuration des hyperparamètres
└──  requirements.txt         # Dépendances

```

## Démarrage Rapide

### 1. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 2. Configuration de l'environnement
```bash
# Initialiser DVC
dvc init

# Configurer MLFlow (optionnel)
mlflow ui  # Lance l'interface MLFlow
```

### 3. Exécution du pipeline complet
```bash
# Pipeline automatique avec DVC
dvc repro

# Ou exécution manuelle étape par étape
python src/data/download.py
python src/features/advanced_features.py
python src/models/advanced_training.py
python src/models/advanced_evaluation.py
```

### 4. Lancement des services

#### API de prédiction
```bash
python api/prediction_api.py
# Accessible sur http://localhost:5000
```

#### Dashboard de monitoring
```bash
streamlit run dashboard/ml_dashboard.py
# Accessible sur http://localhost:8501
```

## 📊 Utilisation de l'API

### Prédiction unique
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 24,
    "amount": 5000,
    "age": 35,
    "status": "0 <= ... < 200 DM"
  }'
```

### Prédiction par lot
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"duration": 24, "amount": 5000, "age": 35},
      {"duration": 12, "amount": 2000, "age": 28}
    ]
  }'
```

### Health check
```bash
curl http://localhost:5000/health
```

##  Tests

```bash
# Exécuter tous les tests
python -m pytest tests/ -v

# Tests avec couverture
python -m pytest tests/ --cov=src --cov-report=html

# Tests spécifiques
python tests/test_ml_pipeline.py
```

##  Métriques et Monitoring

### Métriques Techniques
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC, Average Precision**
- **Specificity, NPV**

### Métriques Métier
- **Coût total des erreurs**
- **Économies réalisées**
- **Seuil optimal de décision**
- **Taux d'acceptation optimal**

### Monitoring Temps Réel
- **Latence des prédictions**
- **Taux d'erreur**
- **Distribution des prédictions**
- **Stabilité du modèle**

##  Configuration Avancée

### Hyperparamètres (params.yaml)
```yaml
train:
  test_size: 0.2
  random_state: 42

preprocessing:
  outlier_method: "iqr"
  scaling_method: "standard"
  feature_selection: true

hyperparameter_tuning:
  n_trials: 50
  optimization_method: "random_search"

monitoring:
  performance_threshold: 0.85
  drift_detection: true
```

### Variables d'environnement
```bash
# MLFlow tracking
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Logging level
export LOG_LEVEL=INFO

# API configuration
export API_HOST=0.0.0.0
export API_PORT=5000
```

##  Améliorations 

### Code Quality
-  **Architecture modulaire** avec séparation des responsabilités
-  **Configuration centralisée** vs hardcodage
-  **Logging structuré** vs print statements
-  **Gestion d'erreurs robuste** avec try/catch appropriés
-  **Documentation complète** avec docstrings
-  **Type hints** pour la clarté du code

### ML Engineering
-  **Feature engineering automatisé** vs manuel
-  **Multiple algorithmes** vs un seul RandomForest
-  **Hyperparameter tuning** vs paramètres fixes
-  **Ensemble learning** vs modèle unique
-  **Validation croisée stratifiée** vs simple train/test
-  **Calibration des probabilités** vs prédictions brutes

### Production Ready
-  **API RESTful complète** vs script isolé
-  **Monitoring et métriques** vs pas de suivi
-  **Dashboard interactif** vs pas de visualisation
-  **Tests automatisés** vs pas de tests
-  **Pipeline reproductible** avec DVC
-  **Déploiement containerisé** ready

### Business Impact
-  **Métriques métier** (coût, profit) vs métriques techniques uniquement
-  **Optimisation de seuil** pour le business
-  **Analyse de rentabilité** intégrée
-  **Monitoring de la dérive** pour la maintenance

##  Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request


## Remerciements

- Dataset German Credit de UCI ML Repository
- Outils open source utilisés : scikit-learn, MLFlow, DVC, Streamlit


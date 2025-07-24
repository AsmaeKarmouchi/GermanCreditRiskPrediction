#  Guide de Démarrage Rapide

## Lancement Automatique (Recommandé)

### Option 1: Script Python (Multiplateforme)
```bash
python3 launch_project.py
```

### Option 2: Script Linux/WSL
```bash
chmod +x run_project.sh
./run_project.sh
```

### Option 3: Script Windows
```cmd
run_project.bat
```

## Lancement Manuel Étape par Étape

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Exécuter le pipeline ML (optionnel)
```bash
# Feature engineering
python src/features/advanced_features.py

# Entraînement des modèles
python src/models/advanced_training.py

# Évaluation
python src/models/advanced_evaluation.py
```

### 3. Lancer les services

#### Terminal 1: API de prédiction
```bash
python api/prediction_api.py
```
 API disponible sur: http://localhost:5000

#### Terminal 2: Dashboard de monitoring
```bash
streamlit run dashboard/ml_dashboard.py
```
 Dashboard disponible sur: http://localhost:8501

##  Interfaces Web

###  Dashboard Streamlit (http://localhost:8501)
- **Vue d'ensemble**: Métriques principales et alertes
- **Performance Modèle**: Évaluation détaillée avec graphiques
- **Monitoring Temps Réel**: Suivi des prédictions en direct
- **Analyse des Données**: Upload et analyse de nouveaux fichiers
- **Configuration**: Paramètres du système

###  API REST (http://localhost:5000)
- **GET /health**: Vérification de santé du service
- **POST /predict**: Prédiction unique
- **POST /predict/batch**: Prédictions en lot
- **GET /monitoring/stats**: Statistiques de monitoring
- **GET /model/info**: Informations du modèle

##  Exemples d'utilisation de l'API

### Test de santé
```bash
curl http://localhost:5000/health
```

### Prédiction unique
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "status": "0 <= ... < 200 DM",
    "duration": 24,
    "amount": 5000,
    "age": 35,
    "employment_duration": "1 <= ... < 4 years"
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

### Statistiques de monitoring
```bash
curl http://localhost:5000/monitoring/stats
```

##  Fichiers Générés

Après exécution, vous trouverez:

- **models/best_model.joblib**: Meilleur modèle entraîné
- **data/processed/**: Données préprocessées
- **reports/figures/**: Graphiques d'évaluation (PNG)
- **reports/metrics/**: Métriques sauvegardées (JSON)
- **logs/**: Logs d'exécution
- **mlflow.db**: Base de données MLFlow (si utilisée)

##  Résolution de Problèmes

### Erreur de dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Port déjà utilisé
- Streamlit: Changez le port avec `--server.port 8502`
- API: Modifiez le port dans `api/prediction_api.py`

### Erreur MLFlow
```bash
# Désactiver temporairement MLFlow dans utils/logging_utils.py
# Commenter les lignes d'import mlflow
```

### Modèle non trouvé
```bash
# Ré-exécuter l'entraînement
python src/models/advanced_training.py
```

## Fonctionnalités Démontrées

**Pipeline ML Automatisé**
- Feature engineering avancé
- Ensemble learning (4+ algorithmes)
- Hyperparameter tuning
- Validation croisée stratifiée

**Métriques Complètes**
- Techniques: ROC-AUC, Precision, Recall, F1
- Métier: Coût, profit, seuil optimal
- Monitoring: Latence, taux d'erreur

 **Production Ready**
- API RESTful avec monitoring
- Dashboard interactif
- Tests automatisés
- Configuration centralisée

 **MLOps Best Practices**
- Versioning avec DVC
- Logging structuré
- Pipeline reproductible
- Architecture modulaire


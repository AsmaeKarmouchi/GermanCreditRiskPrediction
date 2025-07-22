import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yaml
import os

# Charger les hyperparamètres depuis params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

def train():
    # Charger les données prétraitées
    df = pd.read_csv("data/processed/GermanCredit_processed.csv")

    # colonne cible (credit_risk)
    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    # Séparer les données en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"]
    )

    # Initialiser le modèle
    clf = RandomForestClassifier(
        n_estimators=params["train"]["n_estimators"],
        max_depth=params["train"]["max_depth"]
    )
    clf.fit(X_train, y_train)

    # Sauvegarder le modèle
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/random_forest.joblib")
    print(" Modèle sauvegardé dans models/random_forest.joblib")

if __name__ == "__main__":
    train()

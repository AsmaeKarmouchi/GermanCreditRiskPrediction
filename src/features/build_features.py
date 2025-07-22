import pandas as pd
import os

def preprocess():
    raw_path = "data/raw/GermanCredit.csv"
    processed_path = "data/processed/GermanCredit_processed.csv"
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(raw_path)

    # Exemple de traitement simple
    df = df.dropna()  # Suppression des lignes vides
    df = pd.get_dummies(df, drop_first=True)  # Encodage one-hot

    df.to_csv(processed_path, index=False)
    print("Données prétraitées sauvegardées dans", processed_path)

if __name__ == "__main__":
    preprocess()

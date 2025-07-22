import pandas as pd
from sklearn.metrics import classification_report
import joblib

def evaluate():
    df = pd.read_csv("data/processed/GermanCredit_processed.csv")
    X = df.drop("credit_risk", axis=1)  
    y = df["credit_risk"]               

    model = joblib.load("models/random_forest.joblib")
    y_pred = model.predict(X)

    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate()

import pandas as pd
import os

def download_german_credit():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
    os.makedirs("data/raw", exist_ok=True)
    df = pd.read_csv(url)
    df.to_csv("data/raw/GermanCredit.csv", index=False)
    print("Données téléchargées et sauvegardées dans data/raw/GermanCredit.csv")

if __name__ == "__main__":
    download_german_credit() 
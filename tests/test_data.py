# tests/test_data.py
import pandas as pd

def test_data_loading():
    df = pd.read_csv("data/raw/GermanCredit.csv")
    assert not df.empty, " Le dataset est vide"
    assert "Class" in df.columns, " Colonne cible 'Class' manquante"

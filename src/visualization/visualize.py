import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize():
    df = pd.read_csv("data/raw/GermanCredit.csv")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Class")
    plt.title("Distribution des classes")
    plt.savefig("data/processed/class_distribution.png")
    plt.show()

if __name__ == "__main__":
    visualize()

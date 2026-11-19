
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer


def collect_data():
    print("Starting data collection...")

    # Load Breast Cancer dataset
    dataset = load_breast_cancer()

    # Convert dataset into a DataFrame
    df = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    # Add target column
    df["target"] = dataset.target

    # Create raw data directory
    os.makedirs("data/raw", exist_ok=True)

    # Save dataset
    output_path = "data/raw/breast_cancer.csv"
    df.to_csv(output_path, index=False)

    print("Dataset collected successfully!")
    print("Dataset shape:", df.shape)
    print("Saved location:", output_path)

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nFirst five records:")
    print(df.head())


if __name__ == "__main__":
    collect_data()


import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data():

    print("Starting data preprocessing...")

    # Load raw dataset
    df = pd.read_csv("data/raw/breast_cancer.csv")

    print("Original dataset shape:", df.shape)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove rows containing missing values
    df = df.dropna()

    print("Dataset shape after cleaning:", df.shape)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale features without data leakage
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data into DataFrames
    X_train_df = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    X_test_df = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    # Add target column
    X_train_df["target"] = y_train.reset_index(drop=True)
    X_test_df["target"] = y_test.reset_index(drop=True)

    # Create output directory
    os.makedirs("data/processed", exist_ok=True)

    # Save processed datasets
    X_train_df.to_csv(
        "data/processed/train.csv",
        index=False
    )

    X_test_df.to_csv(
        "data/processed/test.csv",
        index=False
    )

    print("\nPreprocessing completed successfully!")

    print("Training dataset shape:", X_train_df.shape)
    print("Testing dataset shape:", X_test_df.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    print("\nProcessed datasets saved successfully!")


if __name__ == "__main__":
    preprocess_data()

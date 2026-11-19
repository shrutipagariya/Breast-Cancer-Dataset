
import os
import pandas as pd


def validate_data():
    print("Starting data validation...")

    train_path = "data/features/train_features.csv"
    test_path = "data/features/test_features.csv"

    # Check whether files exist
    if not os.path.exists(train_path):
        raise FileNotFoundError("Training feature file not found!")

    if not os.path.exists(test_path):
        raise FileNotFoundError("Testing feature file not found!")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Training dataset shape:", train_df.shape)
    print("Testing dataset shape:", test_df.shape)

    required_columns = [
        "target",
        "mean_radius_squared",
        "mean_perimeter_squared"
    ]

    # Check required columns
    for column in required_columns:
        if column not in train_df.columns:
            raise ValueError(f"Missing column in training data: {column}")

        if column not in test_df.columns:
            raise ValueError(f"Missing column in testing data: {column}")

    # Check missing values
    if train_df.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in training data!")

    if test_df.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in testing data!")

    # Check target values
    valid_targets = {0, 1}

    if not set(train_df["target"].unique()).issubset(valid_targets):
        raise ValueError("Invalid target values in training data!")

    if not set(test_df["target"].unique()).issubset(valid_targets):
        raise ValueError("Invalid target values in testing data!")

    print("\nAll required columns are present.")
    print("No missing values found.")
    print("Target values are valid.")
    print("\nData validation completed successfully!")


if __name__ == "__main__":
    validate_data()

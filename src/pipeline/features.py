
import os
import pandas as pd


def engineer_features():
    print("Starting feature engineering...")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    print("Original training shape:", train_df.shape)
    print("Original testing shape:", test_df.shape)

    # Create squared features
    train_df["mean_radius_squared"] = train_df["mean radius"] ** 2
    train_df["mean_perimeter_squared"] = train_df["mean perimeter"] ** 2

    test_df["mean_radius_squared"] = test_df["mean radius"] ** 2
    test_df["mean_perimeter_squared"] = test_df["mean perimeter"] ** 2

    # Create output directory
    os.makedirs("data/features", exist_ok=True)

    # Save engineered datasets
    train_df.to_csv("data/features/train_features.csv", index=False)
    test_df.to_csv("data/features/test_features.csv", index=False)

    print("\nFeature engineering completed successfully!")
    print("New training shape:", train_df.shape)
    print("New testing shape:", test_df.shape)

    print("\nNew features:")
    print("1. mean_radius_squared")
    print("2. mean_perimeter_squared")

    print("\nFirst five engineered records:")
    print(train_df.head())

    print("\nEngineered datasets saved successfully!")


if __name__ == "__main__":
    engineer_features()


import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the Breast Cancer dataset
data = load_breast_cancer()

# Convert dataset into a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df["target"] = data.target

# Save dataset as CSV
df.to_csv("data/raw/breast_cancer.csv", index=False)

# Display dataset information
print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nTarget distribution:")
print(df["target"].value_counts())

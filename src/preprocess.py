
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load the raw dataset
df = pd.read_csv("data/raw/breast_cancer.csv")

# Separate input features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create DataFrames
X_train_df = pd.DataFrame(
    X_train_scaled, columns=X.columns
)

X_test_df = pd.DataFrame(
    X_test_scaled, columns=X.columns
)

# Add target column
X_train_df["target"] = y_train.reset_index(drop=True)
X_test_df["target"] = y_test.reset_index(drop=True)

# Create processed data folder
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save processed datasets
X_train_df.to_csv("data/processed/train.csv", index=False)
X_test_df.to_csv("data/processed/test.csv", index=False)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Display results
print("Data preprocessing completed!")

print("\nTraining dataset shape:", X_train_df.shape)
print("Testing dataset shape:", X_test_df.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nProcessed datasets and scaler saved successfully!")

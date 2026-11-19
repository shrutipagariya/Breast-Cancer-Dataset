
import mlflow
import pandas as pd

# Connect to MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered model using its candidate alias
model_uri = "models:/Breast-Cancer-Classifier@candidate"

print("Loading registered model...")

model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully!")

# Load test dataset
test_df = pd.read_csv("data/processed/test.csv")

# Separate features and target
X_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

# Select first five records for prediction
sample_data = X_test.head(5)
actual_values = y_test.head(5)

# Make predictions
predictions = model.predict(sample_data)

# Display results
print("\nMODEL PREDICTIONS")
print("-----------------")

for i in range(len(predictions)):
    predicted = int(predictions[i])
    actual = int(actual_values.iloc[i])

    predicted_label = "Benign" if predicted == 1 else "Malignant"
    actual_label = "Benign" if actual == 1 else "Malignant"

    print(f"\nRecord {i + 1}")
    print("Predicted:", predicted_label)
    print("Actual:", actual_label)

print("\nRegistered model testing completed successfully!")

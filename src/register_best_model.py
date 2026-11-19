
import mlflow
from mlflow.tracking import MlflowClient

# Connect to the MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# Get the Breast Cancer experiment
experiment = client.get_experiment_by_name(
    "Breast_Cancer_Classification"
)

if experiment is None:
    raise RuntimeError("Breast Cancer experiment not found.")

experiment_id = experiment.experiment_id

print("Experiment ID:", experiment_id)

# Find all successfully completed runs
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"]
)

if not runs:
    raise RuntimeError("No successful runs found.")

# Select the model with the highest accuracy
best_run = runs[0]

best_run_id = best_run.info.run_id
best_accuracy = best_run.data.metrics["accuracy"]
best_model_name = best_run.data.params["model_name"]

print("\nBEST MODEL")
print("Model:", best_model_name)
print("Run ID:", best_run_id)
print("Accuracy:", round(best_accuracy, 4))

# Register the best model
model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "Breast-Cancer-Classifier"

print("\nRegistering best model...")

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print("\nMODEL REGISTERED SUCCESSFULLY")
print("Model Name:", registered_model_name)
print("Model Version:", model_version.version)

# Assign candidate alias (MLflow 3.x)
client.set_registered_model_alias(
    name=registered_model_name,
    alias="candidate",
    version=model_version.version
)

print("\nModel alias assigned successfully.")
print("Alias: candidate")

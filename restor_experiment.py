import mlflow
from mlflow.tracking import MlflowClient
import mlflow.entities

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Create an MLflow client
client = MlflowClient()

# List all experiments, including deleted ones
experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

# Print all experiments and their states
print("Experiments before restoring:")
for experiment in experiments:
    print(f"ID: {experiment.experiment_id}, Name: {experiment.name}, State: {experiment.lifecycle_stage}")

# Find the deleted experiment (example ID 0)
deleted_experiment_id = 0  # Replace with the actual ID of your deleted experiment

# Restore the deleted experiment
client.restore_experiment(deleted_experiment_id)
print(f"Experiment {deleted_experiment_id} has been restored.")

# List all experiments again to verify
experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

print("Experiments after restoring:")
for experiment in experiments:
    print(f"ID: {experiment.experiment_id}, Name: {experiment.name}, State: {experiment.lifecycle_stage}")

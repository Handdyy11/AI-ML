"""Shared configuration for the project's local MLflow tracking store."""
import mlflow

from src import config


def configure_tracking() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT,
            artifact_location=config.MLFLOW_ARTIFACT_ROOT,
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

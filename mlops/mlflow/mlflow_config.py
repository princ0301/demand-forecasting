import mlflow

from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "demand-forecasting"


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep))
        else:
            items[new_key] = value
    return items


def log_run(model_name: str, params: dict, metrics: dict) -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    flat_params = flatten_dict(params)
    flat_params = {key: str(value) for key, value in flat_params.items()}

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_params(flat_params)
        mlflow.log_metrics(metrics)

    logger.info(f"logged mlflow run for {model_name}")
from mlops.mlflow.mlflow_config import log_run
from src.utils.io import load_config

config = load_config()

RESULTS = {
    "lstm": {
        "wrmsse": 1.0775048974224855,
        "mean_rmse": 1.7393834892480247,
        "mean_mape": 60.7822575759572,
    },
    "nbeats": {
        "wrmsse": 1.0493057564559753,
        "mean_rmse": 1.7280922772839822,
        "mean_mape": 61.669845722618476,
    },
    "tft": {
        "wrmsse": 1.0547250981580782,
        "mean_rmse": 1.7884287213541128,
        "mean_mape": 76.0103666948323,
    },
}

if __name__ == "__main__":
    for model_name, metrics in RESULTS.items():
        params = config["models"].get(model_name, {})
        log_run(model_name, params, metrics)
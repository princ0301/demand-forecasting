import sys
from pathlib import Path

import polars as pl

from mlops.mlflow.mlflow_config import log_run
from src.evaluation.backtest import evaluate_predictions, train_test_split
from src.models.baseline import NaiveModel, SeasonalNaiveModel
from src.models.lightgbm_model import LightGBMModel
from src.models.lstm_model import LSTMModel
from src.models.nbeats_model import NBeatsModel
from src.models.xgboost_model import XGBoostModel
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_BUILDERS = {
    "naive": lambda config: NaiveModel(),
    "seasonal_naive": lambda config: SeasonalNaiveModel(),
    "lightgbm": lambda config: LightGBMModel(
        params=config["models"]["lightgbm"]["params"],
        lags=config["features"]["lags"],
        rolling_windows=config["features"]["rolling_windows"],
        num_boost_round=config["models"]["lightgbm"]["num_boost_round"],
    ),
    "xgboost": lambda config: XGBoostModel(
        params=config["models"]["xgboost"]["params"],
        lags=config["features"]["lags"],
        rolling_windows=config["features"]["rolling_windows"],
        num_boost_round=config["models"]["xgboost"]["num_boost_round"],
    ),
    "lstm": lambda config: LSTMModel(
        lookback=config["models"]["lstm"]["lookback"],
        horizon=config["evaluation"]["horizon"],
        hidden_size=config["models"]["lstm"]["hidden_size"],
        embedding_dim=config["models"]["lstm"]["embedding_dim"],
        num_layers=config["models"]["lstm"]["num_layers"],
        epochs=config["models"]["lstm"]["epochs"],
        batch_size=config["models"]["lstm"]["batch_size"],
        learning_rate=config["models"]["lstm"]["learning_rate"],
        stride=config["models"]["lstm"]["stride"],
    ),
    "nbeats": lambda config: NBeatsModel(
        lookback=config["models"]["nbeats"]["lookback"],
        horizon=config["evaluation"]["horizon"],
        hidden_size=config["models"]["nbeats"]["hidden_size"],
        embedding_dim=config["models"]["nbeats"]["embedding_dim"],
        num_blocks=config["models"]["nbeats"]["num_blocks"],
        num_fc_layers=config["models"]["nbeats"]["num_fc_layers"],
        epochs=config["models"]["nbeats"]["epochs"],
        batch_size=config["models"]["nbeats"]["batch_size"],
        learning_rate=config["models"]["nbeats"]["learning_rate"],
        stride=config["models"]["nbeats"]["stride"],
    ),
}


def run_models(config: dict, model_names: list[str]) -> dict:
    features_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    df = pl.read_parquet(features_path)

    horizon = config["evaluation"]["horizon"]
    weight_window = config["evaluation"]["weight_window"]
    train, test = train_test_split(df, horizon)

    results = {}
    for name in model_names:
        logger.info(f"training {name}")
        model = MODEL_BUILDERS[name](config)
        model.fit(train)
        predictions = model.predict(test)
        results[name] = evaluate_predictions(train, test, predictions, weight_window)

        params = config["models"].get(name, {})
        log_run(name, params, results[name])

    return results


if __name__ == "__main__":
    # usage: uv run python -m src.training.train xgboost
    #        uv run python -m src.training.train           (runs all registered models)
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_BUILDERS.keys())

    config = load_config()
    results = run_models(config, requested)

    for name, metrics in results.items():
        logger.info(f"{name}: {metrics}")
"""Optuna hyperparameter search for the existing LightGBM holdout experiment."""

import argparse
from pathlib import Path

import optuna
import polars as pl

from mlops.mlflow.mlflow_config import log_run
from src.evaluation.backtest import evaluate_predictions, train_test_split
from src.models.lightgbm_model import LightGBMModel
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def suggest_params(trial: optuna.Trial, base_params: dict) -> tuple[dict, int]:
    """Sample one coherent LightGBM configuration for the fixed holdout."""
    params = {
        **base_params,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
        "min_data_in_leaf": trial.suggest_categorical("min_data_in_leaf", [50, 100, 200, 500]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.05, 1.35),
    }
    num_boost_round = trial.suggest_categorical("num_boost_round", [300, 500, 750])
    return params, num_boost_round


def run_tuning(config: dict, n_trials: int) -> optuna.Study:
    features_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    df = pl.read_parquet(features_path)
    train, test = train_test_split(df, config["evaluation"]["horizon"])
    model_config = config["models"]["lightgbm"]
    tuning_config = config["models"]["tuning"]

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_params(trial, model_config["params"])
        model = LightGBMModel(
            params=params,
            lags=config["features"]["lags"],
            rolling_windows=config["features"]["rolling_windows"],
            num_boost_round=num_boost_round,
        )
        model.fit(train)
        predictions = model.predict(test)
        metrics = evaluate_predictions(train, test, predictions, config["evaluation"]["weight_window"])
        trial.set_user_attr("metrics", metrics)
        log_run(f"lightgbm_optuna_trial_{trial.number}", {**params, "num_boost_round": num_boost_round}, metrics)
        logger.info("trial %s: WRMSSE=%.6f", trial.number, metrics["wrmsse"])
        return metrics["wrmsse"]

    sampler = optuna.samplers.TPESampler(seed=tuning_config["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="lightgbm-optuna")
    study.optimize(objective, n_trials=n_trials)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None, help="Override models.tuning.n_trials.")
    args = parser.parse_args()

    config = load_config()
    trials = args.trials or config["models"]["tuning"]["n_trials"]
    study = run_tuning(config, trials)
    logger.info("best WRMSSE: %.6f", study.best_value)
    logger.info("best parameters: %s", study.best_params)

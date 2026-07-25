from pathlib import Path
import polars as pl

from src.evaluation.backtest import evaluate_predictions, train_test_split
from src.models.baseline import NaiveModel, SeasonalNaiveModel
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "naive": NaiveModel,
    "seasonal_naive": SeasonalNaiveModel,
}

def run_baselines(config: dict) -> dict:
    features_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    df = pl.read_parquet(features_path)

    horizon = config["evaluation"]["horizon"]
    weight_window = config["evaluation"]["weight_window"]
    train, test = train_test_split(df, horizon)

    results = {}
    for name, model_cls in MODEL_REGISTRY.items():
        model = model_cls()
        model.fit(train)
        predictions = model.predict(test)
        results[name] = evaluate_predictions(train, test, predictions, weight_window)

    return results

if __name__ == "__main__":
    config = load_config()
    results = run_baselines(config)

    for name, metrics in results.items():
        logger.info(f"{name}: {metrics}")
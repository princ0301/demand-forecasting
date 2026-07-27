import gc
import sys
from pathlib import Path

from src.data.merge_data import build_dataset
from src.data.validate_data import run_validation
from src.features.build_features import build_feature_table
from src.training.train import run_models
from src.utils.io import load_config, write_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODELS = ["naive", "seasonal_naive", "lightgbm", "xgboost"]


def run_pipeline(config: dict, model_names: list[str]) -> dict:
    logger.info("stage 1/4: loading and merging raw data")
    dataset = build_dataset(config)
    interim_path = Path(config["paths"]["interim_dir"]) / "sales_long.parquet"
    write_parquet(dataset, interim_path)

    logger.info("stage 2/4: validating data")
    run_validation(dataset)
    del dataset
    gc.collect()

    logger.info("stage 3/4: building features")
    features = build_feature_table(config)
    processed_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(processed_path)
    del features
    gc.collect()

    logger.info("stage 4/4: training and evaluating models")
    results = run_models(config, model_names)

    return results


if __name__ == "__main__":
    # usage: uv run python -m src.pipeline.run_pipeline
    #        uv run python -m src.pipeline.run_pipeline lightgbm xgboost
    requested = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS

    config = load_config()
    results = run_pipeline(config, requested)

    for name, metrics in results.items():
        logger.info(f"{name}: {metrics}")
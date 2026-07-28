import numpy as np
import polars as pl
from pathlib import Path

from src.evaluation.backtest import train_test_split
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

MONITORED_COLUMNS = ["sales", "sell_price", "sales_lag_7", "sales_roll_mean_7", "sales_roll_std_28"]
PSI_THRESHOLD = 0.2
NUM_BINS = 10

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = NUM_BINS) -> float:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def check_drift(train: pl.DataFrame, test: pl.DataFrame, columns: list[str]) -> dict[str, dict]:
    results = {}
    for col in columns:
        reference = train[col].to_numpy().astype(float)
        current = test[col].to_numpy().astype(float)

        psi = compute_psi(reference, current)
        results[col] = {"psi": psi, "drifted": psi > PSI_THRESHOLD}

    return results

def run_drift_check(config: dict) -> dict:
    features_path = Path(config["paths"]["processed_dir"]) / "features.parquet"
    df = pl.read_parquet(features_path)

    horizon = config["evaluation"]["horizon"]
    train, test = train_test_split(df, horizon)

    results = check_drift(train, test, MONITORED_COLUMNS)

    for col, result in results.items():
        status = "DRIFT DETECTED" if result["drifted"] else "stable"
        logger.info(f"{col}: psi={result['psi']:.4f} ({status})")
 
    return results
 
 
if __name__ == "__main__":
    config = load_config()
    run_drift_check(config)
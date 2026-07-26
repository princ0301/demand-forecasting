from datetime import timedelta

import numpy as np
import polars as pl

from src.evaluation.metrics import mape, rmse, rmsse_from_scale, weighted_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_test_split(df: pl.DataFrame, horizon: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    max_date = df["date"].max()
    cutoff = max_date - timedelta(days=horizon)
    train = df.filter(pl.col("date") <= cutoff)
    test = df.filter(pl.col("date") > cutoff)
    return train, test


def compute_series_weights(train: pl.DataFrame, weight_window: int) -> dict[str, float]:
    # weight by each series' share of total revenue in the last weight_window
    # days of training, so high-volume products matter more to the final score,
    # matching the intent of the official M5 weighting scheme
    max_date = train["date"].max()
    window_start = max_date - timedelta(days=weight_window - 1)

    recent = train.filter(pl.col("date") >= window_start)
    revenue = recent.with_columns((pl.col("sales") * pl.col("sell_price")).alias("revenue"))
    series_revenue = revenue.group_by("id").agg(pl.col("revenue").sum())

    total_revenue = series_revenue["revenue"].sum()
    series_revenue = series_revenue.with_columns((pl.col("revenue") / total_revenue).alias("weight"))

    return dict(zip(series_revenue["id"].to_list(), series_revenue["weight"].to_list()))


def compute_series_scale(train: pl.DataFrame) -> dict[str, float]:
    # scale is the mean squared error of a naive one-step-ahead forecast on
    # each series' training history, this is what makes RMSSE comparable
    # across products with very different sales volumes. computed here as a
    # vectorized Polars aggregation so the full training frame never needs
    # to be duplicated into pandas
    scale_df = (
        train.select(["id", "sales"])
        .with_columns((pl.col("sales") - pl.col("sales").shift(1).over("id")).alias("diff"))
        .with_columns((pl.col("diff") ** 2).alias("sq_diff"))
        .group_by("id")
        .agg(pl.col("sq_diff").mean().alias("scale"))
    )
    return dict(zip(scale_df["id"].to_list(), scale_df["scale"].to_list()))


def evaluate_predictions(
    train: pl.DataFrame,
    test: pl.DataFrame,
    predictions: pl.DataFrame,
    weight_window: int = 28,
) -> dict:
    # predictions must have columns: id, date, prediction
    weights = compute_series_weights(train, weight_window)
    scales = compute_series_scale(train)

    # predictions may come back from a pandas round-trip with plain string ids
    # and datetime-typed dates, while test carries categorical ids and date-typed
    # dates, casting both sides to the same types keeps this join model-agnostic
    test_keys = test.select(["id", "date", "sales"]).with_columns([
        pl.col("id").cast(pl.Utf8),
        pl.col("date").cast(pl.Date),
    ])
    predictions_keys = predictions.with_columns([
        pl.col("id").cast(pl.Utf8),
        pl.col("date").cast(pl.Date),
    ])
    merged = test_keys.join(predictions_keys, on=["id", "date"]).to_pandas()

    rmsse_scores = {}
    rmse_scores = {}
    mape_scores = {}

    for series_id, group in merged.groupby("id"):
        y_true = group["sales"].to_numpy()
        y_pred = group["prediction"].to_numpy()

        rmsse_scores[series_id] = rmsse_from_scale(y_true, y_pred, scales.get(series_id, float("nan")))
        rmse_scores[series_id] = rmse(y_true, y_pred)
        mape_scores[series_id] = mape(y_true, y_pred)

    result = {
        "wrmsse": weighted_score(rmsse_scores, weights),
        "mean_rmse": float(np.nanmean(list(rmse_scores.values()))),
        "mean_mape": float(np.nanmean(list(mape_scores.values()))),
    }
    logger.info(f"evaluation result: {result}")
    return result
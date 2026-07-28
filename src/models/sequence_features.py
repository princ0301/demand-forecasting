import numpy as np
import polars as pl

NUMERIC_FEATURE_ORDER = [
    "sales_log", "price_scaled", "wday_scaled", "month_scaled",
    "day_of_week_scaled", "is_event", "is_weekend", "is_snap",
]

def compute_price_stats(train: pl.DataFrame) -> tuple[float, float]:
    mean = train["sell_price"].mean()
    std = train["sell_price"].std()
    return float(mean), float(std if std and std > 0 else 1.0)

def transform_numeric(df: pl.DataFrame, price_mean: float, price_std: float) -> pl.DataFrame:
    df = df.with_columns(pl.col("sell_price").fill_null(price_mean))
    return df.with_columns([
        pl.col("sales").log1p().alias("sales_log"),
        ((pl.col("sell_price") - price_mean) / price_std).alias("price_scaled"),
        (pl.col("wday") / 7.0).alias("wday_scaled"),
        (pl.col("month") / 12.0).alias("month_scaled"),
        (pl.col("day_of_week") / 7.0).alias("day_of_week_scaled"),
        pl.col("is_event").cast(pl.Float32),
        pl.col("is_weekend").cast(pl.Float32),
        pl.col("is_snap").cast(pl.Float32),
    ])

def build_series_arrays(
    df: pl.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # aggregates into one row per series before converting to pandas, so the
    # full long-format frame is never duplicated, only a table with as many
    # rows as there are series
    grouped = df.group_by("id", maintain_order=True).agg(
        [pl.col(c) for c in numeric_cols] + [pl.col(c).first() for c in categorical_cols]
    )
    grouped_pd = grouped.to_pandas()

    series_arrays = {}
    series_static = {}
    for row in grouped_pd.itertuples(index=False):
        series_id = row.id
        numeric_matrix = np.column_stack([np.array(getattr(row, c), dtype=np.float32) for c in numeric_cols])
        series_arrays[series_id] = numeric_matrix
        series_static[series_id] = np.array([getattr(row, c) for c in categorical_cols], dtype=np.int64)

    return series_arrays, series_static

def build_last_window(df: pl.DataFrame, numeric_cols: list[str], lookback: int) -> dict[str, np.ndarray]:
    tail = df.select(["id"] + numeric_cols).group_by("id", maintain_order=True).tail(lookback)
    grouped = tail.group_by("id", maintain_order=True).agg([pl.col(c) for c in numeric_cols])
    grouped_pd = grouped.to_pandas()

    windows = {}
    for row in grouped_pd.itertuples(index=False):
        windows[row.id] = np.column_stack([np.array(getattr(row, c), dtype=np.float32) for c in numeric_cols])
    return windows
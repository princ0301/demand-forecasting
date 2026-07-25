import polars as pl

def add_rolling_features(df: pl.DataFrame, windows: list[int]) -> pl.DataFrame:
    # rolling stats are computed on sales shifted by 1 day so the current day's
    # own sales value is never part of its own feature, this avoids leakage
    shifted = pl.col("sales").shift(1).over("id")

    rolling_exprs = []
    for window in windows:
        rolling_exprs.append(
            shifted.rolling_mean(window_size=window, min_periods=1).over("id").alias(f"sales_roll_mean_{window}")
        )
        rolling_exprs.append(
            shifted.rolling_std(window_size=window, min_periods=1).over("id").alias(f"sales_roll_std_{window}")
        )

    return df.with_columns(rolling_exprs)
import polars as pl

def add_lag_features(df: pl.DataFrame, lags: list[int]) -> pl.DataFrame:
    lag_exprs = [
        pl.col("sales").shift(lag).over("id").alias(f"sales_lag_{lag}")
        for lag in lags
    ]
    return df.with_columns(lag_exprs)
import polars as pl


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("event_name_1").is_not_null().cast(pl.Int8).alias("is_event"),
    ])

    df = df.with_columns([
        (pl.col("day_of_week") >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.when(pl.col("state_id") == "CA").then(pl.col("snap_CA"))
          .when(pl.col("state_id") == "TX").then(pl.col("snap_TX"))
          .when(pl.col("state_id") == "WI").then(pl.col("snap_WI"))
          .otherwise(0)
          .alias("is_snap"),
    ])

    return df.drop(["event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"])
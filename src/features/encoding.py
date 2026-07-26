import polars as pl

def build_category_maps(df: pl.DataFrame, cols: list[str]) -> dict[str, dict]:
    maps = {}
    for col in cols:
        uniques = df[col].cast(pl.Utf8).unique().to_list()
        maps[col] = {value: idx for idx, value in enumerate(uniques)}
    return maps


def encode_categoricals(df: pl.DataFrame, maps: dict[str, dict], cols: list[str]) -> pl.DataFrame:
    for col in cols:
        mapping = maps[col]
        mapping_df = pl.DataFrame({
            col: list(mapping.keys()),
            f"{col}_code": list(mapping.values()),
        })
        df = df.with_columns(pl.col(col).cast(pl.Utf8))
        df = df.join(mapping_df, on=col, how="left")
        df = df.drop(col).rename({f"{col}_code": col})
    return df
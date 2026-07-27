from datetime import date, timedelta

import pandas as pd
import polars as pl

FUTURE_FRAME_COLS = [
    "id", "date", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "sell_price", "wday", "month", "year", "day_of_week", "is_event", "is_weekend", "is_snap",
]


def build_wday_map(calendar_reference: pd.DataFrame) -> dict[int, int]:
    # M5's own "wday" column is just a fixed relabeling of day-of-week, this
    # recovers that mapping from real recent dates rather than assuming its
    # convention, so future rows get a consistent value without guessing
    reference = calendar_reference.copy()
    reference["iso_weekday"] = reference["date"].apply(lambda d: d.isoweekday())
    pairs = reference[["iso_weekday", "wday"]].drop_duplicates()
    return dict(zip(pairs["iso_weekday"], pairs["wday"]))


def generate_future_frame(last_known: pd.DataFrame, wday_map: dict[int, int], horizon: int) -> pl.DataFrame:
    rows = []
    for row in last_known.itertuples(index=False):
        last_date = row.date if isinstance(row.date, date) else row.date.date()

        for offset in range(1, horizon + 1):
            future_date = last_date + timedelta(days=offset)
            day_of_week = future_date.isoweekday()

            rows.append({
                "id": row.id,
                "date": future_date,
                "item_id": row.item_id,
                "dept_id": row.dept_id,
                "cat_id": row.cat_id,
                "store_id": row.store_id,
                "state_id": row.state_id,
                "sell_price": row.sell_price,
                "wday": wday_map.get(day_of_week, 1),
                "month": future_date.month,
                "year": future_date.year,
                "day_of_week": day_of_week,
                "is_event": 0,
                "is_weekend": 1 if day_of_week >= 6 else 0,
                "is_snap": 0,
            })

    return pl.from_pandas(pd.DataFrame(rows, columns=FUTURE_FRAME_COLS))
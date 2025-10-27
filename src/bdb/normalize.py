# src/bdb/normalize.py
import polars as pl

FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3

def normalize_left_to_right(tr: pl.DataFrame, plays: pl.DataFrame) -> pl.DataFrame:
    # plays must include 'possessionTeam' and 'playDirection' or similar; fallback to event-based flip if needed.
    if "playDirection" not in tr.columns:
        # heuristic: infer from early frames: increasing x = "right"
        tr = tr.with_columns(
            (pl.col("x").shift(-1) - pl.col("x")).alias("__dx")
        ).with_columns(
            pl.when(pl.col("__dx")>=0).then(pl.lit("right")).otherwise(pl.lit("left")).alias("playDirection_inferred")
        ).drop("__dx")
        dircol = "playDirection_inferred"
    else:
        dircol = "playDirection"

    # flip left-moving plays
    tr = tr.with_columns(
        pl.when(pl.col(dircol)=="left").then(FIELD_X_MAX - pl.col("x")).otherwise(pl.col("x")).alias("x_n"),
        pl.when(pl.col(dircol)=="left").then(FIELD_Y_MAX - pl.col("y")).otherwise(pl.col("y")).alias("y_n"),
    ).with_columns(
        pl.col("x_n").clip(FIELD_X_MIN, FIELD_X_MAX),
        pl.col("y_n").clip(FIELD_Y_MIN, FIELD_Y_MAX),
    )
    return tr

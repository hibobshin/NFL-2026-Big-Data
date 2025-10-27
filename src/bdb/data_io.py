# src/bdb/data_io.py
from pathlib import Path
import polars as pl

RAW = Path("data/raw")
OUT = Path("data/processed")

REQ_TRACK_COLS = {"gameId","playId","nflId","frameId","x","y","s","a","o","dir","event"}

def load_tracking() -> pl.DataFrame:
    files = sorted(RAW.glob("tracking_week*.csv"))
    if not files:
        raise FileNotFoundError("No tracking_week*.csv in data/raw")
    df = pl.concat([pl.read_csv(f) for f in files], how="vertical_relaxed")
    missing = REQ_TRACK_COLS - set(df.columns)
    if missing:
        raise ValueError(f"tracking missing cols: {missing}")
    df = df.with_columns(
        pl.col("gameId").cast(pl.Utf8),
        pl.col("gameId").str.slice(0,8).alias("game_date_str"),
    ).with_columns(
        pl.col("game_date_str").str.strptime(pl.Date, "%Y%m%d").alias("game_date")
    ).drop("game_date_str")
    return df

def load_plays() -> pl.DataFrame:
    return pl.read_csv(RAW/"plays.csv")

def load_players() -> pl.DataFrame:
    return pl.read_csv(RAW/"players.csv")

def write_parquet(name: str, df: pl.DataFrame):
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT/f"{name}.parquet")

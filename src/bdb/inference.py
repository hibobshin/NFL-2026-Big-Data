import os
import math
import pandas as pd
import polars as pl
from typing import Dict, Tuple

import kaggle_evaluation.nfl_inference_server


# Global model/state cache (Kaggle container persists between predict() calls)
MODEL = None
STATE: Dict[Tuple[int, int, int], dict] = {}
READY = False


def _to_pl(df):
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


def _infer_dt(cur_frame: int | None, prev_frame: int | None) -> float:
    if cur_frame is None or prev_frame is None:
        return 1.0
    return max(1.0, float(cur_frame - prev_frame))


def _keyed_ids(test_batch: pl.DataFrame):
    have = set(test_batch.columns)
    g = "gameId" if "gameId" in have else None
    p = "playId" if "playId" in have else None
    n = "nflId" if "nflId" in have else None
    return g, p, n


def _extract_frame_cols(df: pl.DataFrame):
    cols = set(df.columns)
    cur_frame_col = "frameId" if "frameId" in cols else None
    prev_frame_col = "prevFrameId" if "prevFrameId" in cols else None
    dt_col = None
    for cand in ("dt", "delta_t", "frame_dt"):
        if cand in cols:
            dt_col = cand
            break
    return cur_frame_col, prev_frame_col, dt_col


def _load_model_once():
    """Load heavy assets exactly once."""
    global READY, MODEL
    if READY:
        return
    # TODO: load your real model here (torch, sklearn, etc.)
    MODEL = None
    READY = True


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Baseline constant-velocity inference. Modify as you improve your model."""
    _load_model_once()

    tb = _to_pl(test)
    ti = _to_pl(test_input)  # Provided for completeness; not used here.

    gcol, pcol, ncol = _keyed_ids(tb)
    cur_f_col, prev_f_col, dt_col = _extract_frame_cols(tb)

    xs, ys = [], []

    for i in range(tb.height):
        row = tb.row(i, named=True)
        key = (
            row.get(gcol, None) if gcol else None,
            row.get(pcol, None) if pcol else None,
            row.get(ncol, None) if ncol else None,
        )

        x_now, y_now = row.get("x", None), row.get("y", None)

        # Estimate dt
        if dt_col and row.get(dt_col) is not None:
            dt = max(1.0, float(row[dt_col]))
        else:
            curf = row.get(cur_f_col, None)
            prevf = row.get(prev_f_col, None)
            dt = _infer_dt(curf, prevf)

        st = STATE.get(key)
        if st is None:
            vx = vy = 0.0
            x_pred = x_now if x_now is not None else 0.0
            y_pred = y_now if y_now is not None else 0.0
            STATE[key] = {
                "x": x_now,
                "y": y_now,
                "vx": vx,
                "vy": vy,
                "frame": row.get(cur_f_col, None),
            }
        else:
            vx, vy = st["vx"], st["vy"]

            # update observed velocity
            if (
                x_now is not None
                and y_now is not None
                and st["x"] is not None
                and st["y"] is not None
            ):
                denom = max(1.0, float((row.get(cur_f_col, 0) or 0) - (st["frame"] or 0)))
                vx_obs = (x_now - st["x"]) / denom
                vy_obs = (y_now - st["y"]) / denom
                alpha = 0.7
                vx = alpha * vx_obs + (1 - alpha) * vx
                vy = alpha * vy_obs + (1 - alpha) * vy

            x_base = x_now if x_now is not None else st["x"] or 0.0
            y_base = y_now if y_now is not None else st["y"] or 0.0
            x_pred = x_base + vx * dt
            y_pred = y_base + vy * dt

            STATE[key].update(
                {
                    "x": x_now if x_now is not None else x_pred,
                    "y": y_now if y_now is not None else y_pred,
                    "vx": vx,
                    "vy": vy,
                    "frame": row.get(cur_f_col, st["frame"]),
                }
            )

        xs.append(float(x_pred))
        ys.append(float(y_pred))

    return pl.DataFrame({"x": xs, "y": ys})


def get_inference_server():
    """Factory so notebooks stay one-liner clean."""
    return kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)

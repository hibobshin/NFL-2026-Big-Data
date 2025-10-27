# NFL Big Data Bowl 2026 — Prediction (Serving Repo)

## What this repo is
Minimal, submission-ready structure. All serving logic is in `src/bdb/inference.py`. The Kaggle notebook just boots the server and streams inference.

## Dev quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/local_gateway_test.py
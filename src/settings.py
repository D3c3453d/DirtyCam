from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / Path("models")

RAW_DIR = BASE_DIR / "data/raw"
GOOD_DIR = BASE_DIR / "data/good"
BAD_DIR = BASE_DIR / "data/bad"
PREDICT_DIR = BASE_DIR / "data/predict"

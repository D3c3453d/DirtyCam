from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / Path("models")

RAW_DIR = BASE_DIR / "data/raw"
GOOD_DIR = BASE_DIR / "data/good"
BAD_DIR = BASE_DIR / "data/bad"
PREDICT_DIR = BASE_DIR / "data/predict"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d\n%(message)s\n"

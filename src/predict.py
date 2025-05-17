import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from features import FeatureExtractor
from settings import LOG_FORMAT, MODEL_DIR, PREDICT_DIR

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger(__name__)


def load_models(model_dir: Path) -> dict:
    """Load all .joblib models from a directory."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    models = {}
    for file in model_dir.glob("*.joblib"):
        try:
            models[file.stem] = joblib.load(file)
            logger.info(f"Model loaded: {file.name}")
        except Exception as e:
            logger.warning(f"Failed to load model {file.name}: {e}")

    if not models:
        raise RuntimeError(f"No models found in {model_dir}")

    return models


def _predict_image(models: dict, feat_ext: FeatureExtractor, image_path: Path) -> list[dict]:
    logger.info(f"Processing: {image_path.name}")
    features = feat_ext.extract_features(image_path)

    if features is None:
        logger.warning(f"Skipping unreadable image: {image_path.name}")
        return []

    df = pd.DataFrame([features])
    predictions = []

    for name, model in models.items():
        try:
            pred = model.predict(df)[0]
            predictions.append(
                {
                    "file": image_path.name,
                    "model": name,
                    "prediction": int(pred),
                }
            )
        except Exception as e:
            logger.error(f"Model {name} failed to predict {image_path.name}: {e}")

    return predictions


def predict(models: dict, predict_dir: Path) -> pd.DataFrame:
    """Extract features and predict labels using each model."""
    feat_ext = FeatureExtractor()
    image_files = list(predict_dir.glob("*")) if predict_dir.is_dir() else []
    if not image_files:
        raise FileNotFoundError(f"No images found in {predict_dir}")

    all_predictions = []
    for image_path in image_files:
        all_predictions.extend(_predict_image(models, feat_ext, image_path))

    if not all_predictions:
        raise RuntimeError("No predictions were generated.")

    return pd.DataFrame(all_predictions)


def main():
    parser = argparse.ArgumentParser(description="Predict image quality using trained models.")
    parser.add_argument("--predict-dir", default=PREDICT_DIR, help="Directory with images to predict.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory with saved models (.joblib)")
    args = parser.parse_args()

    logger.info("Loading models...")
    models = load_models(Path(args.model_dir))

    logger.info("Starting predictions...")
    results = predict(models, Path(args.predict_dir))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

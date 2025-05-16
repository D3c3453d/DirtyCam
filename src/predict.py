import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from features import FeatureExtractor
from settings import MODEL_DIR, PREDICT_DIR

logging.basicConfig(level=logging.INFO)


def load_models(model_dir: str) -> dict:
    """Load all .joblib models from a directory."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    models = {}
    for file in model_dir.glob("*.joblib"):
        model_name = file.stem
        try:
            models[model_name] = joblib.load(file)
            logging.info(f"Model loaded: {file.name}")
        except Exception as e:
            logging.warning(f"Failed to load model {file.name}: {e}")
    if not models:
        raise RuntimeError(f"No models found in {model_dir}")
    return models


def predict(models: dict, predict_dir: str) -> pd.DataFrame:
    """
    Extract features from images and predict using all models.
    Returns a DataFrame with predictions per image.
    """
    predict_path = Path(predict_dir)
    if not predict_path.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {predict_path}")

    image_files = list(predict_path.glob("*"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {predict_dir}")

    feat_ext = FeatureExtractor()
    rows = []

    for image_path in image_files:
        logging.info(f"Processing: {image_path.name}")
        feat_row = feat_ext.extract_features(image_path)
        if feat_row is None:
            logging.warning(f"Skipping unreadable image: {image_path.name}")
            continue

        df = pd.DataFrame([feat_row])
        for name, model in models.items():
            try:
                pred = model.predict(df)[0]
                rows.append({"file": image_path.name, "model": name, "prediction": int(pred)})
            except Exception as e:
                logging.error(f"Model {name} failed to predict {image_path.name}: {e}")

    if not rows:
        raise RuntimeError("No predictions were generated.")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Predict image quality using trained models.")
    parser.add_argument("--predict-dir", default=PREDICT_DIR, help="Directory with images to predict.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory with saved models (.joblib)")
    args = parser.parse_args()

    logging.info("Loading models...")
    models = load_models(args.model_dir)

    logging.info("Starting predictions...")
    results = predict(models, args.predict_dir)

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

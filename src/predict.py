import argparse
import os

import cv2
import joblib
import pandas as pd
from features import FeatureExtractor


def load_models(model_dir: str) -> dict:
    """Load all .pkl models from the directory."""
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".joblib"):
            name = os.path.splitext(fname)[0]
            models[name] = joblib.load(os.path.join(model_dir, fname))
    return models


def predict(models: dict, img_paths: list) -> pd.DataFrame:
    """
    Run prediction over a list of image paths using loaded models.
    """
    rows = []
    feat_ext = FeatureExtractor()
    for path in img_paths:
        img = cv2.imread(path)
        feats = feat_ext.compute_features(img)
        df = pd.DataFrame([feats], columns=feat_ext.columns)
        for name, model in models.items():
            pred = model.predict(df)[0]
            rows.append({"file": path, "model": name, "prediction": int(pred)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imgs", nargs="+", help="Paths to images to predict")
    parser.add_argument("--model-dir", default="../models")
    args = parser.parse_args()

    models = load_models(args.model_dir)
    results = predict(models, args.imgs)
    print(results.to_string(index=False))

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from features import FeatureExtractor
from settings import BAD_DIR, GOOD_DIR, MODEL_DIR, RAW_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)

# Dynamic model constructors
MODEL_CONSTRUCTORS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "KNN": lambda: KNeighborsClassifier(),
    "DecisionTree": lambda: DecisionTreeClassifier(),
    "CatBoost": lambda: CatBoostClassifier(verbose=0),
}


def build_dataframe(raw_dir: str, good_dir: str, bad_dir: str):
    raw_dir, good_dir, bad_dir = Path(raw_dir), Path(good_dir), Path(bad_dir)
    raw_files = list(raw_dir.glob("*")) if raw_dir.is_dir() else []
    good_files = list(good_dir.glob("*")) if good_dir.is_dir() else []
    bad_files = list(bad_dir.glob("*")) if bad_dir.is_dir() else []

    has_raw, has_good, has_bad = bool(raw_files), bool(good_files), bool(bad_files)
    feat_ext = FeatureExtractor()
    data = []

    def process_file(path: Path, label: int):
        feat_row = feat_ext.extract_features(path)
        if feat_row:
            return {"file": path.name, **feat_row, "label": label}
        logging.warning(f"Skipping unreadable image: {path.name}")
        return None

    if has_raw:
        if has_good and not has_bad:
            good_names = {f.name for f in good_files}
            for path in raw_files:
                label = int(path.name in good_names)
                row = process_file(path, label)
                if row:
                    data.append(row)
        elif has_bad and not has_good:
            bad_names = {f.name for f in bad_files}
            for path in raw_files:
                label = int(path.name not in bad_names)
                row = process_file(path, label)
                if row:
                    data.append(row)
        elif not has_good and not has_bad:
            raise FileNotFoundError("Missing both good and bad directories for labeling raw data.")
        else:
            raise FileExistsError("Too many directories populated. Provide any 2 of: raw, good, bad.")
    else:
        if not (has_good and has_bad):
            raise FileNotFoundError("Missing both raw and good/bad directories.")
        for path in good_files:
            row = process_file(path, 1)
            if row:
                data.append(row)
        for path in bad_files:
            row = process_file(path, 0)
            if row:
                data.append(row)

    df = pd.DataFrame(data)
    X = df[feat_ext.columns]
    y = df["label"]
    return X, y


def train_and_save(X: pd.DataFrame, y: pd.Series, model_dir: str, n_splits: int = 5) -> pd.DataFrame:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics = {name: {"precision": [], "recall": []} for name in MODEL_CONSTRUCTORS}

    logging.info("Starting cross-validation...")

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, ctor in MODEL_CONSTRUCTORS.items():
            model = ctor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_metrics[name]["precision"].append(precision_score(y_test, preds))
            cv_metrics[name]["recall"].append(recall_score(y_test, preds))

    results = []
    for name, scores in cv_metrics.items():
        precision = np.mean(scores["precision"])
        recall = np.mean(scores["recall"])
        results.append({"model": name, "precision": precision, "recall": recall})
        logging.info(f"{name}: Precision={precision:.3f}, Recall={recall:.3f}")

    cv_df = pd.DataFrame(results)

    logging.info("Training final models and saving...")
    for name, ctor in MODEL_CONSTRUCTORS.items():
        model = ctor()
        model.fit(X, y)
        model_path = model_dir / f"{name}.joblib"
        joblib.dump(model, model_path, compress=True)
        logging.info(f"Saved model: {model_path.name}")

    return cv_df


def main():
    parser = argparse.ArgumentParser(description="Train models on image features with optional raw/good/bad input.")
    parser.add_argument("--raw-dir", default=RAW_DIR, help="Directory of unlabeled raw images.")
    parser.add_argument("--good-dir", default=GOOD_DIR, help="Directory of labeled 'good' images.")
    parser.add_argument("--bad-dir", default=BAD_DIR, help="Directory of labeled 'bad' images.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory to save trained models.")
    args = parser.parse_args()

    logging.info("Building feature dataset...")
    X, y = build_dataframe(args.raw_dir, args.good_dir, args.bad_dir)

    logging.info("Training models...")
    train_and_save(X, y, args.model_dir)


if __name__ == "__main__":
    main()

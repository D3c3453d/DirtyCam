import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from features import FeatureExtractor
from settings import BAD_DIR, GOOD_DIR, LOG_FORMAT, MODEL_DIR, RAW_DIR
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger(__name__)

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "KNN": lambda: KNeighborsClassifier(),
    "DecisionTree": lambda: DecisionTreeClassifier(),
    "CatBoost": lambda: CatBoostClassifier(verbose=0),
}


def _build_label_list(raw_dir: Path, good_dir: Path, bad_dir: Path) -> list[tuple[Path, int]]:
    raw_files, good_files, bad_files = (list(d.glob("*")) if d.is_dir() else [] for d in (raw_dir, good_dir, bad_dir))

    if raw_files:
        if good_files and not bad_files:
            good_names = {f.name for f in good_files}
            return [(p, int(p.name in good_names)) for p in raw_files]
        elif bad_files and not good_files:
            bad_names = {f.name for f in bad_files}
            return [(p, int(p.name not in bad_names)) for p in raw_files]
        elif not (good_files or bad_files):
            raise FileNotFoundError("Missing both good and bad dirs for raw labeling.")
        else:
            raise FileExistsError("Too many label dirs. Provide any 2 of: raw, good, bad.")
    else:
        if not (good_files and bad_files):
            raise FileNotFoundError("Missing both raw and labeled directories.")
        return [(p, 1) for p in good_files] + [(p, 0) for p in bad_files]


def _extract_features_parallel(files_to_process: list[tuple[Path, int]], feat_ext: FeatureExtractor) -> list[dict]:
    total_count = len(files_to_process)
    processed_count = 0
    data = []

    def process(path: Path, label: int):
        feats = feat_ext.extract_features(path)
        if feats:
            return {"file": path.name, **feats, "label": label}
        logger.warning(f"Unreadable: {path.name}")
        return None

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process, path, label) for path, label in files_to_process]
        for f in as_completed(futures):
            result = f.result()
            if result:
                data.append(result)
            processed_count += 1
            if processed_count % 10 == 0:
                logger.info(f"Processed: {processed_count}/{total_count}")

    return data


def build_dataframe(raw_dir: Path, good_dir: Path, bad_dir: Path) -> tuple[pd.DataFrame]:
    feat_ext = FeatureExtractor()
    files_to_process = _build_label_list(raw_dir, good_dir, bad_dir)
    data = _extract_features_parallel(files_to_process, feat_ext)

    df = pd.DataFrame(data)
    logger.info(df)
    return df[feat_ext.columns], df["label"]


def train_and_save(X: pd.DataFrame, y: pd.Series, model_dir: str, n_splits: int = 5) -> pd.DataFrame:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {name: {"precision": [], "recall": []} for name in MODELS}

    logger.info("Starting cross-validation...")

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, ctor in MODELS.items():
            model = ctor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            scores[name]["precision"].append(precision_score(y_test, preds))
            scores[name]["recall"].append(recall_score(y_test, preds))

    results = [
        {"model": name, "precision": np.mean(vals["precision"]), "recall": np.mean(vals["recall"])}
        for name, vals in scores.items()
    ]

    logger.info("Training final models and saving...")
    for name, ctor in MODELS.items():
        model = ctor()
        model.fit(X, y)
        path = model_dir / f"{name}.joblib"
        joblib.dump(model, path, compress=True)
        logger.info(f"Saved model: {path.name}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Train models on image features.")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--good-dir", default=GOOD_DIR)
    parser.add_argument("--bad-dir", default=BAD_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    logger.info("Extracting features...")
    X, y = build_dataframe(Path(args.raw_dir), Path(args.good_dir), Path(args.bad_dir))

    logger.info("Training models...")
    results = train_and_save(X, y, args.model_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

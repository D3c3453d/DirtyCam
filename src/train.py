import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _process_file(args: tuple[Path, int]) -> dict | None:
    path, label = args
    try:
        feat_ext = FeatureExtractor()
        feats = feat_ext.extract_features(path)
        if feats:
            return {"file": path.name, **feats, "label": label}
        logger.warning(f"Unreadable: {path.name}")
    except Exception as e:
        logger.error(f"Error processing {path}: {str(e)}")
    return None


def _extract_features_parallel(files_to_process: list[tuple[Path, int]]) -> list[dict]:
    total_count = len(files_to_process)
    processed_count = 0
    data = []

    # Use spawn context for ProcessPoolExecutor
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as executor:
        futures = [executor.submit(_process_file, (path, label)) for path, label in files_to_process]
        for future in as_completed(futures):
            result = future.result()
            if result:
                data.append(result)
            processed_count += 1
            if processed_count % 10 == 0:
                logger.info(f"Processed: {processed_count}/{total_count}")

    return data


def build_dataframe(raw_dir: Path, good_dir: Path, bad_dir: Path) -> tuple[pd.DataFrame]:
    files_to_process = _build_label_list(raw_dir, good_dir, bad_dir)
    data = _extract_features_parallel(files_to_process)

    if not data:
        logger.error("No features extracted. Check input data and logs.")
        return pd.DataFrame(), pd.Series()

    df = pd.DataFrame(data)
    feat_ext = FeatureExtractor()
    logger.info(df.head())
    return df[feat_ext.columns], df["label"]


def train_and_save(X: pd.DataFrame, y: pd.Series, model_dir: str, n_splits: int = 5) -> pd.DataFrame:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if X.empty or y.empty:
        logger.error("No data to train on. Exiting.")
        return pd.DataFrame()

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
            scores[name]["precision"].append(precision_score(y_test, preds, zero_division=0))
            scores[name]["recall"].append(recall_score(y_test, preds, zero_division=0))

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
    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train models on image features.")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--good-dir", default=GOOD_DIR)
    parser.add_argument("--bad-dir", default=BAD_DIR)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()

    logger.info("Extracting features...")
    X, y = build_dataframe(Path(args.raw_dir), Path(args.good_dir), Path(args.bad_dir))

    if X.empty:
        logger.error("No features extracted. Exiting.")
        return

    logger.info("Training models...")
    results = train_and_save(X, y, args.model_dir)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

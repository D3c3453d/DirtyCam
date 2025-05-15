import argparse
import os

import cv2
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from features import FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Define model constructors using lambda functions, so that each call returns a new instance
MODEL_CONSTRUCTORS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "KNN": lambda: KNeighborsClassifier(),
    "Decision Tree": lambda: DecisionTreeClassifier(),
    "CatBoost": lambda: CatBoostClassifier(verbose=0),
}


def train_and_save(X: pd.DataFrame, y: pd.Series, model_dir: str, n_splits: int = 5) -> pd.DataFrame:
    """
    Perform K-fold CV, print average scores, train on full data, and save models.
    Returns DataFrame of CV results.
    """
    os.makedirs(model_dir, exist_ok=True)
    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_metrics = {name: {"precision": [], "recall": []} for name in MODEL_CONSTRUCTORS}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train each
        for name, ctor in MODEL_CONSTRUCTORS.items():
            model = ctor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_metrics[name]["precision"].append(precision_score(y_test, preds))
            cv_metrics[name]["recall"].append(recall_score(y_test, preds))

    # Aggregate CV results
    results = []
    for name, m in cv_metrics.items():
        results.append(
            {"model": name, "precision": float(np.mean(m["precision"])), "recall": float(np.mean(m["recall"]))}
        )
    cv_df = pd.DataFrame(results)
    print("Cross-validation results:\n", cv_df)

    # Train final models and save
    for name, ctor in MODEL_CONSTRUCTORS.items():
        model = ctor()
        model.fit(X, y)
        path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(model, path, compress=True)
        print(f"Saved model '{name}' to {path}")

    return cv_df


def build_dataframe(all_dir: str, focus_dir: str):
    files = sorted(f for f in os.listdir(all_dir) if f.endswith(".jpg"))
    data = []
    feat_ext = FeatureExtractor()
    for fname in files:
        img = cv2.imread(os.path.join(all_dir, fname))
        print(f"{fname} read")
        feats = feat_ext.compute_features(img)
        print(f"{fname} feats calculated")
        label = int(fname in os.listdir(focus_dir))
        data.append({"file": fname, **dict(zip(feat_ext.columns, feats)), "label": label})
        print(f"{fname} data saved")
    df = pd.DataFrame(data)
    X = df[feat_ext.columns]
    y = df["label"]
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-frames", required=True)
    parser.add_argument("--focus-frames", required=True)
    parser.add_argument("--model-dir", default="../models")
    args = parser.parse_args()
    X, y = build_dataframe(args.all_frames, args.focus_frames)
    train_and_save(X, y, args.model_dir)

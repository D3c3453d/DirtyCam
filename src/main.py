import os

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from methods import brenner_gradient, laplacian, sobel_variance, tenengrad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FEAT_COLUMNS = [f.__name__ for f in [brenner_gradient, laplacian, sobel_variance, tenengrad]]

# Define model constructors using lambda functions, so that each call returns a new instance
MODEL_CONSTRUCTORS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "KNN": lambda: KNeighborsClassifier(),
    "Decision Tree": lambda: DecisionTreeClassifier(),
    "CatBoost": lambda: CatBoostClassifier(verbose=0),
}


def extract_frames(video_path: str, output_folder: str, frame_interval: int = 1):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_folder, exist_ok=True)
    frame_count = 0
    saved_count = 0
    if cap.isOpened():
        print("Video file successfully retrieved\n")
    else:
        print("Video file wasn't retrieved properly\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/{saved_count:04d}.jpg", frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Extracted {saved_count}/{frame_count} frames to {output_folder}\n")


def compute_features(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return [
        brenner_gradient(gray),
        sobel_variance(gray),
        tenengrad(gray),
        laplacian(gray),
    ]


def create_dataframe(image_folder: str):
    features = []
    files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    files.sort()
    for file in files:
        image = cv2.imread(f"{image_folder}/{file}")
        features.append([file] + compute_features(image))
    return pd.DataFrame(features, columns=["file"] + FEAT_COLUMNS)


def label_dataframe(df: pd.DataFrame, image_folder: str):
    files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    df["label"] = (df["file"].isin(files)).astype(int)
    return df


def train_models(X_train, y_train):
    models = {name: constructor() for name, constructor in MODEL_CONSTRUCTORS.items()}
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate(models: dict, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append(
            {"Model": name, "Precision": precision_score(y_test, y_pred), "Recall": recall_score(y_test, y_pred)}
        )
    return pd.DataFrame(results)


def cross_validate_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {name: {"Precision": [], "Recall": []} for name in MODEL_CONSTRUCTORS.keys()}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_models = train_models(X_train, y_train)
        fold_eval = evaluate(fold_models, X_test, y_test)

        for _, row in fold_eval.iterrows():
            cv_results[row["Model"]]["Precision"].append(row["Precision"])
            cv_results[row["Model"]]["Recall"].append(row["Recall"])

    avg_results = []
    for model_name, metrics in cv_results.items():
        avg_precision = np.mean(metrics["Precision"])
        avg_recall = np.mean(metrics["Recall"])
        avg_results.append({"Model": model_name, "Precision": avg_precision, "Recall": avg_recall})

    return pd.DataFrame(avg_results)


def predict_image(image_path: str, models: dict):
    results = []
    image = cv2.imread(image_path)
    features = pd.DataFrame([compute_features(image)], columns=FEAT_COLUMNS)
    for name, model in models.items():
        results.append({"Model": name, "Prediction": model.predict(features)[0]})
    return pd.DataFrame(results)


if __name__ == "__main__":
    video_path = "./autofocus1.mp4"
    all_folder = "./frames/all/"
    focus_folder = "./frames/focus/"
    frame_interval = 7
    pd.set_option("display.max_rows", None)

    # Prepare
    extract_frames(video_path, all_folder, frame_interval)
    df = create_dataframe(all_folder)
    df = label_dataframe(df, focus_folder)
    print(f"Data\n{df}\n")

    X = df[FEAT_COLUMNS]
    y = df["label"]

    # K-fold cross-validation
    cv_results = cross_validate_models(X, y, n_splits=5)
    print(f"K-fold Cross Validation Results\n{cv_results}\n")

    # Train final models on the entire dataset
    final_models = train_models(X, y)
    prediction = predict_image("./maxim-bogdanov-wjAR4jo979Y-unsplash.jpg", final_models)
    print(f"Prediction\n{prediction}\n")

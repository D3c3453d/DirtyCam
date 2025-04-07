import os

import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from methods import brenner_gradient, laplacian, sobel_variance, tenengrad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

FEAT_COLUMNS = [f.__name__ for f in [brenner_gradient, laplacian, sobel_variance, tenengrad]]


def extract_frames(video_path: str, output_folder: str, frame_interval: int = 1):
    cap = cv2.VideoCapture(f"{video_path}")
    os.makedirs(f"{output_folder}", exist_ok=True)
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
    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0),
    }
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train-evaluate
    models = train_models(X_train, y_train)
    results = evaluate(models, X_test, y_test)
    print(f"Train results\n{results}\n")

    # Predict
    results = predict_image("./maxim-bogdanov-wjAR4jo979Y-unsplash.jpg", models)
    print(f"Prediction\n{results}\n")

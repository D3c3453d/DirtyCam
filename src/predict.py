import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
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


def _ensemble_predict(models: dict, features: dict) -> dict:
    """Make predictions with all models and ensemble results."""
    df = pd.DataFrame([features])
    predictions = []
    probabilities = []

    for name, model in models.items():
        try:
            # Получаем вероятность класса 1 (качественное изображение)
            proba = model.predict_proba(df)[0][1]
            pred = 1 if proba > 0.5 else 0

            predictions.append(pred)
            probabilities.append(proba)
        except AttributeError:
            # Если модель не поддерживает predict_proba
            pred = model.predict(df)[0]
            predictions.append(pred)
            probabilities.append(pred)  # Используем 0/1 как вероятность

    # Ансамблирование: средняя вероятность + порог 0.5
    avg_probability = np.mean(probabilities)
    ensemble_pred = 1 if avg_probability > 0.5 else 0

    # Ансамблирование: большинство голосов
    majority_vote = 1 if sum(predictions) > len(models) / 2 else 0

    return {
        "avg_probability": avg_probability,
        "ensemble_pred": ensemble_pred,
        "majority_vote": majority_vote,
        "individual": dict(zip(models.keys(), predictions)),
    }


def predict_video(models: dict, video_source: str, output_file: str = None, max_frames: int = None):
    """Process video stream in real-time with ensemble prediction."""
    feat_ext = FeatureExtractor()
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    # Открываем видео источник (файл или камеру)
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))  # Веб-камера
    else:
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)  # Видео файл

    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")

    # Настройка вывода видео
    if output_file:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    processing_times = []
    cv2.namedWindow("Video Quality Assessment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Quality Assessment", 1600, 1000)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        start_time = time.time()

        # Преобразуем кадр в серый для извлечения признаков
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            # Извлекаем признаки из кадра
            features = feat_ext.extract_features(gray_frame)

            if features:
                # Получаем предсказания для кадра
                predictions = _ensemble_predict(models, features)

                # Отображаем результаты на кадре
                label = f"Quality: {'Good' if predictions['ensemble_pred'] else 'Bad'}"
                prob = f"Prob: {predictions['avg_probability']:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if predictions["ensemble_pred"] else (0, 0, 255),
                    2,
                )
                cv2.putText(frame, prob, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            logger.error(f"Error processing frame {frame_count}: {e}")

        # Рассчитываем FPS обработки
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        fps = 1 / processing_time if processing_time > 0 else 0

        cv2.putText(frame, f"Proc FPS: {fps:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Показываем кадр
        cv2.imshow("Video Quality Assessment", frame)

        # Сохраняем кадр при необходимости
        if output_file:
            out.write(frame)

        # Прерывание по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Рассчет статистики производительности
    if processing_times:
        avg_fps = 1 / np.mean(processing_times)
        logger.info(f"Processed {frame_count} frames. Average FPS: {avg_fps:.1f}")

    cap.release()
    if output_file:
        out.release()
    cv2.destroyAllWindows()


def predict_images(models: dict, predict_dir: Path):
    """Predict quality for images in directory (original functionality)."""
    feat_ext = FeatureExtractor()
    image_files = list(predict_dir.glob("*")) if predict_dir.is_dir() else []
    if not image_files:
        raise FileNotFoundError(f"No images found in {predict_dir}")

    results = []
    for image_path in image_files:
        logger.info(f"Processing: {image_path.name}")
        features = feat_ext.extract_features_path(image_path)

        if not features:
            logger.warning(f"Skipping unreadable image: {image_path.name}")
            continue

        try:
            predictions = _ensemble_predict(models, features)
            results.append(
                {
                    "file": image_path.name,
                    "avg_probability": predictions["avg_probability"],
                    "ensemble_pred": predictions["ensemble_pred"],
                    "majority_vote": predictions["majority_vote"],
                    **predictions["individual"],
                }
            )
        except Exception as e:
            logger.error(f"Prediction failed for {image_path.name}: {e}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Predict image quality using trained models.")
    parser.add_argument("--video", help="Video file path or camera index (e.g., 0 for webcam).")
    parser.add_argument("--output", help="Output video file path.")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process.")
    parser.add_argument("--predict-dir", default=PREDICT_DIR, help="Directory with images to predict.")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="Directory with saved models (.joblib)")
    args = parser.parse_args()

    logger.info("Loading models...")
    models = load_models(Path(args.model_dir))

    if args.video:
        logger.info("Processing video stream...")
        predict_video(models, args.video, args.output, args.max_frames)
    else:
        logger.info("Processing images...")
        results = predict_images(models, Path(args.predict_dir))
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()

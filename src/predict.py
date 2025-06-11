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

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_models(model_dir: Path) -> dict:
    """
    Загружает все модели .joblib из директории.
    Возвращает словарь name->модель.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    models = {}
    for file in model_dir.glob("*.joblib"):
        try:
            model = joblib.load(file)
            models[file.stem] = model
            logger.info(f"Model loaded: {file.name}")
        except Exception as e:
            logger.warning(f"Failed to load model {file.name}: {e}")
    if not models:
        raise RuntimeError(f"No .joblib models found in {model_dir}")
    return models


def ensemble_predict(models: dict, features: dict[str, float]) -> dict:
    """
    Делаем предсказания всеми моделями, возвращаем:
      - avg_probability: средняя вероятность класса 1 (если есть predict_proba; иначе 0/1)
      - ensemble_pred: 1 если avg_probability > 0.5, иначе 0
      - majority_vote: 1 если большинство моделей предсказало 1
      - individual: словарь name->pred (0/1)
    """
    df = pd.DataFrame([features])
    probs = []
    preds = []

    for name, model in models.items():
        try:
            proba = float(model.predict_proba(df)[0][1])
            pred = 1 if proba > 0.5 else 0
        except (AttributeError, IndexError):
            # Нет predict_proba или неожиданный формат
            pred = int(model.predict(df)[0])
            proba = float(pred)
        preds.append(pred)
        probs.append(proba)

    avg_prob = float(np.mean(probs))
    ensemble_pred = 1 if avg_prob > 0.5 else 0
    majority_vote = 1 if sum(preds) > len(models) / 2 else 0

    return {
        "avg_probability": avg_prob,
        "ensemble_pred": ensemble_pred,
        "majority_vote": majority_vote,
        "individual": dict(zip(models.keys(), preds)),
    }


class Predictor:
    def __init__(self, models: dict, show_window: bool = True, window_name: str = "QualityAssessment"):
        self.models = models
        self.feat_ext = FeatureExtractor()
        self.show_window = show_window
        self.window_name = window_name

    def _draw_overlay(self, frame: np.ndarray, predictions: dict) -> None:
        """
        Рисует на кадре метки качества и FPS.
        """
        ensemble = predictions["ensemble_pred"]
        avg_prob = predictions["avg_probability"]

        label_text = "Quality: Good" if ensemble else "Quality: Bad"
        color = (0, 255, 0) if ensemble else (0, 0, 255)
        prob_text = f"Prob: {avg_prob:.2f}"

        # Тут можно настроить позиции/шрифт/толщину по вкусу
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, prob_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def predict_image(self, image_path: Path) -> dict | None:
        """
        Предсказание по одному изображению.
        Возвращает словарь результатов или None, если не удалось прочитать.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Cannot read image: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            features = self.feat_ext.extract_features(gray)
            if not features:
                logger.warning(f"No features extracted for {image_path}")
                return None
            preds = ensemble_predict(self.models, features)
            result = {
                "file": image_path.name,
                **preds,
                **preds.get("individual", {}),
            }
            return result
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

    def predict_images_in_dir(self, predict_dir: Path) -> pd.DataFrame:
        """
        Проходит по всем файлам в директории, предсказывает.
        Возвращает DataFrame с колонками: file, avg_probability, ensemble_pred, majority_vote, <по моделям>.
        """
        if not predict_dir.is_dir():
            raise FileNotFoundError(f"No such directory: {predict_dir}")
        image_files = sorted(predict_dir.iterdir())
        if not image_files:
            raise FileNotFoundError(f"No files found in {predict_dir}")
        results = []
        for path in image_files:
            logger.info(f"Processing image: {path.name}")
            res = self.predict_image(path)
            if res:
                results.append(res)
        if results:
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame(columns=["file", "avg_probability", "ensemble_pred", "majority_vote"])
        return df

    def predict_video(  # NOQA: C901
        self,
        video_source: str,
        output_file: str = None,
        max_frames: int = None,
        target_fps_display: float = 30.0,
    ) -> None:
        """
        Предсказание на видеопотоке или файле.
        video_source: либо строка с путём, либо индекс камеры ("0", "1", ...).
        output_file: если указан, сохраняем аннотированное видео туда.
        max_frames: при необходимости ограничить число кадров.
        target_fps_display: задержка между кадрами для отображения (в FPS).
        """
        # Открываем источник
        if video_source.isdigit():
            cap = cv2.VideoCapture(int(video_source))
        else:
            if video_source.startswith("rtsp"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
            cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")

        out_writer = None
        if output_file:
            # Параметры видео для записи
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_in = cap.get(cv2.CAP_PROP_FPS) or target_fps_display
            # Используем mp4v или тот же кодек
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(output_file, fourcc, fps_in, (width, height))
            if not out_writer.isOpened():
                logger.warning(f"Cannot open VideoWriter for {output_file}, skipping save.")

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Можно подогнать размер окна при желании
            cv2.resizeWindow(self.window_name, 1600, 1000)

        frame_count = 0
        processing_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream or cannot read frame.")
                break

            frame_count += 1
            if max_frames is not None and frame_count > max_frames:
                logger.info(f"Reached max frames: {max_frames}. Stopping.")
                break

            start = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                features = self.feat_ext.extract_features(gray)
                if features:
                    preds = ensemble_predict(self.models, features)
                    self._draw_overlay(frame, preds)
                else:
                    logger.debug(f"No features for frame {frame_count}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")

            elapsed = time.time() - start
            processing_times.append(elapsed)
            fps_proc = 1.0 / elapsed if elapsed > 0 else 0.0
            # Рисуем FPS обработки
            cv2.putText(frame, f"Proc FPS: {fps_proc:.1f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Показать окно
            if self.show_window:
                cv2.imshow(self.window_name, frame)
            # Сохранить кадр
            if out_writer:
                out_writer.write(frame)

            # Обработка нажатия клавиш
            # Пауза по 'p', выход по 'q' или ESC
            key = cv2.waitKey(int(1000 / target_fps_display)) & 0xFF
            if key == ord("q") or key == 27:  # ESC
                logger.info("User requested exit.")
                break
            elif key == ord("p"):
                logger.info("Paused. Press any key to resume.")
                cv2.waitKey(0)

        # Вывод средней производительности
        if processing_times:
            avg_fps = 1.0 / np.mean(processing_times)
            logger.info(f"Processed {frame_count} frames. Average processing FPS: {avg_fps:.1f}")

        cap.release()
        if out_writer:
            out_writer.release()
        if self.show_window:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Predict quality using trained models.")
    parser.add_argument("--video", type=str, help="Video file path or camera index (e.g., '0' for webcam).")
    parser.add_argument("--output", type=str, help="Output video file path (e.g., annotated video).")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process in video.")
    parser.add_argument("--predict-dir", type=Path, default=Path(PREDICT_DIR), help="Directory with images to predict.")
    parser.add_argument(
        "--model-dir", type=Path, default=Path(MODEL_DIR), help="Directory with saved models (.joblib)."
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable showing OpenCV window during video prediction."
    )
    args = parser.parse_args()

    # Проверяем директории
    model_dir = args.model_dir
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        return
    try:
        models = load_models(model_dir)
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return

    predictor = Predictor(models, show_window=not args.no_display)

    if args.video:
        logger.info("Starting video prediction...")
        try:
            predictor.predict_video(video_source=args.video, output_file=args.output, max_frames=args.max_frames)
        except Exception as e:
            logger.error(f"Video prediction failed: {e}")
    else:
        predict_dir = args.predict_dir
        logger.info(f"Starting image prediction in dir: {predict_dir}")
        try:
            df = predictor.predict_images_in_dir(predict_dir)
            if not df.empty:
                # Вывести в виде таблицы
                print(df.to_string(index=False))
                # Можно сохранить CSV:
                # csv_path = predict_dir / "prediction_results.csv"
                # df.to_csv(csv_path, index=False)
                # logger.info(f"Results saved to {csv_path}")
            else:
                logger.info("No valid images were processed.")
        except Exception as e:
            logger.error(f"Image prediction failed: {e}")


if __name__ == "__main__":
    main()

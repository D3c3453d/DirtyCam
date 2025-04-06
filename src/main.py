import os

import cv2
import methods
import pandas as pd


def extract_frames(video_path: str, output_folder: str, frame_interval: int = 1):
    cap = cv2.VideoCapture(f"{video_path}")
    os.makedirs(f"{output_folder}", exist_ok=True)
    frame_count = 0
    saved_count = 0
    if cap.isOpened():
        print("Video file successfully retrieved")
    else:
        print("Video file wasn't retrieved properly")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{output_folder}/{saved_count:04d}.jpg", frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Extracted {saved_count}/{frame_count} frames to {output_folder}")


def create_dataframe(image_folder):
    features = []
    files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    files.sort()
    for file in files:
        image = cv2.imread(f"{image_folder}/{file}")
        features.append(
            [
                file,
                methods.compute_brenner_gradient(image),
                methods.compute_sobel_variance(image),
                methods.compute_tenengrad(image),
                methods.compute_laplacian(image),
            ]
        )
    return pd.DataFrame(features, columns=["file", "brenner", "sobel", "tenengrad", "laplacian"])


if __name__ == "__main__":
    video_path = "autofocus1.mp4"
    image_folder = "./frames/all/"
    result_path = "result.json"
    frame_interval = 7

    extract_frames(video_path, image_folder, frame_interval)
    df = create_dataframe(image_folder)
    print(df.to_string())

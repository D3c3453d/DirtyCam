import os

import cv2
import pandas as pd
from features import FeatureExtractor
from features0 import FeatureExtractor0


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


feat_ext = FeatureExtractor()
feat_ext_0 = FeatureExtractor0()
img_path = "./data/good/0_IPHONE-SE_S.JPG"

data = [
    {"name": "current", **feat_ext.extract_features_path(img_path)},
    {"name": "original", **feat_ext_0.extract_features_path(img_path)},
]

df = pd.DataFrame(data)
print(df)

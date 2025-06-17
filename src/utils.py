import os
import time

import cv2
import pandas as pd
from features import FeatureExtractor
from features0 import FeatureExtractor0
from settings import PREDICT_DIR


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
img_path = "./data/predict/0050.jpg"

# data = [
#     {"name": "current", **feat_ext.extract_features_path(img_path)},
#     {"name": "original", **feat_ext_0.extract_features_path(img_path)},
# ]

data = []
resize_list = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (640, 360)]
for file in list(PREDICT_DIR.glob("*"))[:10]:
    print(file)
    orig_img = cv2.imread(str(file))
    for w, h in resize_list:
        temp = {}
        img = cv2.resize(orig_img, (w, h))
        temp["res"] = f"{w}x{h}"
        start = time.time()
        feat_ext_0.extract_features_path(img_path)
        temp["old"] = time.time() - start
        start = time.time()
        feat_ext.extract_features_path(img_path)
        temp["new"] = time.time() - start
        data.append(temp)
        print(f"{w}x{h} done")
df = pd.DataFrame(data)
print(df.sort_values("res"))

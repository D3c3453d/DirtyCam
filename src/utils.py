import os

import cv2
import torch


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


# dom = DOM()
# print(dom.get_sharpness(str(PREDICT_DIR / "maxim-bogdanov-wjAR4jo979Y-unsplash.jpg")))


class FeatureExtractor:
    class register:
        def __init__(self, func):
            self.func = func

        def __set_name__(self, owner, name):
            owner._registry.append(name)

        def __get__(self, instance, owner):
            if instance is None:
                # accessing on the class
                return self.func
            # binding the function to the instance
            return self.func.__get__(instance, owner)

    _registry = []  # list of feature‐names

    def __init__(self):
        self.columns = list(self._registry)
        self.a = 111

    def extract_features(self, gray_img) -> dict:
        return {name: getattr(self, name)(gray_img) for name in self.columns}

    @register
    def brenner_gradient(self, gray_img):
        return gray_img + self.a

    @register
    def another_feature(self, gray_img):
        return gray_img * 2


fe = FeatureExtractor()
print(fe.columns)  # ['brenner_gradient', 'another_feature']
print(fe.extract_features(10))  # {'brenner_gradient': 121, 'another_feature': 20}
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(5, 5).to(device)
y = torch.rand(5, 5).to(device)
print(x)
print(y)
z = x + y
print(z)

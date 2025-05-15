import inspect

import cv2
import numpy as np
from brisque import BRISQUE


class FeatureExtractor:
    def __init__(self):
        methods_dict = self.__get_methods()
        self.__brisq = BRISQUE()
        self.columns = list(methods_dict.keys())
        self.__methods = list(methods_dict.values())

    def __brenner_gradient(self, gray_img: np.ndarray, *args, **kwargs):
        shifted = np.roll(gray_img, -2, axis=1)  # Shift by 2 pixels horizontally
        diff = (gray_img - shifted) ** 2  # Compute squared difference
        return np.sum(diff)  # Sum all differences as the focus measure

    def __sobel_variance(self, gray_img: np.ndarray, *args, **kwargs):
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        variance = np.var(gray_img)  # Compute variance of pixel intensities
        return np.mean(sobel_magnitude) + variance  # Combine Sobel and variance

    def __tenengrad(self, gray_img: np.ndarray, *args, **kwargs):
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
        tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        return np.mean(tenengrad)  # Return mean gradient magnitude as focus score

    def __laplacian(self, gray_img: np.ndarray, *args, **kwargs):
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)  # Apply Laplacian filter
        return np.var(laplacian)  # Compute variance of Laplacian

    def __brisque(self, gray_img: np.ndarray, *args, **kwargs):
        score = self.__brisq.get_score(gray_img)
        print(f"brisq: {score}")
        return score

    def __get_methods(self):
        prefix = "__"
        prefix_len = len(prefix)
        this_func_name = inspect.currentframe().f_code.co_name
        return {
            obj.__name__[prefix_len:]: obj
            for name, obj in inspect.getmembers(self.__class__)
            if (
                inspect.isfunction(obj)
                and obj.__name__.startswith(prefix)
                and obj.__name__ != this_func_name
                and obj.__name__ != "__init__"
            )
        }

    def compute_features(self, img: np.ndarray):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        return [m(self, gray_img=gray_img) for m in self.__methods]

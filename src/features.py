import inspect
import logging
from pathlib import Path

import cv2
import numpy as np
from brisque import BRISQUE
from settings import LOG_FORMAT

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self):
        self._methods = self._get_methods()
        self._brisque = BRISQUE()
        self.columns = list(self._methods.keys())

    def extract_features(self, path: Path) -> dict | None:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to read image: {path}")
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {name: method(gray_img=gray) for name, method in self._methods.items()}

    def _get_methods(self):
        prefix = "_feat_"
        return {
            name[len(prefix) :]: method
            for name, method in inspect.getmembers(self, inspect.ismethod)
            if name.startswith(prefix)
        }

    # --- Feature methods ---

    def _feat_brenner_gradient(self, gray_img: np.ndarray):
        shifted = np.roll(gray_img, -2, axis=1)  # Shift by 2 pixels horizontally
        return np.sum((gray_img - shifted) ** 2)  # Sum all squared differences as the focus measure

    def _feat_sobel_variance(self, gray_img: np.ndarray):
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        return np.mean(magnitude) + np.var(gray_img)  # Combine Sobel and variance of pixel intensities

    def _feat_tenengrad(self, gray_img: np.ndarray):
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
        return np.mean(np.sqrt(sobel_x**2 + sobel_y**2))  # Return mean gradient magnitude as focus score

    def _feat_laplacian(self, gray_img: np.ndarray):
        return np.var(cv2.Laplacian(gray_img, cv2.CV_64F))  # Compute variance of Laplacian

    def _feat_brisque(self, gray_img: np.ndarray):
        return self._brisque.get_score(gray_img)

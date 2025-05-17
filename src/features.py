import inspect
import logging
from pathlib import Path

import cv2
import numpy as np
from brisque import BRISQUE
from dom import DOM
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
        self._dom = DOM()
        self.columns = list(self._methods.keys())

    def extract_features(self, path: Path) -> dict | None:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to read image: {path}")
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {name: method(gray_img=gray_img, img=img, path=path) for name, method in self._methods.items()}

    def _get_methods(self):
        prefix = "_feat_"
        return {
            name[len(prefix) :]: method
            for name, method in inspect.getmembers(self, inspect.ismethod)
            if name.startswith(prefix)
        }

    # --- Feature methods ---

    def _feat_brenner_gradient(self, gray_img: np.ndarray, *args, **kwargs):
        shifted = np.roll(gray_img, -2, axis=1)  # Shift by 2 pixels horizontally
        return np.sum((gray_img - shifted) ** 2)  # Sum all squared differences as the focus measure

    def _feat_sobel_variance(self, gray_img: np.ndarray, *args, **kwargs):
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        return np.mean(magnitude) + np.var(gray_img)  # Combine Sobel and variance of pixel intensities

    def _feat_tenengrad(self, gray_img: np.ndarray, *args, **kwargs):
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
        return np.mean(np.sqrt(sobel_x**2 + sobel_y**2))  # Return mean gradient magnitude as focus score

    def _feat_laplacian(self, gray_img: np.ndarray, *args, **kwargs):
        return np.var(cv2.Laplacian(gray_img, cv2.CV_64F))  # Compute variance of Laplacian

    # def _feat_dom(self, img: np.ndarray, *args, **kwargs):
    #     return self._dom.get_sharpness(img)

    def _feat_brisque(self, gray_img: np.ndarray, *args, **kwargs):
        return self._brisque.get_score(gray_img)

    def _feat_texture_quality(self, gray_img: np.ndarray, *args, **kwargs):
        def radial_average(arr: np.ndarray) -> np.ndarray:
            N = arr.shape[0]
            y, x = np.indices((N, N))
            r = np.hypot(x - N // 2, y - N // 2).astype(np.int32)
            radial_sum = np.bincount(r.ravel(), weights=arr.ravel())
            radial_count = np.bincount(r.ravel())
            return radial_sum / np.maximum(radial_count, 1)

        N = min(gray_img.shape)
        if N % 2 == 0:
            N -= 1
        I = gray_img[:N, :N]  # noqa: E741

        I_hat = np.fft.fftshift(np.fft.fft2(I))
        I_hat_abs = np.abs(I_hat)

        y, x = np.indices((N, N))
        r2 = (x - N // 2) ** 2 + (y - N // 2) ** 2
        r2[N // 2, N // 2] = 1  # avoid division by zero

        eta = -1.93
        cN = (I.var() / np.sum(1 / r2 ** (eta / 2))) * N**4
        T_hat = cN / r2 ** (eta / 2)
        T_hat[N // 2, N // 2] = I_hat_abs[N // 2, N // 2]

        K = I_hat_abs / T_hat
        MTF = radial_average(K)

        b, c = 0.2, 0.8
        v = np.arange(len(MTF))
        CSF = v**c * np.exp(-b * v)
        CSF /= np.sum(CSF)

        return np.sum(MTF * CSF)

    def _feat_smd(self, gray_img: np.ndarray, *args, **kwargs):
        dx = np.abs(gray_img[1:, :-1] - gray_img[:-1, :-1])
        dy = np.abs(gray_img[:-1, 1:] - gray_img[:-1, :-1])
        return np.sum(dx + dy)

    def _feat_smd2(self, gray_img: np.ndarray, *args, **kwargs):
        dx = np.abs(gray_img[:-1, :-1] - gray_img[1:, :-1])
        dy = np.abs(gray_img[:-1, :-1] - gray_img[:-1, 1:])
        return np.sum(dx * dy)

    def _feat_variance(self, gray_img: np.ndarray, *args, **kwargs):
        return np.var(gray_img)

    def _feat_energy(self, gray_img: np.ndarray, *args, **kwargs):
        dx = gray_img[1:, :-1] - gray_img[:-1, :-1]
        dy = gray_img[:-1, 1:] - gray_img[:-1, :-1]
        return np.sum((dx**2) * (dy**2))

    def _feat_vollath(self, gray_img: np.ndarray, *args, **kwargs):
        u = np.mean(gray_img)
        shifted = gray_img[1:, :] * gray_img[:-1, :]
        return np.sum(shifted) - gray_img.shape[0] * gray_img.shape[1] * (u**2)

    def _feat_entropy(self, gray_img: np.ndarray, *args, **kwargs):
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        hist_nonzero = hist_norm[hist_norm > 0]
        return -np.sum(hist_nonzero * np.log2(hist_nonzero))

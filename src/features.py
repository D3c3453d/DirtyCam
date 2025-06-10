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
    class feature:
        def __init__(self, func):
            self.func = func

        def __set_name__(self, owner, name):
            owner.columns.append(name)

        def __get__(self, instance, owner):
            if instance is None:
                # accessing on the class
                return self.func
            # binding the function to the instance
            return self.func.__get__(instance, owner)

    columns = []  # list of feature‐names

    def __init__(self):
        self._brisque = BRISQUE()
        self._dom = DOM()

    def extract_features(self, path: Path) -> dict | None:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to read image: {path}")
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
        return {name: getattr(self, name)(gray_img=gray_img, sobel_x=sobel_x, sobel_y=sobel_y) for name in self.columns}

    # --- Feature methods ---

    @feature
    def brenner_gradient(self, gray_img: np.ndarray, *args, **kwargs):  # Shift by 2 pixels horizontally
        return np.sum((gray_img[:, :-2] - gray_img[:, 2:]) ** 2)  # and sum all squared differences as the focus measure

    @feature
    def sobel_variance(self, gray_img: np.ndarray, sobel_x: np.ndarray, sobel_y: np.ndarray, *args, **kwargs):
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        return np.mean(magnitude) + np.var(gray_img)  # Combine Sobel and variance of pixel intensities

    @feature
    def tenengrad(self, sobel_x: np.ndarray, sobel_y: np.ndarray, *args, **kwargs):
        return np.mean(np.sqrt(sobel_x**2 + sobel_y**2))  # Return mean gradient magnitude as focus score

    @feature
    def laplacian(self, gray_img: np.ndarray, *args, **kwargs):
        return np.var(cv2.Laplacian(gray_img, cv2.CV_64F))  # Compute variance of Laplacian

    # @feature
    # def dom(self, img: np.ndarray, *args, **kwargs):
    #     return self._dom.get_sharpness(img)

    # @feature
    # def brisque(self, gray_img: np.ndarray, *args, **kwargs):
    #     return self._brisque.get_score(gray_img)

    @feature
    def texture_quality(self, gray_img: np.ndarray, *args, **kwargs):
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

    @feature
    def smd(self, gray_img: np.ndarray, *args, **kwargs):
        dx = np.abs(gray_img[1:, :-1] - gray_img[:-1, :-1])
        dy = np.abs(gray_img[:-1, 1:] - gray_img[:-1, :-1])
        return np.sum(dx + dy)

    @feature
    def smd2(self, gray_img: np.ndarray, *args, **kwargs):
        dx = np.abs(gray_img[:-1, :-1] - gray_img[1:, :-1])
        dy = np.abs(gray_img[:-1, :-1] - gray_img[:-1, 1:])
        return np.sum(dx * dy)

    @feature
    def variance(self, gray_img: np.ndarray, *args, **kwargs):
        return np.var(gray_img)

    @feature
    def energy(self, gray_img: np.ndarray, *args, **kwargs):
        dx = gray_img[1:, :-1] - gray_img[:-1, :-1]
        dy = gray_img[:-1, 1:] - gray_img[:-1, :-1]
        return np.sum((dx**2) * (dy**2))

    @feature
    def vollath(self, gray_img: np.ndarray, *args, **kwargs):
        u = np.mean(gray_img)
        shifted = gray_img[1:, :] * gray_img[:-1, :]
        return np.sum(shifted) - gray_img.shape[0] * gray_img.shape[1] * (u**2)

    @feature
    def entropy(self, gray_img: np.ndarray, *args, **kwargs):
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        hist_nonzero = hist_norm[hist_norm > 0]
        return -np.sum(hist_nonzero * np.log2(hist_nonzero))

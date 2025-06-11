import logging
import math
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


class FeatureExtractor0:
    class feature:
        def __init__(self, func):
            self.func = func

        def __set_name__(self, owner, name):
            owner.columns.append(name)

        def __get__(self, instance, owner):
            if instance is None:
                return self.func
            return self.func.__get__(instance, owner)

    columns = []

    def __init__(self):
        self._brisque = BRISQUE()
        self._dom = DOM()

    def extract_features_path(self, path: Path) -> dict | None:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to read image: {path}")
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.extract_features(gray_img)

    def extract_features(self, gray_img: np.ndarray) -> dict | None:
        padded = np.pad(gray_img, ((1, 1), (1, 1)), mode="reflect")
        sobel_x = cv2.Sobel(padded, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
        sobel_y = cv2.Sobel(padded, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
        return {name: getattr(self, name)(gray_img=gray_img, sobel_x=sobel_x, sobel_y=sobel_y) for name in self.columns}

    # --- Feature methods ---

    @feature
    def brenner_gradient(self, gray_img: np.ndarray, **kwargs):
        gray_img = gray_img.astype(np.float32)
        shifted = np.roll(gray_img, -2, axis=1)  # Shift by 2 pixels horizontally
        diff = (gray_img - shifted) ** 2  # Compute squared difference
        return np.sum(diff)  # Sum all differences as the focus measure

    @feature
    def sobel_variance(self, gray_img: np.ndarray, sobel_x: np.ndarray, sobel_y: np.ndarray, **kwargs):
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        variance = np.var(gray_img)  # Compute variance of pixel intensities
        return np.mean(sobel_magnitude) + variance  # Combine Sobel and variance

    @feature
    def tenengrad(self, gray_img: np.ndarray, sobel_x: np.ndarray, sobel_y: np.ndarray, **kwargs):
        tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
        return np.mean(tenengrad)  # Return mean gradient magnitude as focus score

    @feature
    def laplacian(self, gray_img: np.ndarray, **kwargs):
        laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)  # Apply Laplacian filter
        return np.var(laplacian)  # Compute variance of Laplacian

    @feature
    def texture_quality(self, gray_img: np.ndarray, **kwargs):
        def radialAverage(arr):
            assert arr.shape[0] == arr.shape[1]
            N = arr.shape[0]
            y, x = np.indices(arr.shape)
            center = np.array([N // 2, N // 2])
            r = np.hypot(x - center[0], y - center[1])
            ind = np.argsort(r.flat)
            r_sorted = r.flat[ind]
            i_sorted = arr.flat[ind]
            r_int = r_sorted.astype(int)
            deltar = r_int - np.roll(r_int, -1)  # shift and substract
            rind = np.where(deltar != 0)[0]  # location of changed radius
            csim = np.cumsum(i_sorted, dtype=float)
            tbin = csim[rind]
            tbin[1:] -= csim[rind[:-1]]
            nr = rind - np.roll(rind, 1)
            nr = nr[1:]
            tbin[1:] /= nr
            return tbin

        N = min(gray_img.shape)
        if N % 2 == 0:
            N -= 1
        I = gray_img[:N, :N]  # NOQA: E741
        I_hat = np.fft.fft2(I)
        I_hat = np.fft.fftshift(I_hat)
        I_hat = np.abs(I_hat)
        eta = -1.93
        Denominator = 0
        for m in range(0, N):
            for n in range(0, N):
                if m == N // 2 and n == N // 2:
                    continue
                Denominator += 1 / pow(((m - N // 2) ** 2 + (n - N // 2) ** 2), eta / 2)
        cN = (I.var() / Denominator) * (N**4)
        T_hat = np.zeros((N, N))
        for m in range(0, N):
            for n in range(0, N):
                if m == N // 2 and n == N // 2:
                    continue
                T_hat[m, n] = cN / ((m - N // 2) ** 2 + (n - N // 2) ** 2) ** (eta / 2)
        T_hat[N // 2, N // 2] = I_hat[N // 2, N // 2]
        K = I_hat / T_hat
        MTF = radialAverage(K)
        b = 0.2
        c = 0.8
        a = 1 / np.sum([pow(v, c) * pow(math.e, -b * v) for v in range(MTF.shape[0])])
        CSF = [a * pow(v, c) * pow(math.e, -b * v) for v in range(MTF.shape[0])]
        A = np.sum([MTF[v] * CSF[v] for v in range(MTF.shape[0])])
        return A

    @feature
    def smd(self, gray_img: np.ndarray, **kwargs):
        shape = np.shape(gray_img)
        out = 0
        for y in range(0, shape[1] - 1):
            for x in range(0, shape[0] - 1):
                out += math.fabs(int(gray_img[x, y]) - int(gray_img[x, y - 1]))
                out += math.fabs(int(gray_img[x, y] - int(gray_img[x + 1, y])))
        return out

    @feature
    def smd2(self, gray_img: np.ndarray, **kwargs):
        shape = np.shape(gray_img)
        out = 0
        for y in range(0, shape[1] - 1):
            for x in range(0, shape[0] - 1):
                out += math.fabs(int(gray_img[x, y]) - int(gray_img[x + 1, y])) * math.fabs(
                    int(gray_img[x, y] - int(gray_img[x, y + 1]))
                )
        return out

    @feature
    def variance(self, gray_img: np.ndarray, **kwargs):
        out = 0
        u = np.mean(gray_img)
        shape = np.shape(gray_img)
        for y in range(0, shape[1]):
            for x in range(0, shape[0]):
                out += (gray_img[x, y] - u) ** 2
        return out

    @feature
    def energy(self, gray_img: np.ndarray, **kwargs):
        shape = np.shape(gray_img)
        out = 0
        for y in range(0, shape[1] - 1):
            for x in range(0, shape[0] - 1):
                out += ((int(gray_img[x + 1, y]) - int(gray_img[x, y])) ** 2) * (
                    (int(gray_img[x, y + 1] - int(gray_img[x, y]))) ** 2
                )
        return out

    @feature
    def vollath(self, gray_img: np.ndarray, **kwargs):
        shape = np.shape(gray_img)
        u = np.mean(gray_img)
        out = -shape[0] * shape[1] * (u**2)
        for y in range(0, shape[1]):
            for x in range(0, shape[0] - 1):
                out += int(gray_img[x, y]) * int(gray_img[x + 1, y])
        return out

    @feature
    def entropy(self, gray_img: np.ndarray, **kwargs):
        [rows, cols] = gray_img.shape
        h = 0
        hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0.0, 255.0])
        # hn valueis not correct
        hb = np.zeros((256, 1), np.float32)
        # hn = np.zeros((256, 1), np.float32)
        for j in range(0, 256):
            hb[j, 0] = hist_gray[j, 0] / (rows * cols)
        for i in range(0, 256):
            if hb[i, 0] > 0:
                h = h - (hb[i, 0]) * math.log(hb[i, 0], 2)

        out = h
        return out

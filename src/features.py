import logging
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
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
                return self.func
            return self.func.__get__(instance, owner)

    columns = []

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._brisque = BRISQUE()
        self._dom = DOM()

        # Initialize kernels on first use
        self.kernel_x = None
        self.kernel_y = None
        self.laplacian_kernel = None

    def _init_kernels(self):
        """Lazily initialize kernels on first call"""
        if self.kernel_x is None:
            self.kernel_x = torch.tensor(
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=self.device
            ).view(1, 1, 3, 3)

            self.kernel_y = torch.tensor(
                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device
            ).view(1, 1, 3, 3)

            self.laplacian_kernel = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=self.device
            ).view(1, 1, 3, 3)

    def extract_features_path(self, path: Path) -> dict | None:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to read image: {path}")
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.extract_features(gray_img)

    def extract_features(self, gray_img) -> dict | None:
        # Initialize kernels on first use
        self._init_kernels()

        gray_tensor = torch.from_numpy(gray_img).float().to(self.device)
        gray_batch = gray_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            sobel_x = F.conv2d(gray_batch, self.kernel_x, padding=1).squeeze()
            sobel_y = F.conv2d(gray_batch, self.kernel_y, padding=1).squeeze()

        return {
            name: getattr(self, name)(gray_tensor=gray_tensor, sobel_x=sobel_x, sobel_y=sobel_y)
            for name in self.columns
        }

    # --- Feature methods ---

    @feature
    def brenner_gradient(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            return torch.sum((gray_tensor[:, :-2] - gray_tensor[:, 2:]) ** 2).item()

    @feature
    def sobel_variance(self, gray_tensor: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor, **kwargs):
        with torch.no_grad():
            magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
            return (torch.mean(magnitude) + torch.var(gray_tensor)).item()

    @feature
    def tenengrad(self, sobel_x: torch.Tensor, sobel_y: torch.Tensor, **kwargs):
        with torch.no_grad():
            return torch.mean(torch.sqrt(sobel_x**2 + sobel_y**2)).item()

    @feature
    def laplacian(self, gray_tensor: torch.Tensor, **kwargs):
        self._init_kernels()
        with torch.no_grad():
            gray_batch = gray_tensor.unsqueeze(0).unsqueeze(0)
            lap = F.conv2d(gray_batch, self.laplacian_kernel, padding=1).squeeze()
            return torch.var(lap).item()

    @feature
    def texture_quality(self, gray_tensor: torch.Tensor, **kwargs):
        def radial_average(arr: torch.Tensor) -> torch.Tensor:
            N = arr.shape[0]
            # Используем float вместо long для индексов
            y, x = torch.meshgrid(
                torch.arange(N, device=arr.device, dtype=torch.float32),
                torch.arange(N, device=arr.device, dtype=torch.float32),
                indexing="ij",
            )
            # Вычисляем расстояния с помощью float
            r = torch.hypot(x - N // 2, y - N // 2).long()
            r_flat = r.flatten()
            arr_flat = arr.flatten()
            max_r = r_flat.max().item()

            sums = torch.zeros(max_r + 1, dtype=arr.dtype, device=arr.device)
            counts = torch.zeros(max_r + 1, dtype=arr.dtype, device=arr.device)
            sums.scatter_add_(0, r_flat, arr_flat)
            counts.scatter_add_(0, r_flat, torch.ones_like(arr_flat))

            return sums / torch.clamp(counts, min=1)

        try:
            with torch.no_grad():
                N = min(gray_tensor.shape)
                if N % 2 == 0:
                    N -= 1
                I = gray_tensor[:N, :N]  # NOQA: E741

                I_hat = torch.fft.fft2(I)
                I_hat = torch.fft.fftshift(I_hat)
                I_hat_abs = torch.abs(I_hat)

                # Также используем float здесь
                y, x = torch.meshgrid(
                    torch.arange(N, device=I.device, dtype=torch.float32),
                    torch.arange(N, device=I.device, dtype=torch.float32),
                    indexing="ij",
                )
                r2 = (x - N // 2) ** 2 + (y - N // 2) ** 2
                r2[N // 2, N // 2] = 1  # avoid division by zero

                eta = -1.93
                cN = (torch.var(I) / torch.sum(1 / r2 ** (eta / 2))) * (N**4)
                T_hat = cN / r2 ** (eta / 2)
                T_hat[N // 2, N // 2] = I_hat_abs[N // 2, N // 2]

                K = I_hat_abs / T_hat
                MTF = radial_average(K)

                b, c = 0.2, 0.8
                v = torch.arange(len(MTF), device=MTF.device, dtype=torch.float32)
                CSF = v**c * torch.exp(-b * v)
                CSF /= torch.sum(CSF)

                return torch.sum(MTF * CSF).item()
        except Exception as e:
            logger.error(f"Error in texture_quality: {str(e)}")
            return 0.0  # Возвращаем значение по умолчанию в случае ошибки

    @feature
    def smd(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            dx = torch.abs(gray_tensor[1:, :-1] - gray_tensor[:-1, :-1])
            dy = torch.abs(gray_tensor[:-1, 1:] - gray_tensor[:-1, :-1])
            return torch.sum(dx + dy).item()

    @feature
    def smd2(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            dx = torch.abs(gray_tensor[:-1, :-1] - gray_tensor[1:, :-1])
            dy = torch.abs(gray_tensor[:-1, :-1] - gray_tensor[:-1, 1:])
            return torch.sum(dx * dy).item()

    @feature
    def variance(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            return torch.var(gray_tensor).item()

    @feature
    def energy(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            dx = gray_tensor[1:, :-1] - gray_tensor[:-1, :-1]
            dy = gray_tensor[:-1, 1:] - gray_tensor[:-1, :-1]
            return torch.sum((dx**2) * (dy**2)).item()

    @feature
    def vollath(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            u = torch.mean(gray_tensor)
            shifted = gray_tensor[1:, :] * gray_tensor[:-1, :]
            return (torch.sum(shifted) - gray_tensor.numel() * (u**2)).item()

    @feature
    def entropy(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            # Normalize to 0-255 range
            normalized = (gray_tensor - gray_tensor.min()) / (gray_tensor.max() - gray_tensor.min()) * 255
            normalized = normalized.byte()

            hist = torch.histc(normalized.float(), bins=256, min=0, max=255)
            hist_norm = hist / hist.sum()
            hist_nonzero = hist_norm[hist_norm > 0]
            return -torch.sum(hist_nonzero * torch.log2(hist_nonzero)).item()

import logging
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
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

        # Initialize kernels on first use
        self.kernel_x = None
        self.kernel_y = None
        self.laplacian_kernel = None

    def _init_kernels(self):
        if self.kernel_x is None:
            # Sobel kernels matching cv2.Sobel for ksize=3
            # Note: sign does not affect magnitude-based measures
            kx = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device=self.device
            )
            ky = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32, device=self.device
            )
            self.kernel_x = kx.view(1, 1, 3, 3)
            self.kernel_y = ky.view(1, 1, 3, 3)

            # Laplacian kernel matching cv2.Laplacian
            lap = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=self.device
            )
            self.laplacian_kernel = lap.view(1, 1, 3, 3)

    def extract_features_path(self, path: Path) -> dict | None:
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to read image: {path}")
            return None
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.extract_features(gray_img)

    def extract_features(self, gray_img) -> dict | None:
        self._init_kernels()

        # Convert to tensor
        gray_tensor = torch.from_numpy(gray_img).float().to(self.device)
        # Add batch and channel dims
        gray_batch = gray_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Use reflect padding to mimic cv2 border handling
            padded = F.pad(gray_batch, (1, 1, 1, 1), mode="reflect")
            sobel_x = F.conv2d(padded, self.kernel_x).squeeze(0).squeeze(0)
            sobel_y = F.conv2d(padded, self.kernel_y).squeeze(0).squeeze(0)
            lap = F.conv2d(padded, self.laplacian_kernel).squeeze(0).squeeze(0)
        return {
            name: getattr(self, name)(gray_tensor=gray_tensor, sobel_x=sobel_x, sobel_y=sobel_y, lap=lap)
            for name in self.columns
        }

    # --- Feature methods ---

    @feature
    def brenner_gradient(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            # roll horizontally by -2 same as numpy roll
            shifted = torch.roll(gray_tensor, shifts=-2, dims=1)
            return torch.sum((gray_tensor - shifted) ** 2).item()

    @feature
    def sobel_variance(self, gray_tensor: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor, **kwargs):
        with torch.no_grad():
            magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
            # Population variance
            var = torch.var(gray_tensor, unbiased=False)
            return (torch.mean(magnitude) + var).item()

    @feature
    def tenengrad(self, gray_tensor: torch.Tensor, sobel_x: torch.Tensor, sobel_y: torch.Tensor, **kwargs):
        with torch.no_grad():
            mag = torch.sqrt(sobel_x**2 + sobel_y**2)
            return torch.mean(mag).item()

    @feature
    def laplacian(self, lap: torch.Tensor, **kwargs):
        with torch.no_grad():
            # Population variance of laplacian
            return torch.var(lap, unbiased=False).item()

    @feature
    def texture_quality(self, gray_tensor: torch.Tensor, **kwargs):
        def radial_average(arr: torch.Tensor) -> torch.Tensor:
            N = arr.shape[0]
            # Create float indices to allow hypot
            y, x = torch.meshgrid(
                torch.arange(N, device=arr.device, dtype=torch.float32),
                torch.arange(N, device=arr.device, dtype=torch.float32),
                indexing="ij",
            )
            # Compute radius distances in float, then cast to long for binning
            r = torch.hypot(x - (N // 2), y - (N // 2)).long().flatten()
            arr_flat = arr.flatten()
            max_r = r.max().item()
            sums = torch.zeros(max_r + 1, dtype=arr.dtype, device=arr.device)
            counts = torch.zeros(max_r + 1, dtype=arr.dtype, device=arr.device)
            sums.scatter_add_(0, r, arr_flat)
            counts.scatter_add_(0, r, torch.ones_like(arr_flat))
            return sums / torch.clamp(counts, min=1)

        with torch.no_grad():
            N = min(gray_tensor.shape)
            if N % 2 == 0:
                N -= 1
            I = gray_tensor[:N, :N]  # NOQA: E741
            I_hat = torch.fft.fftshift(torch.fft.fft2(I))
            I_hat_abs = torch.abs(I_hat)

            y, x = torch.meshgrid(
                torch.arange(N, device=I.device, dtype=torch.float32),
                torch.arange(N, device=I.device, dtype=torch.float32),
                indexing="ij",
            )
            r2 = (x - N // 2) ** 2 + (y - N // 2) ** 2
            r2[N // 2, N // 2] = 1.0

            eta = -1.93
            denom = torch.sum(1.0 / (r2 ** (eta / 2)))
            cN = (torch.var(I, unbiased=False) / denom) * (N**4)
            T_hat = cN / (r2 ** (eta / 2))
            T_hat[N // 2, N // 2] = I_hat_abs[N // 2, N // 2]

            K = I_hat_abs / T_hat
            MTF = radial_average(K)

            b, c = 0.2, 0.8
            v = torch.arange(len(MTF), device=MTF.device, dtype=torch.float32)
            CSF = v**c * torch.exp(-b * v)
            CSF = CSF / torch.sum(CSF)
            return torch.sum(MTF * CSF).item()

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
            # Sum of squared deviations to match original loops: var * N
            mean = torch.mean(gray_tensor)
            diff = (gray_tensor - mean) ** 2
            return torch.sum(diff).item()

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
            # original: sum products minus N*M*(u^2)
            return (torch.sum(shifted) - gray_tensor.numel() * (u**2)).item()

    @feature
    def entropy(self, gray_tensor: torch.Tensor, **kwargs):
        with torch.no_grad():
            tensor = gray_tensor.flatten()
            hist = torch.histc(tensor.float(), bins=256, min=0, max=255)
            prob = hist / hist.sum()
            prob_nonzero = prob[prob > 0]
            return -torch.sum(prob_nonzero * torch.log2(prob_nonzero)).item()

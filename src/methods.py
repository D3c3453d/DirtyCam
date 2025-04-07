import cv2
import numpy as np


def brenner_gradient(gray_image: np.ndarray):
    shifted = np.roll(gray_image, -2, axis=1)  # Shift by 2 pixels horizontally
    diff = (gray_image - shifted) ** 2  # Compute squared difference
    return np.sum(diff)  # Sum all differences as the focus measure


def sobel_variance(gray_image: np.ndarray):
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X gradient
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y gradient
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
    variance = np.var(gray_image)  # Compute variance of pixel intensities
    return np.mean(sobel_magnitude) + variance  # Combine Sobel and variance


def tenengrad(gray_image: np.ndarray):
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
    tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)  # Compute gradient magnitude
    return np.mean(tenengrad)  # Return mean gradient magnitude as focus score


def laplacian(gray_image: np.ndarray):
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)  # Apply Laplacian filter
    return np.var(laplacian)  # Compute variance of Laplacian

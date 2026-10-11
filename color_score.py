from __future__ import annotations
import cv2
import numpy as np
from skimage import exposure, measure, morphology
from .legacy_models import Settings

def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 else n + 1



def circular_field_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Suppress dark microscope borders by finding the bright circular field."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (31, 31), 0)
    threshold = max(12, int(np.percentile(smooth, 8)))
    field = smooth > threshold
    field = morphology.remove_small_holes(field, area_threshold=50000)
    field = morphology.remove_small_objects(field, min_size=max(1000, field.size // 15))
    labels = measure.label(field)
    if labels.max() == 0:
        return np.ones(gray.shape, dtype=bool)
    largest = max(measure.regionprops(labels), key=lambda r: r.area)
    result = labels == largest.label
    # OpenCV's binary morphology is substantially faster on large microscope
    # fields than repeatedly applying a scikit-image disk footprint.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    return cv2.erode(result.astype(np.uint8), kernel) > 0



def stain_score(image_bgr: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Compute a robust purple/blue Jenner–Giemsa score in [0, 1]."""
    image = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    field = circular_field_mask(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Illumination correction: divide by a slowly varying background.
    k = _odd(settings.background_kernel)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    corrected = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    darkness = 1.0 - corrected.astype(np.float32) / 255.0

    # Jenner–Giemsa myotubes are purple/blue relative to the pale yellow-green field.
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    purple = np.maximum(a, 0.0) + np.maximum(-b, 0.0) * 0.75
    purple = exposure.rescale_intensity(purple, in_range='image', out_range=(0, 1)).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0
    darkness = exposure.rescale_intensity(darkness, in_range='image', out_range=(0, 1)).astype(np.float32)

    # Emphasize stained elongated cytoplasm while reducing pale texture.
    score = 0.58 * purple + 0.27 * darkness + 0.15 * saturation
    score[~field] = 0
    score = cv2.GaussianBlur(score, (3, 3), 0)
    return np.clip(score, 0, 1), field


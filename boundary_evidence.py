"""Experimental color evidence; parameters require calibration on real images."""
import cv2
import numpy as np
from .config import Settings


def color_evidence(image: np.ndarray, settings: Settings):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected an 8-bit BGR image')
    work = image
    if settings.color_normalization:
        white = np.percentile(image.reshape(-1, 3), 95, axis=0)
        gains = np.clip(np.mean(white) / np.maximum(white, 1), 0.8, 1.25)
        work = np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV).astype(np.float32)
    light = lab[:, :, 0] / 255
    saturation = hsv[:, :, 1] / 255
    hue = hsv[:, :, 0]
    # White and pale yellow provide a barrier BEFORE grouping cytoplasm.
    white = light * (1 - saturation)
    yellow = light * np.clip((lab[:, :, 2] - 128) / 25, 0, 1)
    yellow *= (hue >= 15) & (hue <= 45)
    boundary_score = np.maximum(white, yellow)
    if settings.evidence_sigma_px > 0:
        boundary_score = cv2.GaussianBlur(boundary_score, (0, 0), settings.evidence_sigma_px)
    barrier = boundary_score >= settings.boundary_threshold
    if settings.boundary_continuity:
        # Join only short gaps with independent, weaker background evidence.
        expanded = barrier.copy()
        for kernel in (np.ones((1, 5), np.uint8), np.ones((5, 1), np.uint8),
                       np.eye(5, dtype=np.uint8), np.fliplr(np.eye(5, dtype=np.uint8)).copy()):
            expanded |= cv2.morphologyEx(barrier.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        barrier |= expanded & (boundary_score >= settings.boundary_threshold * 0.75)
    purple = np.maximum(lab[:, :, 1] - 128, 0) + 0.75 * np.maximum(128 - lab[:, :, 2], 0)
    purple = np.clip(purple / 50, 0, 1)
    blue_hue = ((hue >= 95) & (hue <= 175)).astype(np.float32) * saturation
    cytoplasm = np.maximum(purple, blue_hue)
    if settings.evidence_sigma_px > 0:
        cytoplasm = cv2.GaussianBlur(cytoplasm, (0, 0), settings.evidence_sigma_px)
    field = hsv[:, :, 2] > 20  # exclude black outside the microscope field
    return cytoplasm.astype(np.float32), boundary_score.astype(np.float32), barrier, field

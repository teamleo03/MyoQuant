from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unable to read image: {path}')
    return image



def resize_for_analysis(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    """Return a smaller working image for fast analysis.

    Measurements are converted with a matching scaled calibration value by the
    application, so the reported micrometre diameters remain on the user's
    original calibration scale.
    """
    scale = float(scale)
    if not 0.10 <= scale <= 1.0:
        raise ValueError('Analysis scale must be between 0.10 and 1.00.')
    if scale >= 0.999:
        return image_bgr
    height, width = image_bgr.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)



def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ext = path.suffix or '.png'
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f'Unable to encode image: {path}')
    encoded.tofile(str(path))


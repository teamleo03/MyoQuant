from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .config import Settings
from .image_io import read_image
from .segmentation_engine import segment
from .diameter_engine import measure_diameters


@dataclass
class ImageRecord:
    path: Path
    image_bgr: np.ndarray | None = None
    settings_used: Settings | None = None
    score: np.ndarray | None = None
    raw_mask: np.ndarray | None = None
    accepted_mask: np.ndarray | None = None
    candidates: list = field(default_factory=list)
    measurements: list = field(default_factory=list)
    summaries: list = field(default_factory=list)
    selected_ids: set = field(default_factory=set)
    analyzed: bool = False


def analyze_record(record: ImageRecord, settings: Settings) -> ImageRecord:
    image = record.image_bgr if record.image_bgr is not None else read_image(record.path)
    score, raw, labels, candidates = segment(image, settings)
    ranked = sorted((c for c in candidates if c.status != 'Excluded' and c.area_px >= settings.min_area),
                    key=lambda c: (c.status == 'Ready', c.score, c.length_px), reverse=True)
    if settings.auto_select_limit:
        ranked = ranked[:settings.auto_select_limit]
    selected = {c.id for c in ranked}
    measurements, summaries = measure_diameters(labels, candidates, settings, selected)
    return ImageRecord(record.path, image, settings, score, raw, labels,
                       candidates, measurements, summaries, selected, True)

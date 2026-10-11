from __future__ import annotations

from dataclasses import asdict, dataclass, field
import numpy as np


@dataclass
class Candidate:
    id: int
    label: int
    area_px: int
    length_px: float
    length_um: float
    width_px: float
    aspect_ratio: float
    eccentricity: float
    solidity: float
    centroid_x: float
    centroid_y: float
    score: float
    status: str = 'Ready'
    review_reason: str = ''
    segmentation_reason: str = ''
    centerline: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), repr=False)

    def row(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if name != 'centerline'}


@dataclass
class DiameterMeasurement:
    myotube_id: int
    measurement_id: int
    fraction: float
    center_x: float | None = None
    center_y: float | None = None
    endpoint1_x: float | None = None
    endpoint1_y: float | None = None
    endpoint2_x: float | None = None
    endpoint2_y: float | None = None
    diameter_px: float | None = None
    diameter_um: float | None = None
    status: str = 'Review'
    reason: str = ''
    method: str = 'automatic'

    def row(self):
        return asdict(self)


@dataclass
class DiameterSummary:
    myotube_id: int
    measurement_count: int
    mean_diameter_um: float | None
    sd_um: float | None
    cv_percent: float | None
    status: str
    review_reason: str
    diameters_um: tuple = (None, None, None, None, None)

    def row(self):
        result = asdict(self)
        result.pop('diameters_um')
        result.update({f'D{i + 1}_um': value for i, value in enumerate(self.diameters_um)})
        return result

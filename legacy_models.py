from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Settings:
    min_area: int = 900
    min_length: float = 85.0
    min_aspect_ratio: float = 3.0
    min_eccentricity: float = 0.88
    min_solidity: float = 0.28
    max_width: float = 180.0
    background_kernel: int = 101
    local_block: int = 101
    local_offset: float = 0.018
    remove_border: bool = False
    measurements_per_myotube: int = 5
    calibration_pixels: float = 558.0
    calibration_microns: float = 250.0
    end_exclusion_fraction: float = 0.12
    nuclear_hole_area: int = 2600
    envelope_closing_radius: int = 4
    overlap_cut_padding: int = 3
    strand_bridge_gap: float = 80.0
    analysis_scale: float = 0.5



@dataclass
class Candidate:
    id: int
    label: int
    area_px: int
    length_px: float
    width_px: float
    aspect_ratio: float
    eccentricity: float
    solidity: float
    centroid_x: float
    centroid_y: float
    score: float
    overlap_detected: bool = False
    overlap_group: int = 0
    overlap_x: float | None = None
    overlap_y: float | None = None

    def row(self):
        return {
            'myotube_id': self.id,
            'area_px': self.area_px,
            'length_px': round(self.length_px, 3),
            'width_px': round(self.width_px, 3),
            'aspect_ratio': round(self.aspect_ratio, 3),
            'eccentricity': round(self.eccentricity, 4),
            'solidity': round(self.solidity, 4),
            'centroid_x': round(self.centroid_x, 2),
            'centroid_y': round(self.centroid_y, 2),
            'confidence': round(self.score, 4),
            'overlap_detected': self.overlap_detected,
            'overlap_group': self.overlap_group or '',
            'overlap_x': '' if self.overlap_x is None else round(self.overlap_x, 2),
            'overlap_y': '' if self.overlap_y is None else round(self.overlap_y, 2),
        }



@dataclass
class DiameterMeasurement:
    myotube_id: int
    measurement_id: int
    center_x: float
    center_y: float
    endpoint1_x: float
    endpoint1_y: float
    endpoint2_x: float
    endpoint2_y: float
    diameter_px: float
    diameter_um: float

    def row(self):
        return {
            'myotube_id': self.myotube_id,
            'measurement_id': self.measurement_id,
            'center_x': round(self.center_x, 2),
            'center_y': round(self.center_y, 2),
            'diameter_px': round(self.diameter_px, 3),
            'diameter_um': round(self.diameter_um, 3),
            'endpoint1_x': round(self.endpoint1_x, 2),
            'endpoint1_y': round(self.endpoint1_y, 2),
            'endpoint2_x': round(self.endpoint2_x, 2),
            'endpoint2_y': round(self.endpoint2_y, 2),
        }



@dataclass
class DiameterSummary:
    myotube_id: int
    measurement_count: int
    mean_diameter_px: float
    median_diameter_px: float
    min_diameter_px: float
    max_diameter_px: float
    mean_diameter_um: float
    median_diameter_um: float
    min_diameter_um: float
    max_diameter_um: float

    def row(self):
        return {
            'myotube_id': self.myotube_id,
            'measurement_count': self.measurement_count,
            'mean_diameter_px': round(self.mean_diameter_px, 3),
            'median_diameter_px': round(self.median_diameter_px, 3),
            'min_diameter_px': round(self.min_diameter_px, 3),
            'max_diameter_px': round(self.max_diameter_px, 3),
            'mean_diameter_um': round(self.mean_diameter_um, 3),
            'median_diameter_um': round(self.median_diameter_um, 3),
            'min_diameter_um': round(self.min_diameter_um, 3),
            'max_diameter_um': round(self.max_diameter_um, 3),
        }


from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from .legacy_models import Settings, Candidate, DiameterMeasurement, DiameterSummary
from .centerlines import _longest_skeleton_path, _interpolate_path, _distance_to_mask_edge

def measure_diameters(accepted: np.ndarray, candidates: list[Candidate],
                      settings: Settings | None = None,
                      selected_ids: set[int] | None = None):
    """Measure local width at evenly spaced centerline locations."""
    settings = settings or Settings()
    if settings.calibration_pixels <= 0 or settings.calibration_microns <= 0:
        raise ValueError('Calibration values must be greater than zero.')
    count = 5  # Always exactly five measurements per selected myotube.
    start = float(np.clip(settings.end_exclusion_fraction, 0.0, 0.45))
    fractions = np.linspace(start, 1.0 - start, count)
    um_per_px = settings.calibration_microns / settings.calibration_pixels

    measurements: list[DiameterMeasurement] = []
    summaries: list[DiameterSummary] = []

    labels = measure.label(accepted, connectivity=2)
    component_slices = ndi.find_objects(labels)
    foreground = None
    # Map candidates by overlap/centroid because accepted has been relabeled.
    for candidate in candidates:
        if selected_ids is not None and candidate.id not in selected_ids:
            continue
        cx = int(np.clip(round(candidate.centroid_x), 0, accepted.shape[1] - 1))
        cy = int(np.clip(round(candidate.centroid_y), 0, accepted.shape[0] - 1))
        component_label = int(labels[cy, cx])
        if component_label == 0:
            # Region centroids can fall in a concavity or internal hole. Find
            # the closest accepted pixel so every selected candidate is measured.
            if foreground is None:
                foreground = np.argwhere(labels > 0)
            if foreground.size == 0:
                continue
            distances_to_centroid = (
                (foreground[:, 0] - candidate.centroid_y) ** 2
                + (foreground[:, 1] - candidate.centroid_x) ** 2
            )
            nearest_y, nearest_x = foreground[int(np.argmin(distances_to_centroid))]
            component_label = int(labels[nearest_y, nearest_x])
        component_slice = component_slices[component_label - 1]
        if component_slice is None:
            continue
        source_y, source_x = component_slice
        y0 = max(0, source_y.start - 4)
        y1 = min(accepted.shape[0], source_y.stop + 4)
        x0 = max(0, source_x.start - 4)
        x1 = min(accepted.shape[1], source_x.stop + 4)
        crop = labels[y0:y1, x0:x1] == component_label

        distance = ndi.distance_transform_edt(crop)
        float_crop = crop.astype(np.float32, copy=False)
        path = _longest_skeleton_path(crop)
        if len(path) < 2:
            continue
        positions, tangents = _interpolate_path(path, fractions)
        candidate_measurements = []

        for measurement_id, (position, tangent) in enumerate(zip(positions, tangents), start=1):
            py, px = position
            iy = int(np.clip(round(py), 0, crop.shape[0] - 1))
            ix = int(np.clip(round(px), 0, crop.shape[1] - 1))
            radius = float(distance[iy, ix])
            if radius <= 0.5:
                # Guarantee five measurements by using the nearest positive
                # distance-transform point around the requested location.
                yy0, yy1 = max(0, iy - 8), min(crop.shape[0], iy + 9)
                xx0, xx1 = max(0, ix - 8), min(crop.shape[1], ix + 9)
                neighborhood = distance[yy0:yy1, xx0:xx1]
                if neighborhood.size and float(neighborhood.max()) > 0:
                    rel_y, rel_x = np.unravel_index(int(np.argmax(neighborhood)), neighborhood.shape)
                    iy, ix = yy0 + rel_y, xx0 + rel_x
                    py, px = float(iy), float(ix)
                    position = np.array([py, px], dtype=float)
                    radius = float(distance[iy, ix])
                else:
                    radius = max(float(candidate.width_px) / 2.0, 0.5)

            # Tangent is (dy, dx); perpendicular normal is (-dx, dy).
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            normal_norm = np.linalg.norm(normal)
            normal = normal / normal_norm if normal_norm > 0 else np.array([1.0, 0.0])
            maximum_distance = max(float(settings.max_width), radius * 3.0)
            edge1 = _distance_to_mask_edge(float_crop, position, -normal, maximum_distance)
            edge2 = _distance_to_mask_edge(float_crop, position, normal, maximum_distance)
            if edge1 <= 0.25 or edge2 <= 0.25:
                # This fallback preserves the promised five measurements even
                # when a selected point lies on an imperfect mask boundary.
                edge1 = edge2 = max(radius, 0.5)
            endpoint1 = position - normal * edge1
            endpoint2 = position + normal * edge2
            diameter_px = edge1 + edge2
            diameter_um = diameter_px * um_per_px

            item = DiameterMeasurement(
                myotube_id=candidate.id,
                measurement_id=measurement_id,
                center_x=float(px + x0), center_y=float(py + y0),
                endpoint1_x=float(endpoint1[1] + x0), endpoint1_y=float(endpoint1[0] + y0),
                endpoint2_x=float(endpoint2[1] + x0), endpoint2_y=float(endpoint2[0] + y0),
                diameter_px=diameter_px, diameter_um=diameter_um,
            )
            measurements.append(item)
            candidate_measurements.append(item)

        if candidate_measurements:
            px_values = np.array([m.diameter_px for m in candidate_measurements])
            um_values = np.array([m.diameter_um for m in candidate_measurements])
            summaries.append(DiameterSummary(
                myotube_id=candidate.id,
                measurement_count=len(candidate_measurements),
                mean_diameter_px=float(np.mean(px_values)),
                median_diameter_px=float(np.median(px_values)),
                min_diameter_px=float(np.min(px_values)),
                max_diameter_px=float(np.max(px_values)),
                mean_diameter_um=float(np.mean(um_values)),
                median_diameter_um=float(np.median(um_values)),
                min_diameter_um=float(np.min(um_values)),
                max_diameter_um=float(np.max(um_values)),
            ))
    return measurements, summaries



def recompute_summaries(measurements: list[DiameterMeasurement],
                        settings: Settings | None = None) -> list[DiameterSummary]:
    """Recalculate per-myotube statistics after manual endpoint edits."""
    settings = settings or Settings()
    grouped: dict[int, list[DiameterMeasurement]] = {}
    for item in measurements:
        grouped.setdefault(item.myotube_id, []).append(item)
    summaries: list[DiameterSummary] = []
    for myotube_id in sorted(grouped):
        items = grouped[myotube_id]
        px_values = np.asarray([m.diameter_px for m in items], dtype=float)
        um_values = np.asarray([m.diameter_um for m in items], dtype=float)
        summaries.append(DiameterSummary(
            myotube_id=myotube_id,
            measurement_count=len(items),
            mean_diameter_px=float(np.mean(px_values)),
            median_diameter_px=float(np.median(px_values)),
            min_diameter_px=float(np.min(px_values)),
            max_diameter_px=float(np.max(px_values)),
            mean_diameter_um=float(np.mean(um_values)),
            median_diameter_um=float(np.median(um_values)),
            min_diameter_um=float(np.min(um_values)),
            max_diameter_um=float(np.max(um_values)),
        ))
    return summaries


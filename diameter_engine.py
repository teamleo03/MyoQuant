"""Five arc-length samples; missing boundaries remain missing, never fabricated."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from .centerlines import _interpolate_path
from .config import Settings
from .models import Candidate, DiameterMeasurement, DiameterSummary

FRACTIONS = np.arange(1, 6, dtype=float) / 6


def trace_edge(mask, position, direction, limit):
    steps = np.arange(0, limit + 0.25, 0.25)
    points = position[:, None] + direction[:, None] * steps
    valid = (points[0] >= 0) & (points[0] <= mask.shape[0]-1) & (points[1] >= 0) & (points[1] <= mask.shape[1]-1)
    values = ndi.map_coordinates(mask.astype(np.float32), points, order=1, mode='constant', cval=0)
    exits = np.flatnonzero((values < 0.5) | ~valid)
    if not len(exits):
        return None, 'edge_not_found_within_limit'
    first = int(exits[0])
    if first == 0:
        return None, 'sample_outside_instance'
    if not valid[first]:
        return None, 'ray_reaches_image_border'
    fraction = (values[first-1] - 0.5) / max(values[first-1] - values[first], 1e-9)
    return float(steps[first-1] + 0.25 * np.clip(fraction, 0, 1)), ''


def measure_diameters(labels, candidates: list[Candidate], settings=None, selected_ids=None):
    settings = settings or Settings()
    measurements = []
    slices = ndi.find_objects(labels)
    for candidate in candidates:
        if candidate.status == 'Excluded' or (selected_ids is not None and candidate.id not in selected_ids):
            continue
        candidate.review_reason = candidate.segmentation_reason
        candidate.status = 'Review' if candidate.segmentation_reason else 'Ready'
        if len(candidate.centerline) < 2 or candidate.label > len(slices) or slices[candidate.label-1] is None:
            for i, f in enumerate(FRACTIONS, 1):
                measurements.append(DiameterMeasurement(candidate.id, i, float(f), reason='missing_instance_or_centerline'))
            candidate.status = 'Review'
            candidate.review_reason = ';'.join(filter(None, [candidate.review_reason, 'missing_instance_or_centerline']))
            continue
        sy, sx = slices[candidate.label-1]
        # Full component bounding box + background pad, preserving real image edges.
        y0, x0 = max(0, sy.start-2), max(0, sx.start-2)
        y1, x1 = min(labels.shape[0], sy.stop+2), min(labels.shape[1], sx.stop+2)
        mask = labels[y0:y1, x0:x1] == candidate.label
        path = candidate.centerline - np.array([y0, x0])
        positions, tangents = _interpolate_path(path, FRACTIONS)
        failures = set()
        for i, (fraction, position, tangent) in enumerate(zip(FRACTIONS, positions, tangents), 1):
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            normal /= max(np.linalg.norm(normal), 1e-12)
            item = DiameterMeasurement(candidate.id, i, float(fraction),
                                       center_x=float(position[1]+x0), center_y=float(position[0]+y0))
            left, reason1 = trace_edge(mask, position, -normal, settings.max_width)
            right, reason2 = trace_edge(mask, position, normal, settings.max_width)
            if left is None or right is None or left <= 0 or right <= 0:
                item.reason = ';'.join(sorted(set(filter(None, [reason1, reason2, 'invalid_boundary_intersection']))))
                failures.add(item.reason)
            else:
                a = position - normal * left
                b = position + normal * right
                item.endpoint1_x, item.endpoint1_y = float(a[1]+x0), float(a[0]+y0)
                item.endpoint2_x, item.endpoint2_y = float(b[1]+x0), float(b[0]+y0)
                item.diameter_px = float(left + right)
                item.diameter_um = item.diameter_px / settings.pixels_per_um
                item.status = candidate.status
                item.reason = candidate.segmentation_reason
            measurements.append(item)
        if failures:
            candidate.status = 'Review'
            candidate.review_reason = ';'.join(filter(None, [candidate.segmentation_reason, *sorted(failures)]))
    return measurements, recompute_summaries(measurements, settings, candidates)


def recompute_summaries(measurements, settings=None, candidates=None):
    lookup = {c.id: c for c in candidates or []}
    grouped = {}
    for item in measurements:
        grouped.setdefault(item.myotube_id, {})[item.measurement_id] = item
    summaries = []
    for key, items in sorted(grouped.items()):
        values = tuple(items[i].diameter_um if i in items else None for i in range(1, 6))
        valid = [v for v in values if v is not None and np.isfinite(v) and v > 0]
        complete = len(valid) == 5
        candidate = lookup.get(key)
        reason = candidate.review_reason if candidate else ''
        review = not complete or (candidate is not None and candidate.status != 'Ready') or any(m.status != 'Ready' for m in items.values())
        if not complete:
            reason = ';'.join(filter(None, [reason, 'incomplete_five_measurements']))
        if any(m.method == 'manual' for m in items.values()):
            reason = ';'.join(filter(None, [reason, 'manual_endpoint_edit']))
            review = True
        mean = float(np.mean(valid)) if complete else None
        sd = float(np.std(valid, ddof=1)) if complete else None
        summaries.append(DiameterSummary(key, len(valid), mean, sd,
                                         100 * sd / mean if complete and mean > 0 else None,
                                         'Review' if review else 'Ready', reason, values))
    return summaries

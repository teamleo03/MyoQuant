from __future__ import annotations
from pathlib import Path
import csv
import cv2
import numpy as np
from skimage import measure
from .legacy_models import Candidate, DiameterMeasurement, DiameterSummary
from .image_io import write_image

def make_overlay(image_bgr: np.ndarray, accepted: np.ndarray, candidates: list[Candidate],
                 measurements: list[DiameterMeasurement] | None = None,
                 selected_ids: set[int] | None = None) -> np.ndarray:
    selected_ids = selected_ids or set()
    overlay = image_bgr.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 255
    alpha = accepted.astype(np.float32)[..., None] * 0.22
    overlay = (overlay * (1 - alpha) + tint * alpha).astype(np.uint8)

    component_labels = measure.label(accepted, connectivity=2)
    for candidate in candidates:
        cx = int(np.clip(round(candidate.centroid_x), 0, accepted.shape[1] - 1))
        cy = int(np.clip(round(candidate.centroid_y), 0, accepted.shape[0] - 1))
        component_label = int(component_labels[cy, cx])
        if component_label == 0:
            continue
        component = np.uint8(component_labels == component_label) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = (0, 255, 0) if candidate.id in selected_ids else (0, 255, 255)
        thickness = 3 if candidate.id in selected_ids else 2
        cv2.drawContours(overlay, contours, -1, color, thickness)

    if measurements:
        for item in measurements:
            p1 = (int(round(item.endpoint1_x)), int(round(item.endpoint1_y)))
            p2 = (int(round(item.endpoint2_x)), int(round(item.endpoint2_y)))
            center = (int(round(item.center_x)), int(round(item.center_y)))
            cv2.line(overlay, p1, p2, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(overlay, center, 2, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, p1, 5, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(overlay, p2, 5, (255, 255, 0), -1, cv2.LINE_AA)

    for c in candidates:
        if c.id in selected_ids:
            cv2.putText(overlay, str(c.id), (int(c.centroid_x), int(c.centroid_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2, cv2.LINE_AA)

    # Red target markers identify the junction shared by split overlap arms.
    # They make it clear why the adjoining strands have separate IDs and why
    # no cyan measurement line is placed through the crossing itself.
    shown_junctions: set[tuple[int, int, int]] = set()
    for candidate in candidates:
        if candidate.id not in selected_ids:
            continue
        if not candidate.overlap_detected or candidate.overlap_x is None or candidate.overlap_y is None:
            continue
        key = (candidate.overlap_group, round(candidate.overlap_x), round(candidate.overlap_y))
        if key in shown_junctions:
            continue
        shown_junctions.add(key)
        center = (int(round(candidate.overlap_x)), int(round(candidate.overlap_y)))
        cv2.drawMarker(overlay, center, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
    return overlay



def save_results(output_dir: str | Path, image_bgr: np.ndarray, score: np.ndarray,
                 raw_mask: np.ndarray, accepted: np.ndarray, candidates: list[Candidate],
                 measurements: list[DiameterMeasurement] | None = None,
                 summaries: list[DiameterSummary] | None = None,
                 selected_ids: set[int] | None = None) -> dict:
    measurements = measurements or []
    summaries = summaries or []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = make_overlay(image_bgr, accepted, candidates, measurements, selected_ids=selected_ids)
    paths = {
        'overlay': output_dir / 'overlay_with_diameters.png',
        'score': output_dir / 'stain_score.png',
        'raw_mask': output_dir / 'raw_mask.png',
        'accepted_mask': output_dir / 'accepted_mask.png',
        'candidates_csv': output_dir / 'candidates.csv',
        'diameters_csv': output_dir / 'diameter_measurements.csv',
        'summary_csv': output_dir / 'myotube_summary.csv',
    }
    write_image(paths['overlay'], overlay)
    write_image(paths['score'], np.uint8(np.clip(score, 0, 1) * 255))
    write_image(paths['raw_mask'], np.uint8(raw_mask) * 255)
    write_image(paths['accepted_mask'], np.uint8(accepted) * 255)

    with open(paths['candidates_csv'], 'w', newline='', encoding='utf-8-sig') as f:
        fields = list(Candidate(0,0,0,0,0,0,0,0,0,0,0).row().keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(c.row() for c in candidates)

    with open(paths['diameters_csv'], 'w', newline='', encoding='utf-8-sig') as f:
        fields = list(DiameterMeasurement(0,0,0,0,0,0,0,0,0,0).row().keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(m.row() for m in measurements)

    with open(paths['summary_csv'], 'w', newline='', encoding='utf-8-sig') as f:
        fields = list(DiameterSummary(0,0,0,0,0,0,0,0,0,0).row().keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(s.row() for s in summaries)
    return paths


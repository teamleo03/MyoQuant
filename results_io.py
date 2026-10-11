from __future__ import annotations
import csv
import hashlib
import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from .config import save_config
from .image_io import write_image
from .models import Candidate, DiameterMeasurement, DiameterSummary


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fieldnames = list(fieldnames or (rows[0].keys() if rows else ['image_name', 'myotube_id', 'status', 'review_reason']))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_overlay(image_bgr, labels, candidates, measurements=None, selected_ids=None):
    result = image_bgr.copy()
    slices = ndi.find_objects(labels)
    for c in candidates:
        if 'small_object_requires_review' in c.segmentation_reason:
            continue
        box = slices[c.label-1] if c.label <= len(slices) else None
        if box is None:
            continue
        sy, sx = box
        mask = labels[sy, sx] == c.label
        color = (80, 220, 80) if c.status == 'Ready' else (0, 170, 255) if c.status == 'Review' else (140, 140, 140)
        crop = result[sy, sx]
        crop[mask] = np.clip(crop[mask].astype(float) * .82 + np.array(color) * .18, 0, 255)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 1, offset=(sx.start, sy.start))
        if len(c.centerline) > 1:
            points = np.round(c.centerline[:, ::-1]).astype(np.int32)
            cv2.polylines(result, [points], False, (255, 100, 180), 1, cv2.LINE_AA)
        pixels = np.argwhere(mask)
        yy, xx = pixels[len(pixels) // 2]
        suffix = '*' if selected_ids is not None and c.id in selected_ids else ''
        cv2.putText(result, f'{c.id}{suffix} {c.status}', (int(xx+sx.start), int(yy+sy.start)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    for m in measurements or []:
        if m.diameter_px is None:
            continue
        a = (round(m.endpoint1_x), round(m.endpoint1_y))
        b = (round(m.endpoint2_x), round(m.endpoint2_y))
        cv2.line(result, a, b, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(result, a, 3, (255, 255, 0), -1)
        cv2.circle(result, b, 3, (255, 255, 0), -1)
        cv2.putText(result, f'D{m.measurement_id}', a, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 0), 1)
    return result


def save_results(output_dir, image_bgr, score, raw_mask, labels, candidates,
                 measurements=None, summaries=None, selected_ids=None, settings=None, source_path=None):
    from .config import Settings
    settings = settings or Settings()
    selected_ids = set(selected_ids if selected_ids is not None else [c.id for c in candidates if c.status != 'Excluded'])
    measurements, summaries = measurements or [], summaries or []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_image(output_dir / 'overlay_with_diameters.png', make_overlay(image_bgr, labels, candidates, measurements, selected_ids))
    write_image(output_dir / 'cytoplasm_score.png', np.uint8(np.clip(score, 0, 1) * 255))
    write_image(output_dir / 'raw_mask.png', raw_mask.astype(np.uint8)*255)
    accepted = np.isin(labels, [c.label for c in candidates if c.status != 'Excluded'])
    write_image(output_dir / 'accepted_mask.png', accepted.astype(np.uint8)*255)
    np.savez_compressed(output_dir / 'instance_labels.npz', labels=labels.astype(np.int32))
    from .boundary_evidence import color_evidence
    _, boundary_score, barriers, _ = color_evidence(image_bgr, settings)
    write_image(output_dir / 'boundary_evidence.png', np.uint8(boundary_score*255))
    write_image(output_dir / 'boundary_barriers.png', barriers.astype(np.uint8)*255)
    candidate_fields = [f.name for f in fields(Candidate) if f.name != 'centerline']
    write_csv(output_dir / 'candidates.csv', [c.row() for c in candidates], candidate_fields)
    write_csv(output_dir / 'diameter_measurements.csv', [m.row() for m in measurements], DiameterMeasurement(0, 0, 0).row().keys())
    summary_lookup = {s.myotube_id: s.row() for s in summaries}
    summary_rows = []
    for c in candidates:
        row = summary_lookup.get(c.id)
        if row is None:
            status = c.status if c.status in {'Excluded', 'Review'} else 'NotSelected'
            row = DiameterSummary(c.id, 0, None, None, None, status, c.review_reason or 'not_selected').row()
        summary_rows.append(row)
    write_csv(output_dir / 'myotube_summary.csv', summary_rows, DiameterSummary(0, 0, None, None, None, '', '').row().keys())
    centerline_rows = ({'myotube_id': c.id, 'point_index': i, 'x': float(x), 'y': float(y)}
                       for c in candidates for i, (y, x) in enumerate(c.centerline))
    write_csv(output_dir / 'centerlines.csv', centerline_rows, ['myotube_id', 'point_index', 'x', 'y'])
    write_csv(output_dir / 'manual_review.csv',
              ({'myotube_id': c.id, 'status': c.status, 'reason': c.review_reason,
                'false_merge': '', 'false_split': '', 'related_ids': '', 'reviewer_note': ''} for c in candidates),
              ['myotube_id', 'status', 'reason', 'false_merge', 'false_split', 'related_ids', 'reviewer_note'])
    save_config(settings, output_dir / 'config_used.csv')
    from . import __version__
    import scipy
    import skimage
    metadata = {
        'software_version': __version__, 'algorithm': 'experimental_boundary_constrained',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'image_shape': list(image_bgr.shape),
        'coordinate_system': 'original image pixels; x=column, y=row; zero-based',
        'source_path': str(source_path) if source_path else None,
        'decoded_image_sha256': hashlib.sha256(np.ascontiguousarray(image_bgr).tobytes()).hexdigest(),
        'settings': asdict(settings), 'selected_ids': sorted(selected_ids),
        'numpy': np.__version__, 'opencv': cv2.__version__, 'scipy': scipy.__version__, 'skimage': skimage.__version__,
        'accuracy_validated': False, 'score_meaning': 'heuristic ranking, not calibrated probability',
        'count_candidates': len(candidates), 'count_summaries': len(summaries),
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    return output_dir

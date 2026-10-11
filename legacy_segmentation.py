from __future__ import annotations
import cv2
import numpy as np
from skimage import filters, measure, morphology, segmentation
from .legacy_models import Settings, Candidate
from .color_score import stain_score, _odd
from .legacy_strands import _bridge_collinear_strands, _split_overlap_component

def segment(image_bgr: np.ndarray, settings: Settings | None = None):
    settings = settings or Settings()
    score, field = stain_score(image_bgr, settings)

    block = _odd(settings.local_block)
    local = filters.threshold_local(score, block_size=block, method='gaussian', offset=settings.local_offset)
    values = score[field]
    global_t = filters.threshold_otsu(values) if values.size and np.ptp(values) > 0 else 0.25
    mask = (score > local) & (score > max(global_t * 0.82, 0.13)) & field

    # Prefer continuous cytoplasmic regions; remove isolated nuclei/debris.
    mask = morphology.remove_small_objects(mask, min_size=110)
    # Close narrow nuclear grooves and staining interruptions without joining
    # distant myotubes. The filled mask is the outer cytoplasmic envelope used
    # for diameter measurement; the original image remains unchanged.
    closing_radius = max(1, int(settings.envelope_closing_radius))
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * closing_radius + 1, 2 * closing_radius + 1),
    )
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel) > 0
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, opening_kernel) > 0
    mask = morphology.remove_small_holes(
        mask, area_threshold=max(1, int(settings.nuclear_hole_area))
    )
    # Reconnect interrupted portions of the same fiber only when their nearby
    # skeleton endpoints face each other along the same local direction.
    mask = _bridge_collinear_strands(mask, settings)
    if settings.remove_border:
        mask = segmentation.clear_border(mask)

    labels = measure.label(mask, connectivity=2)
    accepted = np.zeros(mask.shape, dtype=bool)
    candidates = []
    next_id = 1
    overlap_group = 0

    for r in measure.regionprops(labels):
        # Cheap filters run before skeletonization. This avoids building a
        # skeleton for every small stain fragment in a high-resolution field.
        if r.area < settings.min_area or r.axis_major_length < settings.min_length:
            continue

        y0, x0, y1, x1 = r.bbox
        # Work in the component's tight bounding box instead of repeatedly
        # allocating a full-image boolean mask for every labeled region.
        component = r.image
        strand_masks, junctions = _split_overlap_component(component, settings)
        is_overlap = bool(junctions)
        if is_overlap:
            overlap_group += 1

        for strand_mask in strand_masks:
            # Removing a small disk around the junction keeps crossing strands
            # separate. It does not remove the strands themselves: each arm is
            # retained as its own measurable candidate.
            parts = measure.label(strand_mask, connectivity=2)
            for part in measure.regionprops(parts):
                length = float(part.axis_major_length)
                width = max(float(part.axis_minor_length), 1.0)
                aspect = length / width
                if part.area < settings.min_area:
                    continue
                if length < settings.min_length:
                    continue
                if width > settings.max_width:
                    continue
                if aspect < settings.min_aspect_ratio:
                    continue
                if part.eccentricity < settings.min_eccentricity:
                    continue
                if part.solidity < settings.min_solidity:
                    continue

                part_mask = parts == part.label
                accepted[y0:y1, x0:x1][part_mask] = True
                quality = float(np.clip(
                    0.35 * min(aspect / 12, 1) +
                    0.25 * min(length / 400, 1) +
                    0.20 * part.eccentricity +
                    0.10 * min(part.area / 12000, 1) +
                    0.10 * min(part.solidity, 1), 0, 1))
                # A branch junction may yield several arm candidates. Give all
                # of them the same overlap group so the CSV and overlay retain
                # their relationship.
                nearest_junction = min(
                    junctions,
                    key=lambda point: (point[0] - part.centroid[0]) ** 2
                    + (point[1] - part.centroid[1]) ** 2,
                ) if junctions else None
                candidates.append(Candidate(
                    id=next_id, label=next_id, area_px=int(part.area),
                    length_px=length, width_px=width, aspect_ratio=aspect,
                    eccentricity=float(part.eccentricity), solidity=float(part.solidity),
                    centroid_x=float(part.centroid[1] + x0),
                    centroid_y=float(part.centroid[0] + y0),
                    score=quality, overlap_detected=is_overlap,
                    overlap_group=overlap_group if is_overlap else 0,
                    overlap_x=float(nearest_junction[1] + x0) if nearest_junction else None,
                    overlap_y=float(nearest_junction[0] + y0) if nearest_junction else None,
                ))
                next_id += 1

    return score, mask, accepted, candidates


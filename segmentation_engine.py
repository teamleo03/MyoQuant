"""Experimental boundary-constrained instance segmentation at native resolution."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from .boundary_evidence import color_evidence
from .boundary_tracker import bridge_fragments
from .centerlines import _longest_skeleton_path
from .config import Settings
from .models import Candidate
from .watershed_instances import build_instances
from .skeleton_utils import skeleton_degree, prune_spurs


def segment(image_bgr: np.ndarray, settings: Settings | None = None):
    settings = settings or Settings()
    score, boundary_score, barrier, field = color_evidence(image_bgr, settings)
    raw = (score >= settings.cytoplasm_threshold) & ~barrier & field
    # No blanket closing or large hole filling across observed boundaries.
    initial = measure.label(raw, connectivity=2)
    counts = np.bincount(initial.ravel())
    keep = counts >= 20
    keep[0] = False
    mask = keep[initial]
    del initial, counts
    # Compact enclosed holes can be nuclei; long white/yellow separations must
    # remain barriers. This repairs envelopes without filling every gap.
    holes = ndi.binary_fill_holes(mask) & ~mask
    hole_labels = measure.label(holes, connectivity=2)
    fill_ids = [r.label for r in measure.regionprops(hole_labels)
                if r.area < 2600 and r.axis_major_length / max(r.axis_minor_length, 1) < 2.5]
    if fill_ids:
        mask |= np.isin(hole_labels, fill_ids)
    del holes, hole_labels
    mask, added = bridge_fragments(mask, boundary_score, barrier, settings)
    labels, split_labels = build_instances(mask, settings.min_area, boundary_score)
    padded_boundary = np.pad(boundary_score, 1)
    candidates = []
    for region in measure.regionprops(labels):
        y0, x0, y1, x1 = region.bbox
        component = np.pad(region.image, 1)
        # Suppress tiny candidates cheaply before skeleton graph construction.
        path = _longest_skeleton_path(component) if region.area >= settings.min_area else np.empty((0, 2))
        path = path + np.array([y0 - 1, x0 - 1])
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
        length_um = length / settings.pixels_per_um
        width = max(float(region.axis_minor_length), 1)
        aspect = float(region.axis_major_length) / width
        reasons = []
        status = 'Ready'
        if region.area < settings.min_area:
            reasons.append('small_object_requires_review')
        if region.area >= settings.min_area and len(path) < 2:
            status = 'Excluded'
            reasons.append('no_measurable_centerline')
        elif region.area >= settings.min_area and settings.min_length_um > 0 and length_um < settings.min_length_um:
            status = 'Excluded'
            reasons.append('length_below_minimum')
        if status != 'Excluded' and region.area >= settings.min_area:
            if region.label in split_labels:
                reasons.append('crowded_instance_requires_review')
            if aspect < settings.min_aspect_ratio or region.eccentricity < settings.min_eccentricity:
                reasons.append('shape_requires_review')
            skeleton = prune_spurs(morphology.skeletonize(component), ndi.distance_transform_edt(component))
            degree = skeleton_degree(skeleton)
            # Many 8-neighbor staircase corners have degree 3; report only
            # multi-ended or closed topology, rather than cutting every corner.
            endpoints = int(np.count_nonzero(skeleton & (degree == 1)))
            if endpoints != 2:
                reasons.append('ambiguous_branch_or_loop')
            if np.any(added[y0:y1, x0:x1] & region.image):
                reasons.append('reconstructed_gap')
            if y0 == 0 or x0 == 0 or y1 == labels.shape[0] or x1 == labels.shape[1]:
                reasons.append('touches_image_border')
            rim = ndi.binary_dilation(component) & ~component
            local_boundary = padded_boundary[y0:y1+2, x0:x1+2]
            if rim.any() and np.mean(local_boundary[rim] >= settings.boundary_threshold * 0.75) < 0.6:
                reasons.append('weak_outer_boundary')
            if reasons:
                status = 'Review'
        if status == 'Ready' and reasons:
            status = 'Review'
        reason = ';'.join(reasons)
        candidates.append(Candidate(
            id=int(region.label), label=int(region.label), area_px=int(region.area),
            length_px=length, length_um=length_um, width_px=width, aspect_ratio=aspect,
            eccentricity=float(region.eccentricity), solidity=float(region.solidity),
            centroid_x=float(region.centroid[1]), centroid_y=float(region.centroid[0]),
            score=float(np.clip(aspect / 12, 0, 1)), status=status,
            review_reason=reason, segmentation_reason=reason, centerline=path,
        ))
    # The third return value is an int32 instance map, NOT a relabeled binary mask.
    return score, raw, labels, candidates

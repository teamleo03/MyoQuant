from __future__ import annotations
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from .legacy_models import Settings

def _endpoint_outward_direction(skeleton: np.ndarray, endpoint: tuple[int, int]) -> np.ndarray | None:
    """Follow an endpoint inward briefly, then return its outward unit vector."""
    previous = None
    current = endpoint
    for _ in range(8):
        y, x = current
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < skeleton.shape[0] and 0 <= xx < skeleton.shape[1] and skeleton[yy, xx]:
                    if previous != (yy, xx):
                        neighbors.append((yy, xx))
        if not neighbors:
            break
        previous, current = current, neighbors[0]

    vector = np.asarray(endpoint, dtype=float) - np.asarray(current, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else None



def _bridge_collinear_strands(mask: np.ndarray, settings: Settings) -> np.ndarray:
    """Bridge short, collinear endpoint gaps without merging nearby fibers.

    Global closing would indiscriminately fuse parallel myotubes. This instead
    joins only endpoint pairs that face one another and whose centerlines agree
    with the gap direction, which reconnects broken strands while preserving
    nearby distinct fibers.
    """
    max_gap = float(max(2.0, settings.strand_bridge_gap))
    labels = measure.label(mask, connectivity=2)
    endpoints: list[tuple[int, int, int, np.ndarray, float]] = []

    for region in measure.regionprops(labels):
        if region.area < max(20, settings.min_area * 0.25):
            continue
        if region.axis_major_length < max(12, settings.min_length * 0.35):
            continue
        y0, x0, _, _ = region.bbox
        skeleton = morphology.skeletonize(region.image)
        neighbor_count = ndi.convolve(
            skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
            mode='constant', cval=0,
        ) - skeleton.astype(np.uint8)
        for y, x in np.argwhere(skeleton & (neighbor_count == 1)):
            direction = _endpoint_outward_direction(skeleton, (int(y), int(x)))
            if direction is not None:
                endpoints.append((int(y + y0), int(x + x0), int(region.label), direction,
                                  float(region.axis_minor_length)))

    if len(endpoints) < 2:
        return mask

    bridges: list[tuple[float, int, int]] = []
    for first in range(len(endpoints) - 1):
        y1, x1, label1, direction1, width1 = endpoints[first]
        for second in range(first + 1, len(endpoints)):
            y2, x2, label2, direction2, width2 = endpoints[second]
            if label1 == label2:
                continue
            delta = np.asarray([y2 - y1, x2 - x1], dtype=float)
            distance = float(np.linalg.norm(delta))
            if distance < 1.0 or distance > max_gap:
                continue
            gap_direction = delta / distance
            # The endpoint directions must face each other along the gap.
            facing1 = float(np.dot(direction1, gap_direction))
            facing2 = float(np.dot(direction2, -gap_direction))
            if facing1 < 0.82 or facing2 < 0.82:
                continue
            # Prefer the nearest, most collinear valid connection.
            score = distance + 12.0 * (2.0 - facing1 - facing2)
            bridges.append((score, first, second))

    if not bridges:
        return mask

    result = mask.astype(np.uint8, copy=True)
    used_endpoints: set[int] = set()
    for _, first, second in sorted(bridges):
        if first in used_endpoints or second in used_endpoints:
            continue
        y1, x1, _, _, width1 = endpoints[first]
        y2, x2, _, _, width2 = endpoints[second]
        thickness = max(1, int(round(min(width1, width2) * 0.45)))
        cv2.line(result, (x1, y1), (x2, y2), 1, thickness, cv2.LINE_AA)
        used_endpoints.update((first, second))
    return result.astype(bool)



def _skeleton_branch_regions(skeleton: np.ndarray) -> np.ndarray:
    """Label true junction cores in an 8-connected skeleton.

    A normal centerline pixel has two neighbors. A junction has at least three;
    grouping adjacent branch pixels prevents a thick crossing from being counted
    as multiple independent overlaps.
    """
    neighbor_count = ndi.convolve(
        skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
        mode='constant', cval=0,
    ) - skeleton.astype(np.uint8)
    branch_pixels = skeleton & (neighbor_count >= 3)
    return measure.label(branch_pixels, connectivity=2)



def _split_overlap_component(
    component: np.ndarray,
    settings: Settings,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Split a fused/crossing component into retained strand segments.

    The segmentation remains binary and black/white. We skeletonize that mask,
    locate its branch junctions, then cut only a narrow local exclusion zone.
    This prevents a crossing from being treated as one huge object while keeping
    the surrounding strands available for automatic thickness measurements.
    """
    skeleton = morphology.skeletonize(component)
    branch_labels = _skeleton_branch_regions(skeleton)
    if branch_labels.max() == 0:
        return [component], []

    distance = ndi.distance_transform_edt(component)
    exclusion = np.zeros(component.shape, dtype=bool)
    junctions: list[tuple[float, float]] = []
    for region in measure.regionprops(branch_labels):
        y, x = region.centroid
        iy = int(np.clip(round(y), 0, component.shape[0] - 1))
        ix = int(np.clip(round(x), 0, component.shape[1] - 1))
        # Scale the cut to the local strand radius, with conservative limits so
        # narrow myotubes are not over-cut and broad fused regions separate.
        radius = int(np.clip(
            np.ceil(distance[iy, ix]) + settings.overlap_cut_padding,
            4,
            24,
        ))
        core = branch_labels == region.label
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1),
        )
        exclusion |= cv2.dilate(core.astype(np.uint8), kernel) > 0
        junctions.append((float(y), float(x)))

    cut_component = component & ~exclusion
    pieces = measure.label(cut_component, connectivity=2)
    masks = [pieces == region.label for region in measure.regionprops(pieces)]
    # Keep the original object in the rare case that a noisy skeleton creates a
    # false branch but the local cut did not form at least two separate pieces.
    return (masks, junctions) if len(masks) >= 2 else ([component], [])


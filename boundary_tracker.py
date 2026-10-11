"""Conservative, short-gap proposals. Every reconstructed connection is Review."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure, morphology
from .legacy_strands import _endpoint_outward_direction


def bridge_fragments(mask, boundary_score, barrier, settings):
    added = np.zeros_like(mask)
    if settings.strand_bridge_gap == 0:
        return mask, added
    labels = measure.label(mask, connectivity=2)
    endpoints = []
    for region in measure.regionprops(labels):
        if region.area < max(20, settings.min_area * 0.25):
            continue
        skeleton = morphology.skeletonize(np.pad(region.image, 1))
        degrees = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - skeleton
        distance = ndi.distance_transform_edt(np.pad(region.image, 1))
        y0, x0, _, _ = region.bbox
        for y, x in np.argwhere(skeleton & (degrees == 1)):
            direction = _endpoint_outward_direction(skeleton, (int(y), int(x)))
            if direction is not None:
                endpoints.append((np.array([y + y0 - 1, x + x0 - 1]), direction,
                                  region.label, float(distance[y, x])))
    if len(endpoints) < 2:
        return mask, added
    proposals = []
    points = np.array([e[0] for e in endpoints])
    for i, j in cKDTree(points).query_pairs(settings.strand_bridge_gap):
        p, d1, id1, r1 = endpoints[i]
        q, d2, id2, r2 = endpoints[j]
        delta = q - p
        distance = float(np.linalg.norm(delta))
        if id1 == id2 or distance < 1 or max(r1, r2) / max(min(r1, r2), 1) > 1.5:
            continue
        direction = delta / distance
        if np.dot(d1, direction) < 0.95 or np.dot(d2, -direction) < 0.95:
            continue
        y0, x0 = np.maximum(np.minimum(p, q) - int(max(r1, r2)) - 2, 0)
        y1, x1 = np.minimum(np.maximum(p, q) + int(max(r1, r2)) + 3, mask.shape)
        corridor = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.line(corridor, (int(p[1]-x0), int(p[0]-y0)), (int(q[1]-x0), int(q[0]-y0)),
                 1, max(1, int(2 * min(r1, r2) - 1)))
        area = corridor.astype(bool)
        local_labels = labels[y0:y1, x0:x1]
        gap = area & ~mask[y0:y1, x0:x1]
        if not gap.any():
            continue
        if np.any(area & (local_labels != 0) & (local_labels != id1) & (local_labels != id2)):
            continue
        # Never draw through strong white/yellow evidence.
        if np.any(barrier[y0:y1, x0:x1][gap]):
            continue
        if float(np.mean(boundary_score[y0:y1, x0:x1][gap])) > settings.boundary_threshold * 0.75:
            continue
        proposals.append((distance, i, j, y0, y1, x0, x1, gap))
    used = set()
    for _, i, j, y0, y1, x0, x1, gap in sorted(proposals, key=lambda x: x[0]):
        if i in used or j in used:
            continue
        # Don't join a third object, including a previously proposed bridge.
        if np.any(ndi.binary_dilation(gap) & added[y0:y1, x0:x1]):
            continue
        added[y0:y1, x0:x1] |= gap
        used.update((i, j))
    return mask | added, added

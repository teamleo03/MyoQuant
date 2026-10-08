from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import exposure, filters, measure, morphology, segmentation


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


def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 else n + 1


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unable to read image: {path}')
    return image


def resize_for_analysis(image_bgr: np.ndarray, scale: float) -> np.ndarray:
    """Return a smaller working image for fast analysis.

    Measurements are converted with a matching scaled calibration value by the
    application, so the reported micrometre diameters remain on the user's
    original calibration scale.
    """
    scale = float(scale)
    if not 0.10 <= scale <= 1.0:
        raise ValueError('Analysis scale must be between 0.10 and 1.00.')
    if scale >= 0.999:
        return image_bgr
    height, width = image_bgr.shape[:2]
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ext = path.suffix or '.png'
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f'Unable to encode image: {path}')
    encoded.tofile(str(path))


def circular_field_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Suppress dark microscope borders by finding the bright circular field."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (31, 31), 0)
    threshold = max(12, int(np.percentile(smooth, 8)))
    field = smooth > threshold
    field = morphology.remove_small_holes(field, area_threshold=50000)
    field = morphology.remove_small_objects(field, min_size=max(1000, field.size // 15))
    labels = measure.label(field)
    if labels.max() == 0:
        return np.ones(gray.shape, dtype=bool)
    largest = max(measure.regionprops(labels), key=lambda r: r.area)
    result = labels == largest.label
    # OpenCV's binary morphology is substantially faster on large microscope
    # fields than repeatedly applying a scikit-image disk footprint.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    return cv2.erode(result.astype(np.uint8), kernel) > 0


def stain_score(image_bgr: np.ndarray, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Compute a robust purple/blue Jenner–Giemsa score in [0, 1]."""
    image = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    field = circular_field_mask(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Illumination correction: divide by a slowly varying background.
    k = _odd(settings.background_kernel)
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    corrected = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    darkness = 1.0 - corrected.astype(np.float32) / 255.0

    # Jenner–Giemsa myotubes are purple/blue relative to the pale yellow-green field.
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    purple = np.maximum(a, 0.0) + np.maximum(-b, 0.0) * 0.75
    purple = exposure.rescale_intensity(purple, in_range='image', out_range=(0, 1)).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0
    darkness = exposure.rescale_intensity(darkness, in_range='image', out_range=(0, 1)).astype(np.float32)

    # Emphasize stained elongated cytoplasm while reducing pale texture.
    score = 0.58 * purple + 0.27 * darkness + 0.15 * saturation
    score[~field] = 0
    score = cv2.GaussianBlur(score, (3, 3), 0)
    return np.clip(score, 0, 1), field


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



def _longest_skeleton_path(region_mask: np.ndarray) -> np.ndarray:
    """Return an ordered longest geodesic path through a skeleton as (y, x)."""
    skeleton = morphology.skeletonize(region_mask)
    coordinates = np.argwhere(skeleton)
    if len(coordinates) < 2:
        return coordinates.astype(float)

    index = {tuple(point): i for i, point in enumerate(coordinates)}
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]
    neighbors: list[list[int]] = [[] for _ in range(len(coordinates))]
    for i, (y, x) in enumerate(coordinates):
        for dy, dx in neighbor_offsets:
            j = index.get((int(y + dy), int(x + dx)))
            if j is not None:
                neighbors[i].append(j)

    degree = np.asarray([len(items) for items in neighbors], dtype=int)
    endpoints = np.flatnonzero(degree == 1)

    # Overlap regions have already been split before diameter measurement, so
    # almost every selected candidate is an unbranched centerline. Traversing
    # it directly is linear-time and avoids constructing a sparse graph plus
    # running Dijkstra twice for each selected myotube.
    if endpoints.size >= 2 and int(degree.max(initial=0)) <= 2:
        path_indices = [int(endpoints[0])]
        previous = -1
        current = path_indices[0]
        while True:
            following = [node for node in neighbors[current] if node != previous]
            if not following:
                break
            next_node = following[0]
            if next_node in path_indices:
                break
            path_indices.append(next_node)
            previous, current = current, next_node
        return coordinates[np.asarray(path_indices)].astype(float)

    rows, cols, weights = [], [], []
    for i, items in enumerate(neighbors):
        for j in items:
            dy, dx = coordinates[j] - coordinates[i]
            rows.append(i); cols.append(j)
            weights.append(float(np.hypot(dy, dx)))
    graph = csr_matrix((weights, (rows, cols)), shape=(len(coordinates), len(coordinates)))
    start = int(endpoints[0]) if endpoints.size else 0

    distances = dijkstra(graph, directed=False, indices=start)
    finite = np.isfinite(distances)
    if not finite.any():
        return coordinates.astype(float)
    far_a = int(np.argmax(np.where(finite, distances, -1)))

    distances, predecessors = dijkstra(
        graph, directed=False, indices=far_a, return_predecessors=True)
    finite = np.isfinite(distances)
    far_b = int(np.argmax(np.where(finite, distances, -1)))

    path_indices = [far_b]
    current = far_b
    while current != far_a:
        current = int(predecessors[current])
        if current < 0 or current == path_indices[-1]:
            break
        path_indices.append(current)
    path_indices.reverse()
    return coordinates[np.asarray(path_indices)].astype(float)


def _interpolate_path(path: np.ndarray, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate positions and tangents at normalized path-length fractions."""
    if len(path) < 2:
        return path, np.zeros_like(path)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total <= 0:
        return np.repeat(path[:1], len(fractions), axis=0), np.zeros((len(fractions), 2))

    positions, tangents = [], []
    for fraction in fractions:
        target = float(fraction) * total
        i = int(np.searchsorted(cumulative, target, side='right') - 1)
        i = int(np.clip(i, 0, len(path) - 2))
        span = max(cumulative[i + 1] - cumulative[i], 1e-9)
        alpha = (target - cumulative[i]) / span
        point = path[i] * (1 - alpha) + path[i + 1] * alpha

        window = max(3, min(10, len(path) // 12))
        left = max(0, i - window)
        right = min(len(path) - 1, i + window + 1)
        tangent = path[right] - path[left]
        norm = np.linalg.norm(tangent)
        tangent = tangent / norm if norm > 0 else np.array([0.0, 1.0])
        positions.append(point)
        tangents.append(tangent)
    return np.asarray(positions), np.asarray(tangents)


def _distance_to_mask_edge(
    mask: np.ndarray,
    position: np.ndarray,
    direction: np.ndarray,
    maximum_distance: float,
) -> float:
    """Trace one normal ray to the first background pixel, in subpixels."""
    step = 0.5
    distances = np.arange(0.0, maximum_distance + step, step)
    rows = position[0] + direction[0] * distances
    cols = position[1] + direction[1] * distances
    values = ndi.map_coordinates(mask, [rows, cols], order=1, mode='constant', cval=0.0)
    outside = np.flatnonzero(values < 0.5)
    if outside.size == 0:
        return maximum_distance
    first = int(outside[0])
    if first == 0:
        return 0.0
    # Linearly interpolate the binary transition for a less stair-stepped
    # boundary position than a whole-pixel distance-transform diameter.
    previous, current = values[first - 1], values[first]
    fraction = (previous - 0.5) / max(previous - current, 1e-9)
    return float(distances[first - 1] + step * np.clip(fraction, 0.0, 1.0))


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

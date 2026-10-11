from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import morphology

def _longest_skeleton_path(region_mask: np.ndarray) -> np.ndarray:
    """Return an ordered longest geodesic path through a skeleton as (y, x)."""
    skeleton = morphology.skeletonize(region_mask)
    coordinates = np.argwhere(skeleton)
    if len(coordinates) < 2:
        return coordinates.astype(float)

    # Build the same 8-neighbour weighted graph in arrays. Avoid millions of
    # Python dictionary lookups on full-resolution, densely connected images.
    index=np.full(skeleton.shape,-1,np.int32)
    index[coordinates[:,0],coordinates[:,1]]=np.arange(len(coordinates),dtype=np.int32)
    rows,cols,weights=[],[],[]
    for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
        ys=coordinates[:,0]+dy
        xs=coordinates[:,1]+dx
        inside=(ys>=0)&(ys<skeleton.shape[0])&(xs>=0)&(xs<skeleton.shape[1])
        source=np.flatnonzero(inside)
        target=index[ys[inside],xs[inside]]
        found=target>=0
        a,b=source[found],target[found]
        rows.extend([a,b]); cols.extend([b,a])
        w=np.full(len(a),np.hypot(dy,dx),dtype=float)
        weights.extend([w,w])
    graph=csr_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),
                     shape=(len(coordinates),len(coordinates)))
    degree=np.diff(graph.indptr)
    endpoints = np.flatnonzero(degree == 1)

    # Overlap regions have already been split before diameter measurement, so
    # almost every selected candidate is an unbranched centerline. Traversing
    # it directly is linear-time and avoids constructing a sparse graph plus
    # running Dijkstra twice for each selected myotube.
    if endpoints.size >= 2 and int(degree.max(initial=0)) <= 2:
        path_indices = [int(endpoints[0])]
        visited = {path_indices[0]}
        previous = -1
        current = path_indices[0]
        while True:
            neighbors=graph.indices[graph.indptr[current]:graph.indptr[current+1]]
            following = [node for node in neighbors if node != previous]
            if not following:
                break
            next_node = following[0]
            if next_node in visited:
                break
            path_indices.append(next_node)
            visited.add(next_node)
            previous, current = current, next_node
        return coordinates[np.asarray(path_indices)].astype(float)

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


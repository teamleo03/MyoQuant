"""Separate crowded components; preserve unresolved junctions as review evidence."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from .skeleton_utils import skeleton_degree, prune_spurs


def build_instances(mask, min_area):
    labels = measure.label(mask, connectivity=2)
    output = np.zeros(mask.shape, np.int32)
    split_labels = set()
    next_id = 1
    for region in measure.regionprops(labels):
        y0,x0,y1,x1=region.bbox
        local = np.pad(region.image, 2)
        split = False
        parts = local.astype(np.int32)
        if region.area >= min_area:
            distance = ndi.distance_transform_edt(local)
            skeleton = prune_spurs(morphology.skeletonize(local), distance)
            degree = skeleton_degree(skeleton)
            cores = skeleton & (degree >= 3)
            if cores.any():
                # Remove only junctions, not all branches. Consequent fragments
                # are explicitly recorded as strands requiring review.
                core_labels = measure.label(cores,connectivity=2)
                exclusion=np.zeros(local.shape,np.uint8)
                for core in measure.regionprops(core_labels):
                    cy,cx=np.round(core.centroid).astype(int)
                    radius=int(np.clip(np.ceil(distance[cy,cx])+2,3,40))
                    # Ignore tiny spurs near the edge; they are not crossings.
                    if distance[cy,cx] < 3:
                        continue
                    cv2.circle(exclusion,(int(cx),int(cy)),radius,1,-1)
                cut = local & ~exclusion.astype(bool)
                tentative=measure.label(cut,connectivity=2)
                sizable=[p for p in measure.regionprops(tentative) if p.area >= min_area]
                if len(sizable)>=2:
                    parts=tentative
                    split=True
        for part in measure.regionprops(parts):
            if part.area < 20:
                continue
            keep=(parts[2:-2,2:-2]==part.label)
            output[y0:y1,x0:x1][keep]=next_id
            if split:
                split_labels.add(next_id)
            next_id+=1
    return output, split_labels

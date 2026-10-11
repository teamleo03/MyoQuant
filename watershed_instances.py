import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure,morphology,segmentation
from .skeleton_utils import prune_spurs
from .strand_paths import continuous_paths


def build_instances(mask, min_area, boundary_score):
    components=measure.label(mask,connectivity=2)
    markers=np.zeros(mask.shape,np.int32)
    next_id=1
    crowded=set()
    for r in measure.regionprops(components):
        y0,x0,y1,x1=r.bbox
        local=np.pad(r.image,2)
        distance=cv2.distanceTransform(local.astype(np.uint8),cv2.DIST_L2,5)
        paths=[]
        if r.area>=min_area:
            skeleton=prune_spurs(morphology.skeletonize(local),distance)
            paths=continuous_paths(skeleton,distance)
        if not paths:
            y,x=np.unravel_index(np.argmax(distance),distance.shape)
            paths=[np.array([[y,x]],float)]
        local_markers=np.zeros(local.shape,np.int32)
        for path in paths:
            points=np.round(path[:,::-1]).astype(np.int32)
            if len(points)>1:
                cv2.polylines(local_markers,[points],False,next_id,1)
            else:
                x,y=points[0]
                local_markers[y,x]=next_id
            if len(paths)>1:
                crowded.add(next_id)
            next_id+=1
        local_markers[~local]=0
        markers[y0:y1,x0:x1] = np.where(r.image,local_markers[2:-2,2:-2],markers[y0:y1,x0:x1])
    distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    # White/yellow evidence raises watershed barriers; distance resolves ties.
    elevation=boundary_score.astype(np.float32)-.3*np.minimum(distance/30,1)
    labels=segmentation.watershed(elevation,markers,mask=mask,watershed_line=True).astype(np.int32)
    return labels,crowded

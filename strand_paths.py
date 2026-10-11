"""Join skeleton arms by tangent continuity, without excising a disk of cytoplasm."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from .centerlines import _longest_skeleton_path
from .skeleton_utils import skeleton_degree


def continuous_paths(skeleton, distance):
    degree=skeleton_degree(skeleton)
    junctions=measure.label(skeleton&(degree>=3),connectivity=2)
    arms=measure.label(skeleton&(junctions==0),connectivity=2)
    paths=[]
    nodes={}
    for r in measure.regionprops(arms):
        if r.area<5:
            continue
        y0,x0,y1,x1=r.bbox
        path=_longest_skeleton_path(np.pad(r.image,1))+np.array([y0-1,x0-1])
        if len(path)<5:
            continue
        length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
        mid=np.round(path[len(path)//2]).astype(int)
        radius=float(distance[tuple(mid)])
        if length<max(8,radius*1.5):
            continue
        index=len(paths)
        paths.append(path)
        for side in (0,1):
            oriented=path if side==0 else path[::-1]
            point=oriented[0]
            tangent=point-oriented[min(10,len(oriented)-1)]
            norm=np.linalg.norm(tangent)
            if norm==0:
                continue
            y,x=np.round(point).astype(int)
            nearby=junctions[max(0,y-2):y+3,max(0,x-2):x+3]
            node_ids=np.unique(nearby[nearby>0])
            if len(node_ids)==1:
                nodes.setdefault(int(node_ids[0]),[]).append((index,side,tangent/norm,radius))
    connections={}
    for ends in nodes.values():
        proposals=[]
        for i,a in enumerate(ends):
            for b in ends[i+1:]:
                opposition=-float(np.dot(a[2],b[2]))
                if a[0]!=b[0] and opposition>=.85 and max(a[3],b[3])/max(1,min(a[3],b[3]))<=2.5:
                    proposals.append((-opposition,(a[0],a[1]),(b[0],b[1])))
        for _,a,b in sorted(proposals):
            if a not in connections and b not in connections:
                connections[a]=b
                connections[b]=a
    used=set()
    starts=[(i,s) for i in range(len(paths)) for s in (0,1) if (i,s) not in connections]
    starts += [(i,0) for i in range(len(paths))]
    results=[]
    for index,side in starts:
        if index in used:
            continue
        chunks=[]
        while index not in used:
            used.add(index)
            chunks.append(paths[index] if side==0 else paths[index][::-1])
            following=connections.get((index,1-side))
            if following is None:
                break
            index,side=following
        path=np.concatenate(chunks)
        if len(path)>=5:
            results.append(path)
    return results

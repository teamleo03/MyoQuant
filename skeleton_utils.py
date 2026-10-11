import numpy as np


def skeleton_degree(skeleton):
    """Count graph edges without diagonal shortcuts around a pixel corner."""
    p=np.pad(skeleton,1)
    n,s,w,e=p[:-2,1:-1],p[2:,1:-1],p[1:-1,:-2],p[1:-1,2:]
    degree=n.astype(np.uint8)+s+w+e
    degree+=(p[:-2,:-2]&~n&~w)
    degree+=(p[:-2,2:]&~n&~e)
    degree+=(p[2:,:-2]&~s&~w)
    degree+=(p[2:,2:]&~s&~e)
    return degree*skeleton


def prune_spurs(skeleton, distance):
    result=skeleton.copy()
    for _ in range(3):
        degree=skeleton_degree(result)
        remove=[]
        for y,x in np.argwhere(result&(degree==1)):
            current=(int(y),int(x)); previous=None; path=[]; length=0.
            # The radius at a spur TIP is often one pixel, even on a broad
            # tube. Compare spur length to its junction radius instead.
            for step in range(61):
                y,x=current
                if degree[y,x]>=3:
                    if length <= max(5,min(60,float(distance[y,x])*2)):
                        remove.extend(path)
                    break
                path.append(current)
                neighbors=[]
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
                    yy,xx=y+dy,x+dx
                    if not(0<=yy<result.shape[0] and 0<=xx<result.shape[1]) or not result[yy,xx]:
                        continue
                    if dy and dx and (result[y,xx] or result[yy,x]):
                        continue
                    if (yy,xx)!=previous:
                        neighbors.append((yy,xx))
                if len(neighbors)!=1:
                    break
                following=neighbors[0]
                length+=float(np.hypot(following[0]-y,following[1]-x))
                if length>60:
                    break
                previous,current=current,following
        if not remove:
            break
        yy,xx=np.array(remove).T
        result[yy,xx]=False
    return result

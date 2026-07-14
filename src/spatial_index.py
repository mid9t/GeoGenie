"""
2D k-d tree for geographic points (lon, lat).
 
Supports:
- Building a balanced tree from a list of points
- Nearest-neighbor query
- k-nearest-neighbors query
- Range query (axis-aligned box, e.g. [lon_min, lon_max] x [lat_min, lat_max])
 
Note: for large-scale / production geospatial nearest-neighbor work over
a sphere, consider a proper haversine/geodesic distance and structures
like a Ball Tree or an R-tree. This implementation uses squared Euclidean
distance on (lon, lat) directly, which is fine for small regions or when
you just need a fast, correct baseline.
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence


Point = Tuple[float, float] # (lon, lat)


@dataclass
class kDNode:
    point: Point
    left: Optional['kDNode'] = None  
    right: Optional['kDNode'] = None
    axis: int = 0 # 0 =  split on lon, 1 = split on lat


class KDTree:
    def __init__(self, points: Sequence[Point]):
        pts = list(points)
        self.root = self._build(pts, depth=0)

    # ------- Build --------
    def _build(self, pts: List[Point], depth: int) -> Optional[kDNode]:
        if not pts:
            return None
        
        axis = depth % 2    # alternate lon (0) / lat (1)
        pts.sort(key=lambda p:p[axis])
        mid = (len(pts)-1) // 2
        
        node = kDNode(point=pts[mid], axis=axis)
        node.left = self._build(pts[:mid], depth + 1)
        node.right = self._build(pts[mid + 1:], depth + 1)
        return node
    
    # --------- Nearest Neighbor ---------
    def nearest(self, target: Point) -> Tuple[Optional[Point], float]:
        """Return (closest_point, squared_distance)"""
        best: List[Tuple[float, Point]] = [(math.inf, None)]   # mutabel holder

        def _search(node: Optional[kDNode]):
            if node is None:
                return
            d = _sq_dist(node.point, target)
            if d < best[0][0]:
                best[0] = (d, node.point)
            
            axis = node.axis
            diff = target[axis] - node.point[axis]
            near_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            _search(near_branch)
            # Only explore the far branch if the splitting place is closer
            # than our current best distance (pruning step)
            # Note: diff represents the radius of circle centered at target passing through current node
            # After recrursively searching the near side, if the closest point so far is farthur than
            # the spliting line, we need to search the farther side, since there could be a point in area 
            # bounded by the splitting line and a circle centered at target passing through the best point so far.
            # 
            if diff * diff < best[0][0]:
                _search(far_branch)
            
        _search(self.root)
        dist, point = best[0]
        return point, dist
    
    # --------- K Nearest Neighbor ---------
    def k_nearest(self, target: Point, k: int) -> List[Tuple[float, Point]]:
        """
        Retrurn up to k (squared_dstance, point) paris, sorted nearest-first
        Uses a max-head of size k so we only keep the k best candidates seen.
        """
        # heap stores (distance, popint) so heapd (a min-heap) acts as max-heap
        heap: List[Tuple[float, Point]] = []

        def _search(node: Optional[kDNode]):
            if node is None:
                return
            d = _sq_dist(node.point, target)

            if len(heap) < k:
                heapq.heappush(heap, (-d, node.point))
            elif d < heap[0][0]:
                heapq.heapreplace(heap, (-d, node.point))
            
            axis = node.axis
            diff = target[axis] - node.point[axis]
            near_branch = node.left if diff < 0 else node.right
            far_branch = node.right if diff < 0 else node.left

            _search(near_branch)

            # Prune far branch unless it could sitll contain a closer point 
            # than our current worst kept condition

            worst_kept = -heap[0][0] if len(heap) == k else math.inf
            if diff * diff < worst_kept:
                _search(far_branch)

        _search(self.root)
        result = [(-neg_d, p) for neg_d, p in heap]
        result.sort(key=lambda x: x[0])
        return result
    
    # --------- Range Query ---------
    def range_query(
            self, lon_range: Tuple[float, float], lat_range: Tuple[float, float]
    ) -> List[Point]:
        """
        Return all points with lon in [lon_min, lon_max] and 
        lat in [lat_min, lat_max]
        """
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        found: List[Point] = []

        def _search(node: Optional[kDNode]):
            if node is None:
                return
            lon, lat = node.point
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                found.append(node.point)

            axis = node.axis
            lo, hi = (lon_min, lon_max) if axis == 0 else (lat_min, lat_max)
            val = node.point[axis]

            # Only descend into subtree whose region could overlap the box
            if lo <= val:
                _search(node.left)
            if hi >= val:
                _search(node.right)
            
        _search(self.root)
        return found
        

def _sq_dist(a: Point, b: Point) -> float:
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


# ---------------- Example usage ----------------
if __name__ == "__main__":
    cities = [
        (-122.42, 37.77),  # San Francisco
        (-74.01, 40.71),   # New York
        (-0.13, 51.51),    # London
        (139.69, 35.69),   # Tokyo
        (2.35, 48.85),     # Paris
        (13.40, 52.52),    # Berlin
        (-118.24, 34.05),  # Los Angeles
        (151.21, -33.87),  # Sydney
        (37.62, 55.75),    # Moscow
        (77.21, 28.61),    # Delhi
    ]
 
    tree = KDTree(cities)
 
    query_point = (-70.0, 41.0)  # somewhere near the US East Coast
 
    nn_point, nn_dist = tree.nearest(query_point)
    print("Nearest:", nn_point, "sq_dist:", round(nn_dist, 4))
 
    knn = tree.k_nearest(query_point, k=3)
    print("3-NN:", [(round(d, 4), p) for d, p in knn])
 
    box = tree.range_query(lon_range=(-130, -60), lat_range=(20, 55))
    print("Points in North America bounding box:", box)
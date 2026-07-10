'''
To do: Impelemnt 
1) k-d tree
2) R-tree
3) Synthetic POI dataset generator (lat/lon + attributes: category, accessiblity, noise level, hours)
'''

def build_kd_tree(points, depth=0):
    '''
    Builds a k-d tree from a list of points. Each point is represented as a tuple of coordinates (x, y). 
    The function recursively partitions the points based on the median value along the current axis (x or y) 
    at each depth level. The resulting tree structure allows for efficient spatial queries such as nearest 
    neighbor searches and range queries.
    '''
    # select axis based on depth so that axis cycles through all valid valus
    axis = depth % len(points)
    
    # Sort point list and choose median as pivot element 

    # Create node and construct subtrees
    
    None

def build_r_tree(points):
    None

def generate_synthetic_poi_dataset(num_points):
    None
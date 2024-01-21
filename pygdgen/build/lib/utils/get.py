import numpy as np
from scipy.spatial import ConvexHull


def distance(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1-vector2))) 


def get_radius(name):
    if type(name) == int:
        if name == 1000:
            return 17.00     #16.64
        elif name == 10000:
            return 37.00     #35.36   
        elif name == 100000:
            return 78.00     #76.97
        elif name == 100:
            return 9.00      #8.33  
    elif type(name) == str:
        matrix = np.loadtxt(name, skiprows=9)[:, 2:]

        # compute the convex hull of the coordinates
        hull = ConvexHull(matrix)
        # find the indices of the two vertices that are farthest apart
        dists = np.sqrt(np.sum((matrix[hull.vertices][:, np.newaxis, :] - matrix[hull.vertices][np.newaxis, :, :])**2, axis=-1))
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        # get the farthest distance and the corresponding points
        farthest_distance = dists[i, j]
        farthest_points = matrix[hull.vertices[[i, j]]]
        print("Farthest distance:", farthest_distance)
        print("Farthest points:", farthest_points)

        # return max(np.max(matrix, axis=0) - np.min(matrix, axis=0)) / 2

        # revise factor = 0.9
        if name == 'task/special_wulff/special320.xyz':
            revise_factor = 0.95
        elif name == 'task/special_wulff/special310.xyz':
            revise_factor = 0.9
        elif name == 'task/special_wulff/special210.xyz':
            revise_factor = 0.9

        return (farthest_distance / 2) * revise_factor


def get_max_dis(ordinate_matrix):
    # Calculate the squared distance from the origin for each point
    squared_distances = np.sum(ordinate_matrix**2, axis=1)
    
    # Find the maximum squared distance
    max_squared_distance = np.max(squared_distances)
    
    # Take the square root to get the actual maximum distance
    max_distance = np.sqrt(max_squared_distance)
    
    return max_distance


def get_natom(name):
    if type(name) == int:
        if name == 1000:
            return 941
        elif name == 10000:
            return 9585
        elif name == 100000:
            return 99769
        elif name == 100:
            return 93
    elif type(name) == str:
        data = open(name, 'r').readlines()
        return int([item for item in data if item.strip()][1].split()[0])
    

def get_ordinate_matrix(name):
    if type(name) == int:
        if name == 1000:
            path = 'data/Ag_wulff_dpgen_1000t.xyz'
        elif name == 10000:
            path = 'data/Ag_wulff_dpgen_10000t.xyz'
        elif name == 100000:
            path = 'data/Ag_wulff_dpgen_100000t.xyz'
        elif name == 100:
            path = 'data/Ag_wulff_dpgen_100t.xyz'
        data = np.loadtxt(path, skiprows=9)[:, 2:]
    elif type(name) == str:
        with open(name, 'r') as f:
            lines = filter(lambda x: x.strip(), f)
            data = np.loadtxt(lines, skiprows=7)[:, 2:]
    return data


def calculate_rotate_matrix(v1, v2):
    normal_plane = np.cross(np.array(v1), np.array(v2))
    normal_plane = normal_plane / np.linalg.norm(normal_plane)
    print(normal_plane)
    angle = np.arccos(np.dot(np.array(v1), np.array(v2)) / (np.linalg.norm(np.array(v1)) * np.linalg.norm(np.array(v2))))
    w, [x, y, z] = np.cos(angle/2), np.sin(angle/2)*normal_plane
    print(angle)
    rotate_matrix = [[1-2*y**2-2*z**2, 2*x*y-2*w*z, 2*x*z+2*w*y],\
                     [2*x*y+2*w*z, 1-2*x**2-2*z**2, 2*y*z-2*w*x],\
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x**2-2*y**2]] 
    return rotate_matrix
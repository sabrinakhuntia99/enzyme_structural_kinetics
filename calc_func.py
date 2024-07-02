from scipy.spatial import ConvexHull, Delaunay
import numpy as np
def convex_hull_peeling(points, num_layers=10, min_points_percentage=0.1):
    hull = ConvexHull(points)
    vertices = np.copy(hull.vertices)
    hull_points = np.copy(points[vertices])

    # Construct the original hull
    original_hull = [hull_points]

    # Recursively construct scaled-down hulls
    def scale_hull(points, layer_num):
        if layer_num >= num_layers or len(points) <= min_points_percentage * len(original_hull[0]):
            return []
        else:
            scale_factor = 1 - (layer_num / num_layers)  # Decreasing scale factor
            scaled_points = scale_points(points, scale_factor)
            new_hull = ConvexHull(scaled_points)
            vertices = np.copy(new_hull.vertices)
            hull_points = np.copy(scaled_points[vertices])
            return [hull_points] + scale_hull(hull_points, layer_num + 1)

    scaled_hulls = scale_hull(hull_points, 1)

    return original_hull + scaled_hulls
def calculate_centroid(points):
    centroid = np.mean(points, axis=0)
    return centroid

def calculate_distance(point, centroid):
    return np.linalg.norm(point - centroid)

def calculate_average_distance(data):
    average_distances = [0] * len(data)
    for i, timepoint_data in enumerate(data):
        centroid = calculate_centroid(data)
        distances = [calculate_distance(point, centroid) for point in timepoint_data]
        average_distances[i] = np.mean(distances)
    return average_distances

def scale_points(points, scale_factor):
    centroid = np.mean(points, axis=0)
    scaled_points = centroid + scale_factor * (points - centroid)
    return scaled_points

def delaunay_tessellation_peeling(points):
    delaunay = Delaunay(points)
    simplices = np.copy(delaunay.simplices)
    tessellation_points = np.copy(points[simplices])
    layers = [tessellation_points]

    min_points_percentage = 0.1  # Adjust this percentage as needed
    min_points = int(len(points) * min_points_percentage)

    while len(tessellation_points) > min_points:
        try:
            new_delaunay = Delaunay(tessellation_points)
            simplices = np.copy(new_delaunay.simplices)
            tessellation_points = np.copy(points[simplices])
            layers.append(tessellation_points)
        except:
            break

    return layers
def protein_width(points):
    max_width = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            if distance > max_width:
                max_width = distance
    return max_width

def average_thickness(layers):
    thicknesses = []
    for i in range(1, len(layers)):
        layer_thickness = []
        for point in layers[i]:
            distances = [np.linalg.norm(point - prev_point) for prev_point in layers[i-1]]
            layer_thickness.append(min(distances))
        avg_thickness = np.mean(layer_thickness)
        thicknesses.append(avg_thickness)
    return thicknesses

def defined_convex_hull_peeling(points):
    # Calculate the protein width
    width = protein_width(points)

    # Calculate thickness for each layer
    layer_thickness = width / 4

    # Initialize layers list
    defined_layers = [points]

    # Create 4 additional hull layers with the same thickness
    for _ in range(3):
        try:
            # Create convex hull for current points
            hull = ConvexHull(defined_layers[-1])
            # Extrude hull points by layer_thickness
            extruded_points = defined_layers[-1] + hull.equations[:, :-1] * layer_thickness
            defined_layers.append(extruded_points)
        except:
            break

    return defined_layers
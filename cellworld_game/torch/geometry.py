from .device import default_device
import shapely as sp
import torch


def normalize_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def polygons_to_sides(polygons):
    vertices = []
    vertices_sides = []

    def add_vertex(vertex):
        i = 0
        for i, v in enumerate(vertices):
            if v.distance(vertex) <= .001:
                break
        else:
            i = len(vertices)
            vertices.append(vertex)
            vertices_sides.append([])
        return i

    sides_vertices = []

    def find_side(sv):
        for i, (a, b) in enumerate(sides_vertices):
            if (a, b) == sv or (b, a) == sv:
                return i
        return -1

    internal_sides = []
    # Process the exterior ring
    for polygon in polygons:
        exterior_coords = list(polygon)
        origin = add_vertex(sp.Point(exterior_coords[0]))
        vertices_sides.append([])
        point_a = origin
        for i in range(1, len(exterior_coords) - 1):
            point_b = add_vertex(sp.Point(exterior_coords[i]))
            i = find_side((point_a, point_b))
            if i == -1:
                side_number = len(sides_vertices)
                sides_vertices.append((point_a, point_b))
                vertices_sides[point_a].append(side_number)
                vertices_sides[point_b].append(side_number)
            else:
                internal_sides.append(i)
            point_a = point_b
        i = find_side((point_a, origin))
        if i == -1:
            side_number = len(sides_vertices)
            sides_vertices.append((point_a, origin))
            vertices_sides[point_a].append(side_number)
            vertices_sides[origin].append(side_number)
        else:
            internal_sides.append(i)

    filtered_sides_vertices = []
    for i, side_vertices in enumerate(sides_vertices):
        if i not in internal_sides:
            filtered_sides_vertices.append(side_vertices)

    sides_centroids = []
    for a, b in filtered_sides_vertices:
        side = sp.LineString([vertices[a], vertices[b]])
        sides_centroids.append(side.centroid)


    vertices = torch.tensor([[vertex.x, vertex.y]
                             for vertex in vertices],
                            device=default_device)

    sides_centroids = torch.tensor([[centroid.x, centroid.y]
                                    for centroid in sides_centroids],
                                   device=default_device)

    filtered_sides_vertices = torch.tensor(filtered_sides_vertices,
                                           device=default_device)

    return vertices, sides_centroids, filtered_sides_vertices


def get_points_angles(src_tensor: torch.tensor,
                      locations: torch.tensor):
    if locations.numel() == 0:
        return torch.tensor([], device=default_device)
    diff = (locations - src_tensor)
    vertices_angles = torch.atan2(diff[:, 1], diff[:, 0])
    return vertices_angles


def get_points_distances(src_tensor: torch.tensor,
                         locations: torch.tensor):
    if locations.numel() == 0:
        return torch.tensor([], device=default_device)
    distances = torch.sum((locations - src_tensor) ** 2, dim=1)
    return distances


def get_intersections(angles: torch.tensor,
                      segments: torch.tensor):

    A = angles.unsqueeze(1)  # Shape (num_rays, 1)
    V1 = segments[:, 0].unsqueeze(0)  # Shape (1, num_segments)
    V2 = segments[:, 1].unsqueeze(0)  # Shape (1, num_segments)
    diff = torch.abs(V2 - V1)

    # Case 1: V1 < V2 and the difference < PI
    case1 = (V1 < V2) & (diff < torch.pi)
    intersect_case1 = case1 & (A >= V1) & (A <= V2)

    # Case 2: V2 < V1 and the difference < PI
    case2 = (V2 < V1) & (diff < torch.pi)
    intersect_case2 = case2 & (A >= V2) & (A <= V1)

    # Case 3: V1 < V2 and the difference > PI
    case3 = (V1 < V2) & (diff > torch.pi)
    intersect_case3 = case3 & ((A >= V2) | (A <= V1))

    # Case 4: V2 < V1 and the difference > PI
    case4 = (V2 < V1) & (diff > torch.pi)
    intersect_case4 = case4 & ((A >= V1) | (A <= V2))

    # Combine the results from all cases
    intersect = intersect_case1 | intersect_case2 | intersect_case3 | intersect_case4

    return intersect

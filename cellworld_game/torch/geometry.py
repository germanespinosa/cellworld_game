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

    sides = []
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

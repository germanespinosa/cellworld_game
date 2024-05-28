import shapely as sp
import math
from ..util import Point
from ..util import Direction

def theta_in_between(theta: float, start: float, end: float) -> bool:
    if end > start:
        return start < theta < end
    return theta > start or theta < end


def distance2(point1: sp.Point, point2: sp.Point):
    return Point.distance2(src=(point1.x, point1.y), dst=(point2.x, point2.y))


def atan2(src: sp.Point,
          dst: sp.Point):
    return math.atan2(dst.y - src.y, dst.x - src.x)


def move(src: sp.Point,
         theta: float,
         dist: float):
    return sp.Point(src.x + math.cos(theta) * dist, src.y + math.sin(theta) * dist)


def polygons_to_sides(polygons):
    """
    Break a Polygon into individual LineStrings representing its edges.

    Parameters:
    polygon (Polygon): The input Polygon object.
    include_holes (bool): If True, include the interior boundaries (holes) as well.

    Returns:
    List[LineString]: A list of LineString objects representing the edges of the polygon.
    """
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
        sides.append(side)
        sides_centroids.append(side.centroid)

    return sides, sides_centroids, vertices, filtered_sides_vertices

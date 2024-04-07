import math
import typing
import shapely as sp
import colorsys


def distance2(point1: typing.Tuple[float, float],
              point2: typing.Tuple[float, float]):
    return (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2


def generate_distinct_colors(n):
    """
    Generate a list of n distinct RGB colors.

    Parameters:
    n (int): The number of distinct colors to generate.

    Returns:
    List[Tuple[int, int, int]]: A list of n RGB colors.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Evenly distribute hues in the [0, 1) interval
        saturation = 0.7  # Choose a saturation level that avoids white and grays
        lightness = 0.5  # Choose a lightness level that avoids black and white
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert the RGB values from [0, 1] to [0, 255] and round them to integers
        rgb = tuple(round(c * 255) for c in rgb)
        colors.append(rgb)

    return colors


def create_hexagon(center: tuple, diameter: float, angle: float) -> sp.Polygon:
    """
    Create a hexagon centered at the origin, with a specified rotation angle (in degrees)
    and diameter (distance between opposite vertices).

    Parameters:
    angle (float): Rotation angle in degrees.
    diameter (float): Diameter of the hexagon (distance between opposite vertices).

    Returns:
    Polygon: A Shapely Polygon object representing the hexagon.
    """
    radius = diameter / 2
    rotation = math.radians(angle + 90)
    # Generate the points for the hexagon
    points = []
    center_x, center_y = center
    for i in range(6):
        angle_rad = math.radians(60 * i)  # 60 degrees between the points of a hexagon
        x = center_x + radius * math.cos(angle_rad + rotation)
        y = center_y + radius * math.sin(angle_rad + rotation)
        points.append((x, y))

    # Create the hexagon
    hexagon = sp.geometry.Polygon(points)

    return hexagon


def move_point(start: typing.Tuple[float, float], distance: float, direction: float = None, direction_radians: float = None) -> typing.Tuple[float, float]:
    if direction_radians is None:
        direction_radians = math.radians(direction)
    start_x, start_y = start
    delta_x = distance * math.cos(direction_radians)
    delta_y = distance * math.sin(direction_radians)
    return start_x + delta_x, start_y + delta_y


def create_line_string(start: typing.Tuple[float, float], direction: float, distance: float):
    return sp.LineString([start, move_point(start=start,
                                            direction=direction,
                                            distance=distance)])


def polygon_to_linestrings(polygon):
    """
    Break a Polygon into individual LineStrings representing its edges.

    Parameters:
    polygon (Polygon): The input Polygon object.
    include_holes (bool): If True, include the interior boundaries (holes) as well.

    Returns:
    List[LineString]: A list of LineString objects representing the edges of the polygon.
    """
    linestrings = []

    # Process the exterior ring
    exterior_coords = list(polygon.exterior.coords)
    for i in range(1, len(exterior_coords)):
        linestrings.append(sp.LineString([exterior_coords[i - 1], exterior_coords[i]]))
    return linestrings


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
        exterior_coords = list(polygon.exterior.coords)
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


from itertools import combinations


def max_distance_between_vertices(polygon):
    """
    Calculate the maximum distance between any two vertices of a polygon's exterior.

    Parameters:
    polygon (Polygon): The input Polygon object.

    Returns:
    float: The maximum distance between any two vertices of the polygon's exterior.
    """
    # Get the exterior coordinates of the polygon
    exterior_coords = list(polygon.exterior.coords)

    # Initialize the maximum distance to be the minimum possible value
    max_distance = 0

    # Iterate over all combinations of two points among the vertices
    for point1, point2 in combinations(exterior_coords, 2):
        # Calculate the distance between the two points
        distance = sp.Point(point1).distance(point2)

        # Update the maximum distance if the current distance is greater
        if distance > max_distance:
            max_distance = distance

    return max_distance


def theta_in_between(theta, start, end):
    if end > start:
        return start < theta < end
    return theta > start or theta < end


def distance(src, dst):
    return math.sqrt((src[0]-dst[0]) ** 2 + (src[1]-dst[1]) ** 2)


def direction(src, dst):
    return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))


def normalize_direction(direction: float):
    while direction < -180:
        direction += 360
    while direction > 180:
        direction -= 360
    return direction
def direction_difference(direction1: float, direction2: float):
    direction1 = normalize_direction(direction1)
    direction2 = normalize_direction(direction2)
    difference = direction2 - direction1
    if difference > 180:
        difference -= 360
    if difference < -180:
        difference += 360
    return difference


def direction_error_normalization(direction_error: float):
    pi_err = direction_error / 2
    return 1 / (pi_err * pi_err + 1)

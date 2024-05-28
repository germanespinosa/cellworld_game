import bisect
import math
import typing
import shapely as sp
from .geometry import distance2, move, atan2, theta_in_between
from .geometry import polygons_to_sides
from .polygon import Polygon
from ..interfaces import IVisibility
from ..util import Point


class Visibility(IVisibility):
    def __init__(self, arena: sp.Polygon, occlusions: typing.List[sp.Polygon]):
        self.walls, self.walls_centroids, self.vertices, self.walls_vertices = polygons_to_sides(occlusions + [arena])
        arena_vertices = [sp.Point(c) for c in arena[1:]]
        self.max_distance = max([v1.distance(v2) for v1 in arena_vertices for v2 in arena_vertices])

    def walls_by_distance(self,
                          src: sp.Point):
        return sorted([(side_number,
                        vertices,
                        distance2(src, self.walls_centroids[side_number]))
                       for side_number, vertices
                       in enumerate(self.walls_vertices)],
                      key=lambda item: item[2])

    def intersection(self,
                     src: sp.Point,
                     theta: float,
                     walls_by_distance=None):

        if walls_by_distance is None:
            walls_by_distance = self.walls_by_distance(src=src)

        ray_end = move(src, theta, self.max_distance)
        ray = sp.LineString([src, ray_end])

        for side_number, vertices, distance in walls_by_distance:
            side = self.walls[side_number]
            if ray.intersects(side):
                return ray.intersection(side)
        return None

    def line_of_sight_multiple(self,
                               src: Point.type,
                               dst):
        return [self.line_of_sight(dst) for point in dst]

    def line_of_sight(self,
                      src: typing.Union[sp.Point, Point.type],
                      dst: typing.Union[sp.Point, Point.type],
                      walls_by_distance=None) -> bool:
        if isinstance(src, tuple):
            src = sp.Point(*src)

        if isinstance(dst, tuple):
            dst = sp.Point(*dst)

        if walls_by_distance is None:
            walls_by_distance = self.walls_by_distance(src=src)

        target_distance = distance2(src,dst)
        ray = sp.LineString([src, dst])
        for side_number, vertices, distance in walls_by_distance:
            if distance >= target_distance:
                return True
            if ray.intersects(self.walls[side_number]):
                return False
        return True

    def get_visibility_polygon(self,
                               src: Point.type,
                               direction: float,
                               view_field: float = 360):

        start = sp.Point(src)

        if view_field >= 360:
            view_field_radians = None
            direction_radians = None
            view_field_start = None
            view_field_end = None
        else:
            view_field_radians = math.radians(view_field)
            direction_radians = math.radians(direction)
            view_field_start = direction_radians - view_field / 2
            if view_field_start < -math.pi:
                view_field_start += math.pi * 2
            view_field_end = direction_radians + view_field / 2
            if view_field_end > math.pi:
                view_field_end += math.pi * 2

        def theta_in_view_field(theta):
            if view_field_radians is None:
                return True
            return theta_in_between(theta, view_field_start, view_field_end)

        walls_by_distance = self.walls_by_distance(src=start)

        # contains the relative angle in radians between the start point and the occlusion_vertices
        vertices_theta: typing.List[float] = []
        # contains the relative distance between the start point and the occlusion_vertices
        vertices_distances: typing.List[typing.Optional[float]] = []
        # contains the cached visibility information, if None, it must be computed
        vertices_visible: typing.List[typing.Optional[bool]] = []
        for vertex in self.vertices:
            vertex_theta = atan2(start, vertex)
            vertices_theta.append(vertex_theta)
            if theta_in_view_field(vertex_theta):
                vertices_distances.append(distance2(start, vertex))
                vertices_visible.append(None)
            else:
                vertices_distances.append(None)
                vertices_visible.append(False)

        # creates an index of vertices sorted by relative angle to accelerate search
        vertices_by_thetas = sorted([(vertex_number, theta)
                                     for vertex_number, theta
                                     in enumerate(vertices_theta)],
                                    key=lambda item: item[1])

        #  used to track zones in the view field that are
        # covered by occlusion sides. this speeds up the rendering as everything behind
        # them would be tagged as not visible
        def set_vertices_to_not_visible(theta_start, theta_end, distance):
            start_index = bisect.bisect_left(vertices_by_thetas,
                                             theta_start,
                                             key=lambda x: x[1])
            end_index = bisect.bisect_left(vertices_by_thetas,
                                           theta_end,
                                           key=lambda x: x[1])
            if theta_start < theta_end:
                for vertex_number, theta in vertices_by_thetas[start_index: end_index]:
                    if (vertices_visible[vertex_number] is None and
                       vertices_distances[vertex_number] > distance and
                       theta_in_between(theta, theta_start, theta_end)):
                        vertices_visible[vertex_number] = False
            else:
                for vertex_number, theta in vertices_by_thetas[start_index:]:
                    if (vertices_visible[vertex_number] is None and
                       vertices_distances[vertex_number] > distance and
                       theta_in_between(theta, theta_start, theta_end)):
                        vertices_visible[vertex_number] = False
                for vertex_number, theta in vertices_by_thetas[:end_index]:
                    if (vertices_visible[vertex_number] is None and
                       vertices_distances[vertex_number] > distance and
                       theta_in_between(theta, theta_start, theta_end)):
                        vertices_visible[vertex_number] = False

        def solve_thetas(vertices):
            index1, index2 = vertices
            theta1 = vertices_theta[index1]
            theta2 = vertices_theta[index2]
            if theta1 > theta2:
                theta1, theta2 = theta2, theta1
            d1 = theta2 - theta1
            if d1 > math.pi:
                return (index2, theta2), (index1, theta1)
            else:
                return (index1, theta1), (index2, theta2)

        def is_vertex_visible(src: sp.Point,
                              vertex_number,
                              sorted_sides) -> bool:
            dst = self.vertices[vertex_number]
            target_distance = distance2(src, dst)
            ray = sp.LineString([src, dst])
            for side_number, vertices, distance in sorted_sides:
                if vertex_number in vertices:
                    continue
                if distance >= target_distance:
                    return True
                if ray.intersects(self.walls[side_number]):
                    return False
            return True

        connections = []
        for _, vertices, distance in walls_by_distance:
            (index_a, theta_a), (index_b, theta_b) = solve_thetas(vertices)

            if vertices_visible[index_a] is None:
                vertices_visible[index_a] = is_vertex_visible(start, index_a, walls_by_distance)
            a_visible = vertices_visible[index_a]

            if vertices_visible[index_b] is None:
                vertices_visible[index_b] = is_vertex_visible(start, index_b, walls_by_distance)
            b_visible = vertices_visible[index_b]

            if a_visible and b_visible:
                # the whole side of the occlusion is visible, I'll cover that theta range for future use
                set_vertices_to_not_visible(theta_a, theta_b, distance)
                connections.append((index_a, index_b))

        final_vertices = []

        def is_connected(a, b):
            for ca, cb in connections:
                if ca == a and cb == b:
                    return True
                if ca == b and cb == a:
                    return True
            return False

        last_vertex_number, last_theta = vertices_by_thetas[-1]
        for vertex_number, theta in vertices_by_thetas:
            if vertices_visible[vertex_number]:
                vertex = self.vertices[vertex_number]
                if not is_connected(last_vertex_number, vertex_number):
                    ext1 = self.intersection(start, last_theta + 0.001, walls_by_distance)
                    final_vertices.append(ext1)
                    ext2 = self.intersection(start, theta - 0.001, walls_by_distance)
                    final_vertices.append(ext2)
                final_vertices.append(vertex)
                last_vertex_number, last_theta = vertex_number, theta
        return Polygon(final_vertices)


import typing
from ..interfaces import IPolygon
import shapely.affinity as spa
import shapely as sp
from ..util import Point


class Polygon(IPolygon):
    def __init__(self, vertices):
        self._sides = len(vertices)
        self._vertices = None
        self._points = None
        self._polygon = None
        if vertices:
            if isinstance(vertices[0], sp.Point):
                self._points = vertices
            else:
                self._vertices = vertices

    def sides(self):
        return self._sides

    @property
    def vertices(self) -> typing.List[Point.type]:
        if self._vertices is None:
            self._vertices = [(p.x, p.y) for p in self.points]
        return self._vertices

    @property
    def points(self) -> typing.List[sp.Point]:
        if self._points is None:
            self._points = [sp.Point(p) for p in self.vertices]
        return self._points

    @property
    def polygon(self):
        if self._polygon is None:
            self._polygon = sp.geometry.Polygon(self.points)
        return self._polygon

    def contains(self, points):
        if isinstance(points, Polygon):
            inside = all([self.polygon.contains(point) for point in points.points])
        else:
            inside = [self.polygon.contains(sp.Point(point)) for point in points]
        return inside

    def intersects(self, other: "Polygon"):
        return self.polygon.intersects(other.polygon)

    def __getitem__(self, item) -> typing.Tuple[float, float]:
        return self.vertices[item]

    def translate_rotate(self,
                         translation: Point.type,
                         rotation: float,
                         rotation_center: Point.type = (0, 0)) -> "Polygon":
        delta_x, delta_y = translation
        rotated_polygon = spa.rotate(self.polygon,
                                     rotation,
                                     origin=rotation_center,
                                     use_radians=False)
        translated_polygon: spa.Polygon = spa.translate(rotated_polygon, delta_x, delta_y)

        return Polygon(vertices=translated_polygon.exterior.coords)

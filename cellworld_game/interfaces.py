from abc import ABC, abstractmethod
from .coordinate_converter import CoordinateConverter
import typing
import math
from .util import Point


class IPoints(ABC):
    @abstractmethod
    def __init__(self, point_list: typing.List[Point.type]) -> None:
        raise NotImplementedError

    @abstractmethod
    def closest(self, point: Point.type) -> int:
        raise NotImplementedError


class IPolygon(ABC):

    @abstractmethod
    def __init__(self, vertices: typing.List[typing.Tuple[float, float]]):
        raise NotImplementedError

    @abstractmethod
    def contains(self, points):
        raise NotImplementedError

    @abstractmethod
    def intersects(self, other):
        raise NotImplementedError

    @abstractmethod
    def sides(self):
        raise NotImplementedError

    @abstractmethod
    def bounds(self) -> typing.Tuple[float, float, float, float]:
        raise NotImplementedError

    def __iter__(self) -> Point.type:
        side_count = self.sides()
        if side_count:
            for i in range(side_count):
                yield self[i]
            yield self[0]

    @abstractmethod
    def __getitem__(self, item) -> Point.type:
        raise NotImplementedError

    @abstractmethod
    def translate_rotate(self,
                         translation: Point.type,
                         rotation: float,
                         rotation_center: Point.type = (0, 0)) -> "Polygon":
        raise NotImplementedError

    @classmethod
    def regular(cls, center: tuple, diameter: float, angle: float, sides: int):
        radius = diameter / 2
        rotation = math.radians(angle + 90)
        step = math.pi * 2 / sides
        # Generate the points for the hexagon
        points = []
        center_x, center_y = center
        for i in range(sides):
            angle_rad = i * step  # 60 degrees between the points of a hexagon
            x = center_x + radius * math.cos(angle_rad + rotation)
            y = center_y + radius * math.sin(angle_rad + rotation)
            points.append((x, y))

        # Create the regular polygon
        return cls(points)

    def render(self,
               surface,
               coordinate_converter: CoordinateConverter,
               color: typing.Tuple[int, int, int]):

        import pygame

        pygame.draw.polygon(surface,
                            color,
                            [coordinate_converter.from_canonical((float(point_x), float(point_y)))
                             for point_x, point_y
                             in self])


class IVisibility(ABC):

    @abstractmethod
    def line_of_sight(self,
                      src: Point.type,
                      dst: Point.type) -> bool:
        raise NotImplementedError

    @abstractmethod
    def line_of_sight_multiple(self,
                               src: Point.type,
                               dst):
        raise NotImplementedError

    @abstractmethod
    def get_visibility_polygon(self,
                               src: Point.type,
                               direction: float,
                               view_field: float = 360) -> IPolygon:
        raise NotImplementedError

    def render(self,
               surface,
               coordinate_converter: CoordinateConverter,
               location: Point.type,
               direction: float,
               view_field: float = 360,
               color: typing.Tuple[int, int, int] = (180, 180, 180)):

        visibility_polygon = self.get_visibility_polygon(src=location,
                                                         direction=direction,
                                                         view_field=view_field)

        visibility_polygon.render(surface=surface,
                                  coordinate_converter=coordinate_converter,
                                  color=color)


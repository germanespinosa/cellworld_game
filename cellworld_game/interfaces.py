from abc import ABC, abstractmethod
from .coordinate_converter import CoordinateConverter
import typing
import math


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
    def __iter__(self) -> typing.Tuple[float, float]:
        while False:
            yield None

    def __getitem__(self, item) -> typing.Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def translate_rotate(self,
                         translation: typing.Tuple[float, float],
                         rotation: float,
                         rotation_center: typing.Tuple[float, float] = (0, 0)) -> "Polygon":
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


class IVisibility(ABC):

    @abstractmethod
    def line_of_sight(self,
                      src: typing.Tuple[float, float],
                      dst: typing.Tuple[float, float]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_visibility_polygon(self,
                               src: typing.Tuple[float, float],
                               direction: float,
                               view_field: float = 360) -> IPolygon:
        raise NotImplementedError

    def render(self,
               surface,
               coordinate_converter: CoordinateConverter,
               location: typing.Tuple[float, float],
               direction: float,
               view_field: float = 360,
               color: typing.Tuple[int, int, int] = (180, 180, 180)):
        raise NotImplementedError

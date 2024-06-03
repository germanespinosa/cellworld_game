import typing
from ..util import Point
from ..interfaces import IPoints
import numpy as np


class Points(IPoints):

    def __init__(self, point_list: typing.List[Point.type]) -> None:
        self.point_list = point_list
        self.points_array = np.array([point if point else (-1, -1) for point in point_list])

    def closest(self, point: Point.type) -> Point.type:
        point_array = np.array(point)
        distances = np.sum((self.points_array - point_array) ** 2, axis=1)
        return snp.argmin(distances)

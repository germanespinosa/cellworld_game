import typing
from ..util import Point
from ..interfaces import IPoints
from .device import default_device
import torch


class Points(IPoints):

    def __init__(self, point_list: typing.List[Point.type]) -> None:
        self.point_list = point_list
        self.points_tensor = torch.tensor([point if point else (-1, -1) for point in point_list], device=default_device)

    def closest(self, point: Point.type) -> Point.type:
        point_tensor = torch.tensor(point, device=default_device)
        distances = torch.sum((self.points_tensor - point_tensor) ** 2, dim=1)
        closest_index = torch.argmin(distances).item()
        return closest_index


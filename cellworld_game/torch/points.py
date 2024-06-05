import typing
from ..util import Point
from ..interfaces import IPoints
from .device import default_device
import torch


class Points(IPoints):

    def __init__(self, point_list: typing.List[Point.type]) -> None:
        definition = 1000
        x = torch.arange(definition, dtype=torch.float32, device=default_device).repeat(definition, 1)
        y = x.T
        map_points = torch.stack([y, x], dim=2) / definition + 1 / definition / 2
        points_tensor = torch.tensor([point if point else (-1, -1) for point in point_list], device=default_device)
        ext_points = points_tensor.unsqueeze(0).unsqueeze(0).repeat(definition, definition, 1, 1).transpose(0, 2)
        diff = map_points - ext_points
        sdist = diff[:, :, :, 0] ** 2 + diff[:, :, :, 1] ** 2
        self.point_list = point_list
        self.point_map = torch.argmin(sdist, dim=0).cpu().numpy()
        self.definition = definition

    def closest(self, point: Point.type) -> int:
        x, y = point
        index_x = max(0, min(int(x * self.definition), self.definition-1))
        index_y = max(0, min(int(y * self.definition), self.definition-1))
        map_index = self.point_map[index_x, index_y]
        return map_index


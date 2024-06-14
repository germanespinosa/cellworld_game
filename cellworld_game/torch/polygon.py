import math

import torch
import typing
from ..interfaces import IPolygon
from ..util import Point
from .device import default_device


class Polygon(IPolygon):
    def __init__(self, vertices):
        if isinstance(vertices, torch.Tensor):
            self.vertices = vertices.to(default_device)
        else:
            self.vertices = torch.tensor(vertices, device=default_device)
        self._sides = len(self.vertices)
        self._vertices_x: torch.tensor = None
        self._vertices_y: torch.tensor = None
        self._edges: torch.tensor = None
        self._bounds = None

    def sides(self):
        return self._sides

    def bounds(self) -> typing.Tuple[float, float, float, float]:
        if self._bounds is None:
            self._bounds = tuple(self.vertices.min(dim=0)[0].tolist()) + tuple(self.vertices.max(dim=0)[0].tolist())
        return self._bounds

    @property
    def vertices_x(self) -> torch.tensor:
        if self._vertices_x is None:
            self._vertices_x = self.vertices[:, 0]
        return self._vertices_x

    @property
    def vertices_y(self) -> torch.tensor:
        if self._vertices_y is None:
            self._vertices_y = self.vertices[:, 1]
        return self._vertices_y

    @property
    def edges(self) -> torch.tensor:
        if self._edges is None:
            self._edges = self.vertices[(torch.arange(self._sides) + 1) % self._sides] - self.vertices
        return self._edges

    def contains(self, points):

        if isinstance(points, Polygon):
            contain_vertices = self.contains(points.vertices)
            return torch.all(contain_vertices)

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=default_device)

        n_points = points.shape[0]
        n_vertices = self.vertices.shape[0]

        # Extract x and y coordinates of the points
        points_x = points[:, 0]
        points_y = points[:, 1]

        # Repeat polygon coordinates for each point
        poly_x_repeated = self.vertices_x.repeat(n_points, 1)
        poly_y_repeated = self.vertices_y.repeat(n_points, 1)

        # Calculate if edges cross the ray extending to the right of each point
        j = torch.arange(n_vertices) - 1
        vertex_1_y = poly_y_repeated[:, j]
        vertex_2_y = poly_y_repeated
        condition_1 = (vertex_1_y > points_y.unsqueeze(1)) != (vertex_2_y > points_y.unsqueeze(1))
        slope = (poly_x_repeated[:, j] - poly_x_repeated) / (poly_y_repeated[:, j] - poly_y_repeated)
        intercept_x = poly_x_repeated + slope * (points_y.unsqueeze(1) - poly_y_repeated)
        condition_2 = points_x.unsqueeze(1) < intercept_x
        # Determine if the number of crossings is odd or even
        crossings = (condition_1 & condition_2).sum(dim=1)
        inside = crossings % 2 == 1
        return inside

    def intersects(self, other: "Polygon"):
        return self.contains(points=other.vertices).any()

    def __getitem__(self, item) -> typing.Tuple[float, float]:
        return tuple(self.vertices[item, :].tolist())

    def translate_rotate(self,
                         translation: Point.type,
                         rotation: float,
                         rotation_center: Point.type = (0, 0)) -> "Polygon":

        t = torch.tensor(translation, device=default_device)
        rc = torch.tensor(rotation_center, device=default_device)
        r = torch.tensor(math.radians(-rotation), device=default_device)
        vertices = self.vertices.clone() - rc
        rotation_matrix = torch.tensor([
            [torch.cos(r), -torch.sin(r)],
            [torch.sin(r), torch.cos(r)]
        ], dtype=torch.float32, device=default_device)
        rotated_points = torch.matmul(vertices, rotation_matrix)
        rotated_points += rc + t
        return Polygon(vertices=rotated_points)

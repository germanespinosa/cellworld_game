import math
import typing
from .visibility import Visibility
import shapely as sp


class Navigation:
    def __init__(self,
                 locations: typing.List[typing.Tuple[float, float]],
                 paths: typing.List[typing.List[int]],
                 visibility: Visibility):
        self.locations: typing.List[typing.Tuple[float, float]] = locations
        self.paths: typing.List[typing.List[int]] = paths
        self.visibility: Visibility = visibility

    def closest_location(self,
                         location: typing.Tuple[float, float]) -> int:
        min_dist2 = math.inf
        closest = None
        for i, l in enumerate(self.locations):
            dist2 = (l[0] - location[0]) ** 2 + (l[1] - location[1]) ** 2
            if dist2 < min_dist2:
                closest = i
                min_dist2 = dist2
        return closest

    def clear_path(self, src, path):
        clear_path = []
        last_step = src
        src_point = sp.Point(last_step)
        walls_by_distance = self.visibility.walls_by_distance(src_point)
        for step in path:
            if not self.visibility.line_of_side(src=src_point, dst=sp.Point(step), walls_by_distance=walls_by_distance):
                clear_path.append(last_step)
                src_point = sp.Point(last_step)
                walls_by_distance = self.visibility.walls_by_distance(src_point)
            last_step = step
        clear_path.append(path[-1])
        return clear_path

    def get_path(self,
                 src: typing.Tuple[float, float],
                 dst: typing.Tuple[float, float]) -> typing.List[typing.Tuple[float, float]]:
        src_index = self.closest_location(location=src)
        dst_index = self.closest_location(location=dst)
        current = src_index
        path_indexes = [current]
        while current is not None and current != dst_index:
            next = self.paths[current][dst_index]
            if next == current:
                break
            current = next
            path_indexes.append(current)
        return self.clear_path(src=src,
                               path=[self.locations[step_index]
                                     for step_index
                                     in path_indexes])

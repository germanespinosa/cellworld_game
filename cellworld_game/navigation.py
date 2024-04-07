import math
import typing
from .visibility import Visibility
import shapely as sp


class Navigation:
    def __init__(self,
                 locations: typing.List[typing.Optional[typing.Tuple[float, float]]],
                 paths: typing.List[typing.List[int]],
                 visibility: typing.List[typing.List[typing.List[int]]]):
        self.locations = locations
        self.paths = paths
        self.visibility = visibility

    def closest_location(self,
                         location: typing.Tuple[float, float]) -> int:
        min_dist2 = math.inf
        closest = None
        for i, l in enumerate(self.locations):
            if l is None:
                continue
            dist2 = (l[0] - location[0]) ** 2 + (l[1] - location[1]) ** 2
            if dist2 < min_dist2:
                closest = i
                min_dist2 = dist2
        return closest

    def clear_path(self, path_indexes):
        src = path_indexes[0]
        clear_path = []
        last_step = src
        src_point = last_step
        for step in path_indexes:
            is_visible = step in self.visibility[src_point]
            if not is_visible:
                clear_path.append(last_step)
                src_point = last_step
            last_step = step
        clear_path.append(path_indexes[-1])
        return [self.locations[s] for s in clear_path]

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
        return self.clear_path(path_indexes=path_indexes)

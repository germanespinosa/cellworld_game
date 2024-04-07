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

    def get_path(self,
                 src: typing.Tuple[float, float],
                 dst: typing.Tuple[float, float]) -> typing.List[typing.Tuple[float, float]]:
        src_index = self.closest_location(location=src)
        dst_index = self.closest_location(location=dst)
        current = src_index
        last_step = src_index
        path_indexes = [last_step]
        while current is not None and current != dst_index:
            next_step = self.paths[current][dst_index]
            if next_step == current:
                break
            is_visible = next_step in self.visibility[last_step]
            if not is_visible:
                path_indexes.append(current)
                last_step = current
            current = next_step
        path_indexes.append(dst_index)
        return [self.locations[s] for s in path_indexes]

import math
import typing
from .util import Point
from .points import Points


class Navigation:
    def __init__(self,
                 locations: typing.List[typing.Optional[Point.type]],
                 paths: typing.List[typing.List[int]],
                 visibility: typing.List[typing.List[typing.List[int]]]):
        self.points = Points(point_list=locations)
        self.paths = paths
        self.visibility = visibility
        self.cache: typing.Dict[typing.Tuple[int, int], typing.List[int]] = {}

    def closest_location(self,
                         location: Point.type) -> int:
        min_dist2 = math.inf
        closest = None
        for i, l in enumerate(self.points.point_list):
            if l is None:
                continue
            dist2 = (l[0] - location[0]) ** 2 + (l[1] - location[1]) ** 2
            if dist2 < min_dist2:
                closest = i
                min_dist2 = dist2
        return closest

    def get_path(self,
                 src: Point.type,
                 dst: Point.type) -> typing.List[Point.type]:
        src_index = self.points.closest(point=src)
        dst_index = self.points.closest(point=dst)
        cache_index = (src_index, dst_index)
        if cache_index in self.cache:
            path_indexes = self.cache[cache_index]
        else:
            current = src_index
            last_step = src_index
            path_indexes = []
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
            self.cache[cache_index] = path_indexes
        return [self.points.point_list[s] for s in path_indexes]

import math
import typing

class Point(object):

    type = typing.Tuple[float, float]

    @staticmethod
    def move(start: "Point.type",
             distance: float,
             direction_degrees: float = None,
             direction_radians: float = None,
             direction_vector: typing.Tuple[float, float] = None) -> typing.Tuple[float, float]:
        if direction_vector is None:
            if direction_radians is None:
                direction_radians = math.radians(direction_degrees)
            start_x, start_y = start
            delta_x = distance * math.cos(direction_radians)
            delta_y = distance * math.sin(direction_radians)
            return start_x + delta_x, start_y + delta_y
        else:
            scaled_vector = Direction.scale_vector(vector=direction_vector,
                                                   scale=distance)
            return Point.add(start=start, vector=scaled_vector)

    @staticmethod
    def add(start: typing.Tuple[float, float],
            vector: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
        return start[0] + vector[0], start[1] + vector[1]

    @staticmethod
    def distance(src: typing.Tuple[float, float],
                 dst: typing.Tuple[float, float]) -> float:
        return math.sqrt((src[0]-dst[0]) ** 2 + (src[1]-dst[1]) ** 2)

    @staticmethod
    def distance2(src: typing.Tuple[float, float],
                  dst: typing.Tuple[float, float]) -> float:
        return (src[0]-dst[0]) ** 2 + (src[1]-dst[1]) ** 2


class Direction:
    vector_type = typing.Tuple[float, float]

    @staticmethod
    def degrees(src: typing.Tuple[float, float],
                dst: typing.Tuple[float, float]) -> float:
        return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))

    @staticmethod
    def radians(src: typing.Tuple[float, float],
                dst: typing.Tuple[float, float]) -> float:
        return math.atan2(dst[1] - src[1], dst[0] - src[0])

    @staticmethod
    def vector(src: typing.Tuple[float, float],
               dst: typing.Tuple[float, float]) -> typing.Tuple[float, float]:
        if src == dst:
            return 0, 0
        distance = Point.distance(src=src, dst=dst)
        return (dst[0] - src[0]) / distance, (dst[1] - src[1]) / distance

    @staticmethod
    def scale_vector(vector: typing.Tuple[float, float],
                     scale: float = 1):
        return vector[0] * scale, vector[1] * scale

    @staticmethod
    def normalize(direction: float) -> float:
        while direction < -180:
            direction += 360
        while direction > 180:
            direction -= 360
        return direction

    @staticmethod
    def difference(direction1: float,
                   direction2: float) -> float:
        direction1 = Direction.normalize(direction1)
        direction2 = Direction.normalize(direction2)
        difference = direction2 - direction1
        if difference > 180:
            difference -= 360
        if difference < -180:
            difference += 360
        return difference

    @staticmethod
    def error_normalization(direction_error: float):
        pi_err = direction_error / 8
        return 1 / (pi_err * pi_err + 1)


class Line(object):
    type = typing.Tuple[Point.type, Point.type]

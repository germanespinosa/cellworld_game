import math
import typing


class Point(object):

    @staticmethod
    def move(start: typing.Tuple[float, float], distance: float, direction: float = None, direction_radians: float = None) -> typing.Tuple[float, float]:
        if direction_radians is None:
            direction_radians = math.radians(direction)
        start_x, start_y = start
        delta_x = distance * math.cos(direction_radians)
        delta_y = distance * math.sin(direction_radians)
        return start_x + delta_x, start_y + delta_y

    @staticmethod
    def distance(src: typing.Tuple[float, float], dst: typing.Tuple[float, float]) -> float:
        return math.sqrt((src[0]-dst[0]) ** 2 + (src[1]-dst[1]) ** 2)


class Direction:
    @staticmethod
    def to(src: typing.Tuple[float, float], dst: typing.Tuple[float, float]) -> float:
        return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))

    @staticmethod
    def normalize(direction: float) -> float:
        while direction < -180:
            direction += 360
        while direction > 180:
            direction -= 360
        return direction

    @staticmethod
    def difference(direction1: float, direction2: float) -> float:
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

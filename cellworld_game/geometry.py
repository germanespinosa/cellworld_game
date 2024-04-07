import math
import shapely as sp


def distance2(point1: sp.Point, point2: sp.Point):
    return (point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2


def atan2(src: sp.Point,
          dst: sp.Point):
    return math.atan2(dst.y - src.y, dst.x - src.x)


def move(src: sp.Point,
         theta: float,
         dist: float):
    return sp.Point(src.x + math.cos(theta) * dist, src.y + math.sin(theta) * dist)

import math
import typing


class CoordinateConverter(object):
    def __init__(self,
                 screen_size: typing.Tuple[int, int],
                 flip_y: bool = False):
        self.screen_size = screen_size
        self.screen_width, self.screen_height = screen_size
        self.flip_y = flip_y
        self.screen_offset = (self.screen_width - self.screen_height) / 2
        self.hexa_ratio = (math.sqrt(3) / 2)

    def scale_to_canonical(self, screen_value: float) -> float:
        return screen_value / self.screen_width

    def scale_from_canonical(self, canonical_value: float) -> float:
        return canonical_value * self.screen_width

    def from_canonical(self, canonical: typing.Union[tuple, float]):
        if isinstance(canonical, float):
            return canonical * self.screen_width

        canonical_x, canonical_y = canonical
        screen_x = canonical_x * self.screen_width
        if self.flip_y:
            screen_y = (1-canonical_y) * self.screen_width - self.screen_offset
        else:
            screen_y = canonical_y * self.screen_width - self.screen_offset
        return screen_x, screen_y

    def to_canonical(self, screen: typing.Union[tuple, float]):
        if isinstance(screen, float):
            return screen / self.screen_width

        screen_x, screen_y = screen
        y = self.screen_height - screen_y + self.screen_offset
        canonical_y = y / self.screen_height * self.hexa_ratio
        canonical_x = screen_x / self.screen_width
        return canonical_x, canonical_y


import pygame
import math
import typing
import shapely as sp
from .model import Model
from .agent import AgentState
from .util import create_line_string


class RayTracing:
    def __init__(self, model: Model,
                 horizontal_view_field: float = 270.0,
                 vertical_view_field: float = 90,
                 perspective_height: float = .025,
                 resolution: int = 1):
        self.model = model
        self.vertical_view_field = vertical_view_field
        self.horizontal_view_field = horizontal_view_field
        self.perspective_height = perspective_height
        self.resolution = resolution
        self.walls = sp.geometry.LineString(model.arena.exterior)
        self.occlusions = [sp.geometry.LineString(occlusion.exterior) for occlusion in model.occlusions]
        self.ray_length = 1.0
        self.occlusion_color: typing.Tuple[int, int, int] = (255, 255, 255)
        self.occlusion_height: float = 0.1
        self.wall_color: typing.Tuple[int, int, int] = (255, 200, 200)
        self.wall_height: float = 0.5

    def distance_factor(self, distance: float) -> float:
        return (self.ray_length - distance) / self.ray_length

    def line_limits(self, perspective_height, line_bottom, line_top, distance, view_field_offset) -> typing.Tuple[float, float]:
        bottom_angle = self.vertical_view_field / 2 + math.degrees(math.atan2(perspective_height, distance))
        top_angle = self.vertical_view_field / 2 - math.degrees(math.atan2(line_top-perspective_height, distance))
        return bottom_angle / self.vertical_view_field, top_angle / self.vertical_view_field

    def render_line(self, screen, screen_x, distance, view_field_offset: float, color, height):
        screen_width, screen_height = screen.get_size()
        distance_factor = self.distance_factor(distance=distance)
        line_color = tuple(channel * distance_factor for channel in color)
        line_start, line_end = self.line_limits(self.perspective_height,
                                                0,
                                                height,
                                                distance,
                                                view_field_offset)
        y_start = screen_height * line_start
        y_ends = screen_height * line_end
        pygame.draw.line(surface=screen,
                         color=line_color,
                         start_pos=(screen_x, y_start),
                         end_pos=(screen_x, y_ends),
                         width=self.resolution)

    def render(self, perspective: AgentState, screen: pygame.Surface):
        screen_width, screen_height = screen.get_size()
        start_angle = perspective.direction - self.horizontal_view_field / 2
        step_angle = self.horizontal_view_field / screen_width
        start = sp.Point(perspective.location)
        screen.fill((0, 0, 0))
        sorted_occlusions = sorted(self.occlusions, key=lambda line: line.distance(start))
        for x in range(0, screen_width, self.resolution):
            view_angle = step_angle * x
            view_field_offset = view_angle - self.horizontal_view_field / 2
            ray_direction = start_angle + view_angle
            screen_x = screen_width - x - 1
            ray = create_line_string(start=perspective.location,
                                     direction=ray_direction,
                                     distance=self.ray_length)
            wall_intersection = ray.intersection(self.walls)
            wall_distance = wall_intersection.distance(start)
            self.render_line(screen=screen,
                             screen_x=screen_x,
                             color=self.wall_color,
                             height=self.wall_height,
                             distance=wall_distance,
                             view_field_offset=view_field_offset)
            closest_distance = wall_distance
            occluded = False
            for occlusion in sorted_occlusions:
                intersection = ray.intersection(occlusion)
                if intersection:
                    for intersection_point in intersection.geoms:
                        distance = intersection_point.distance(start)
                        if distance < closest_distance:
                            closest_distance = distance
                            occluded = True
                    if occluded:
                        break
            if occluded:
                self.render_line(screen=screen,
                                 screen_x=screen_x,
                                 color=self.occlusion_color,
                                 height=self.occlusion_height,
                                 distance=closest_distance,
                                 view_field_offset=view_field_offset)

        pygame.display.flip()

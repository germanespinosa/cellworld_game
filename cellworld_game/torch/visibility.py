import torch
import math
import typing
import shapely as sp
from .geometry import polygons_to_sides, normalize_angle, get_points_angles, get_points_distances, get_intersections
from .device import default_device
from .polygon import Polygon
from ..interfaces import IVisibility
from ..util import Point


class Visibility(IVisibility):
    def __init__(self, arena: sp.Polygon, occlusions: typing.List[sp.Polygon]):
        self.occlusion_vertices, self.occlusions_centroids, self.occlusions_vertices_indices = polygons_to_sides(occlusions)
        self.wall_vertices, self.walls_centroids, self.walls_vertices_indices = polygons_to_sides([arena])

    def line_of_sight_multiple(self,
                               src: Point.type,
                               dst):
        polygon = self.get_visibility_polygon(src=src, direction=0)
        return polygon.contains(dst)

    def line_of_sight(self,
                      src: Point.type,
                      dst: Point.type) -> bool:

        src_tensor = torch.tensor(src, device=default_device)
        dst_tensor = torch.tensor(dst, device=default_device)

        occlusions_vertices_angles = get_points_angles(src_tensor=src_tensor,
                                                       locations=self.occlusion_vertices)

        occlusions_segments_angles = self.__get_segments_vertices_angles__(vertices_angles=occlusions_vertices_angles,
                                                                           segments=self.occlusions_vertices_indices)

        ray = get_points_angles(src_tensor=src_tensor, locations=dst_tensor.unsqueeze(0))

        intersections = get_intersections(angles=ray,
                                          segments=occlusions_segments_angles)[0, :]

        if not intersections.any():
            return True

        occlusions_distances = get_points_distances(src_tensor=src_tensor,
                                                    locations=self.occlusions_centroids[intersections])

        dst_distance = get_points_distances(src_tensor=src_tensor,
                                            locations=dst_tensor.unsqueeze(0))

        if (dst_distance < occlusions_distances).all():
            return True
        else:
            return False

    def __get_segments_vertices_angles__(self,
                                         vertices_angles: torch.tensor,
                                         segments: torch.tensor):
        if segments.numel() == 0:
            return torch.tensor([], device=default_device)
        return torch.stack([normalize_angle((vertices_angles[segments[:, 0]])),
                            normalize_angle((vertices_angles[segments[:, 1]]))], dim=1).to(default_device)

    def __get_segments_vertices_points__(self,
                                         vertices_points: torch.tensor,
                                         segments: torch.tensor):
        if segments.numel() == 0:
            return torch.tensor([], device=default_device)
        return torch.stack([vertices_points[segments[:, 0]],
                            vertices_points[segments[:, 1]]], dim=1).to(default_device)

    def get_visibility_polygon(self,
                               src: Point.type,
                               direction: float,
                               view_field: float = 360):

        theta = math.radians((direction + 180) % 360) - math.pi

        src_tensor = torch.tensor(src, device=default_device)

        occlusions_vertices_angles = get_points_angles(src_tensor=src_tensor,
                                                       locations=self.occlusion_vertices)

        occlusions_segments_angles = self.__get_segments_vertices_angles__(vertices_angles=occlusions_vertices_angles,
                                                                           segments=self.occlusions_vertices_indices)

        occlusions_segments_points = self.__get_segments_vertices_points__(vertices_points=self.occlusion_vertices,
                                                                           segments=self.occlusions_vertices_indices)

        occlusions_distances = get_points_distances(src_tensor=src_tensor,
                                                    locations=self.occlusions_centroids)

        walls_vertices_angles = get_points_angles(src_tensor=src_tensor,
                                                  locations=self.wall_vertices)

        walls_segments_angles = self.__get_segments_vertices_angles__(vertices_angles=walls_vertices_angles,
                                                                      segments=self.walls_vertices_indices)

        walls_segments_points = self.__get_segments_vertices_points__(vertices_points=self.wall_vertices,
                                                                      segments=self.walls_vertices_indices)

        walls_distances = get_points_distances(src_tensor=src_tensor,
                                               locations=self.walls_centroids) + 1.0 # this ensures walls are checked last for intersections

        vertices_angles = torch.cat((occlusions_vertices_angles, walls_vertices_angles), dim=0)
        segments_vertices_angles = torch.cat((occlusions_segments_angles, walls_segments_angles), dim=0)
        segments_vertices_points = torch.cat((occlusions_segments_points, walls_segments_points), dim=0)
        segments_distances = torch.cat((occlusions_distances, walls_distances), dim=0)
        rays = torch.cat((vertices_angles - .001, vertices_angles, vertices_angles + .001), dim=0)

        if view_field < 360:
            d = math.radians(view_field) / 2
            lb = theta - d
            ub = theta + d
            if lb < -math.pi:
                lb += 2 * math.pi
                filtered_thetas = rays[torch.logical_or(rays > lb, rays < ub)]
            elif ub > math.pi:
                ub -= 2 * math.pi
                filtered_thetas = rays[torch.logical_or(rays > lb, rays < ub)]
            else:
                filtered_thetas = rays[(rays > lb) & (rays < ub)]

            rays = torch.cat((torch.tensor([lb, ub], device=default_device), filtered_thetas), dim=0)

        ordered_rays, rays_indices = torch.sort(rays)

        if view_field < 360:
            # saves_the_index of the lower and upper bound ray indices
            lower_bound_index = torch.nonzero((rays_indices == 0), as_tuple=True)[0]
            upper_bound_index = torch.nonzero((rays_indices == 1), as_tuple=True)[0]

        intersections = get_intersections(angles=ordered_rays,
                                          segments=segments_vertices_angles)

        segments_distances = segments_distances.expand_as(intersections)

        intersected_walls_distances = torch.where(intersections,
                                                  segments_distances,
                                                  math.inf).to(default_device)

        closest_intersected_walls = torch.argmin(intersected_walls_distances, dim=1)

        # filters unnecessary vertices
        non_colinear = torch.zeros_like(closest_intersected_walls, dtype=torch.bool)
        # Set the first and last elements to True
        non_colinear[0] = True
        non_colinear[-1] = True
        # Compare each element with its previous and next neighbors
        non_colinear[1:-1] = (closest_intersected_walls[1:-1] != closest_intersected_walls[:-2]) | (closest_intersected_walls[1:-1] != closest_intersected_walls[2:])

        if view_field < 360:
            # do not filter the bounds rays
            non_colinear[lower_bound_index] = True
            non_colinear[upper_bound_index] = True

        filtered_intersected_walls = closest_intersected_walls[non_colinear]
        filtered_rays = ordered_rays[non_colinear]
        filtered_rays_indices = rays_indices[non_colinear]

        xy_v1 = segments_vertices_points[filtered_intersected_walls, 0, :]
        xy_v2 = segments_vertices_points[filtered_intersected_walls, 1, :]

        direction_x = torch.cos(filtered_rays)
        direction_y = torch.sin(filtered_rays)

        Q0_x, Q0_y, Q1_x, Q1_y = xy_v1[:, 0], xy_v1[:, 1], xy_v2[:, 0], xy_v2[:, 1]

        segment_direction_x = Q1_x - Q0_x
        segment_direction_y = Q1_y - Q0_y

        src_tensor_x, src_tensor_y = src_tensor
        A = torch.stack([direction_x, -segment_direction_x, direction_y, -segment_direction_y], dim=1).reshape(-1, 2, 2)
        b = torch.stack([Q0_x - src_tensor_x, Q0_y - src_tensor_y], dim=1).unsqueeze(2)

        solution = torch.linalg.solve(A, b)
        t = solution[:, 0, 0]

        # Compute the intersection points
        intersection_x = src_tensor_x + t * direction_x
        intersection_y = src_tensor_y + t * direction_y

        # Combine the angle and intersection points into a single tensor

        result = torch.stack([intersection_x, intersection_y], dim=1).to(default_device)

        if view_field < 360:
            # adds a vertex on the source location
            index = torch.nonzero((filtered_rays_indices == 0), as_tuple=True)[0]
            result = torch.cat([result[:index, :], src_tensor.unsqueeze(0), result[index:, :]], dim=0)
        return Polygon(result)


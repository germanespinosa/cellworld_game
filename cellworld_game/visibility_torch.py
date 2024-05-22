import torch
import math
import typing
import shapely as sp
from .util import polygons_to_sides
from .coordinate_converter import CoordinateConverter


def normalize_angle(angle):
    """Normalize angles to be within the range [-pi, pi]"""
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


class Visibility:
    def __init__(self, arena: sp.Polygon, occlusions: typing.List[sp.Polygon]):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")  # Set device to GPU
            print("GPU is available")
        else:
            self.device = torch.device("cpu")  # Set device to CPU
            print("GPU is not available, using CPU instead")

        # occlussions tensors

        _, occlusions_sides_centroids, occlusion_vertices, occlusion_sides_vertices = polygons_to_sides(occlusions)

        self.occlusion_vertices = torch.tensor([[vertex.x, vertex.y]
                                                for vertex in occlusion_vertices],
                                               device=self.device)

        self.occlusions_centroids = torch.tensor([[centroid.x, centroid.y]
                                                  for centroid in occlusions_sides_centroids],
                                                 device=self.device)

        self.occlusions_vertices_indices = torch.tensor(occlusion_sides_vertices,
                                                        device=self.device)

        # arena walls tensors

        _, wall_sides_centroids, wall_vertices, wall_sides_vertices = polygons_to_sides([arena])

        self.wall_vertices = torch.tensor([[vertex.x, vertex.y]
                                           for vertex in wall_vertices],
                                          device=self.device)

        self.walls_centroids = torch.tensor([[centroid.x, centroid.y]
                                             for centroid in wall_sides_centroids],
                                            device=self.device)

        self.walls_vertices_indices = torch.tensor(wall_sides_vertices,
                                                   device=self.device)

    def get_points_angles(self,
                          src_tensor: torch.tensor,
                          locations: torch.tensor):
        if locations.numel() == 0:
            return torch.tensor([], device=self.device)
        diff = (locations - src_tensor).to(self.device)
        vertices_angles = torch.atan2(diff[:, 1], diff[:, 0])
        return vertices_angles

    def get_points_distances(self,
                             src_tensor: torch.tensor,
                             locations: torch.tensor):
        if locations.numel() == 0:
            return torch.tensor([], device=self.device)
        distances = torch.sum((locations - src_tensor) ** 2, dim=1).to(self.device)
        return distances

    @staticmethod
    def get_intersections(angles: torch.tensor,
                          segments: torch.tensor):

        A = angles.unsqueeze(1)  # Shape (num_rays, 1)
        V1 = segments[:, 0].unsqueeze(0)  # Shape (1, num_segments)
        V2 = segments[:, 1].unsqueeze(0)  # Shape (1, num_segments)
        diff = torch.abs(V2 - V1)

        # Case 1: V1 < V2 and the difference < PI
        case1 = (V1 < V2) & (diff < torch.pi)
        intersect_case1 = case1 & (A >= V1) & (A <= V2)

        # Case 2: V2 < V1 and the difference < PI
        case2 = (V2 < V1) & (diff < torch.pi)
        intersect_case2 = case2 & (A >= V2) & (A <= V1)

        # Case 3: V1 < V2 and the difference > PI
        case3 = (V1 < V2) & (diff > torch.pi)
        intersect_case3 = case3 & ((A >= V2) | (A <= V1))

        # Case 4: V2 < V1 and the difference > PI
        case4 = (V2 < V1) & (diff > torch.pi)
        intersect_case4 = case4 & ((A >= V1) | (A <= V2))

        # Combine the results from all cases
        intersect = intersect_case1 | intersect_case2 | intersect_case3 | intersect_case4

        return intersect

    def line_of_sight(self,
                      src: typing.Union[typing.Tuple[float, float]],
                      dst: typing.Union[typing.Tuple[float, float]]) -> bool:

        src_tensor = torch.tensor(src, device=self.device)
        dst_tensor = torch.tensor(dst, device=self.device)

        occlusions_vertices_angles = self.get_points_angles(src_tensor=src_tensor,
                                                            locations=self.occlusion_vertices)

        occlusions_segments_angles = self.__get_segments_vertices_angles__(vertices_angles=occlusions_vertices_angles,
                                                                           segments=self.occlusions_vertices_indices)

        ray = self.get_points_angles(src_tensor=src_tensor, locations=dst_tensor.unsqueeze(0))

        intersections = self.get_intersections(angles=ray,
                                               segments=occlusions_segments_angles)[0, :]

        if not intersections.any():
            return True

        occlusions_distances = self.get_points_distances(src_tensor=src_tensor,
                                                         locations=self.occlusions_centroids[intersections])

        dst_distance = self.get_points_distances(src_tensor=src_tensor,
                                                 locations=dst_tensor.unsqueeze(0))

        if (dst_distance < occlusions_distances).all():
            return True
        else:
            return False

    def __get_segments_vertices_angles__(self,
                                         vertices_angles: torch.tensor,
                                         segments: torch.tensor):
        if segments.numel() == 0:
            return torch.tensor([], device=self.device)
        return torch.stack([normalize_angle((vertices_angles[segments[:, 0]])),
                            normalize_angle((vertices_angles[segments[:, 1]]))], dim=1).to(self.device)

    def __get_segments_vertices_points__(self,
                                         vertices_points: torch.tensor,
                                         segments: torch.tensor):
        if segments.numel() == 0:
            return torch.tensor([], device=self.device)
        return torch.stack([vertices_points[segments[:, 0]],
                            vertices_points[segments[:, 1]]], dim=1).to(self.device)

    def get_visibility_polygon(self,
                               src: typing.Tuple[float, float],
                               direction: float,
                               view_field: float = 360):

        theta = math.radians((direction + 180) % 360) - math.pi

        src_tensor = torch.tensor(src, device=self.device)

        occlusions_vertices_angles = self.get_points_angles(src_tensor=src_tensor,
                                                            locations=self.occlusion_vertices)

        occlusions_segments_angles = self.__get_segments_vertices_angles__(vertices_angles=occlusions_vertices_angles,
                                                                           segments=self.occlusions_vertices_indices)

        occlusions_segments_points = self.__get_segments_vertices_points__(vertices_points=self.occlusion_vertices,
                                                                           segments=self.occlusions_vertices_indices)

        occlusions_distances = self.get_points_distances(src_tensor=src_tensor,
                                                         locations=self.occlusions_centroids)

        walls_vertices_angles = self.get_points_angles(src_tensor=src_tensor,
                                                       locations=self.wall_vertices)

        walls_segments_angles = self.__get_segments_vertices_angles__(vertices_angles=walls_vertices_angles,
                                                                      segments=self.walls_vertices_indices)

        walls_segments_points = self.__get_segments_vertices_points__(vertices_points=self.wall_vertices,
                                                                      segments=self.walls_vertices_indices)

        walls_distances = self.get_points_distances(src_tensor=src_tensor,
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

            rays = torch.cat((torch.tensor([lb, ub], device=self.device), filtered_thetas), dim=0)

        ordered_rays, rays_indices = torch.sort(rays)

        if view_field < 360:
            # saves_the_index of the lower and upper bound ray indices
            lower_bound_index = torch.nonzero((rays_indices == 0), as_tuple=True)[0]
            upper_bound_index = torch.nonzero((rays_indices == 1), as_tuple=True)[0]

        intersections = self.get_intersections(angles=ordered_rays,
                                               segments=segments_vertices_angles)

        segments_distances = segments_distances.expand_as(intersections)

        intersected_walls_distances = torch.where(intersections,
                                                  segments_distances,
                                                  math.inf).to(self.device)

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

        result = torch.stack([intersection_x, intersection_y], dim=1).to(self.device)

        if view_field < 360:
            # adds a vertex on the source location
            index = torch.nonzero((filtered_rays_indices == 0), as_tuple=True)[0]
            result = torch.cat([result[:index, :], src_tensor.unsqueeze(0), result[index:, :]], dim=0)
        return result.cpu()

    def render(self,
               surface,
               coordinate_converter: CoordinateConverter,
               location: typing.Tuple[float, float],
               direction: float,
               view_field: float = 360,
               color: typing.Tuple[int, int, int] = (180, 180, 180)):

        import pygame

        visibility_polygon_vertices = self.get_visibility_polygon(src=location,
                                                                  direction=direction,
                                                                  view_field=view_field)

        pygame.draw.polygon(surface,
                            color,
                            [coordinate_converter.from_canonical((float(point_x), float(point_y)))
                             for point_x, point_y
                             in visibility_polygon_vertices])

        pygame.draw.circle(surface=surface,
                           color=(0, 0, 255),
                           center=coordinate_converter.from_canonical(location),
                           radius=5,
                           width=2)

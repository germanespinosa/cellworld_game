import typing
import cellworld as cw
from .util import create_hexagon
from .navigation import Navigation


class CellWorldLoader:
    def __init__(self,
                 world_name: str):
        self.world = cw.World.get_from_parameters_names(world_configuration_name="hexagonal",
                                                        world_implementation_name="canonical",
                                                        occlusions_name=world_name)
        paths_builder = cw.Paths_builder.get_from_name(world_configuration_name="hexagonal",
                                                       occlusions_name="21_05")
        paths = cw.Paths(builder=paths_builder, world=self.world)
        cellmap = cw.Cell_map(self.world.configuration.cell_coordinates)

        self.locations: typing.List[typing.Optional[typing.Tuple[float, float]]] = [None
                                                                                    if c.occluded
                                                                                    else tuple(c.location.get_values())
                                                                                    for c in self.world.cells]

        locations_paths: typing.List[typing.List[typing.Optional[int]]] = [[None for _ in range(len(self.world.cells))]
                                                                            for _ in range(len(self.world.cells))]
        for src_cell in self.world.cells:
            for dst_cell in self.world.cells:
                move = paths.get_move(src_cell=src_cell, dst_cell=dst_cell)
                next_step_id = cellmap[src_cell.coordinates + move]
                locations_paths[src_cell.id][dst_cell.id] = next_step_id
        self.paths = locations_paths
        self.open_locations = [tuple(c.location.get_values()) for c in self.world.cells.free_cells()]
        arena_center = self.world.implementation.space.center.get_values()
        arena_transformation: cw.Transformation = self.world.implementation.space.transformation
        cell_transformation: cw.Transformation = self.world.implementation.cell_transformation
        self.arena = create_hexagon(arena_center, arena_transformation.size, arena_transformation.rotation)
        self.occlusions = [create_hexagon(cell.location.get_values(),
                                          cell_transformation.size,
                                          arena_transformation.rotation + cell_transformation.rotation)
                           for cell
                           in self.world.cells.occluded_cells()]
        spawn_cells = cw.Cell_group_builder.get_from_name("hexagonal",
                                                          world_name,
                                                          "spawn_locations")
        self.robot_start_locations = [tuple(self.world.cells[sc].location.get_values()) for sc in spawn_cells]
        self.full_action_list = self.open_locations

        lppo_cells = cw.Cell_group_builder.get_from_name("hexagonal",
                                                         world_name,
                                                         "lppo")

        self.tlppo_action_list = [tuple(self.world.cells[sc].location.get_values()) for sc in lppo_cells]

        cell_visibility = cw.get_resource("graph", "hexagonal", world_name, "cell_visibility")

        self.navigation = Navigation(locations=self.locations,
                                     paths=self.paths,
                                     visibility=cell_visibility)

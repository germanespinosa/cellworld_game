import typing
import cellworld as cw
from .navigation import Navigation
from .polygon import Polygon


class CellWorldLoader:
    def __init__(self,
                 world_name: str):
        self.world_name = world_name
        self.world = cw.World.get_from_parameters_names(world_configuration_name="hexagonal",
                                                        world_implementation_name="canonical",
                                                        occlusions_name=world_name)
        paths_builder = cw.Paths_builder.get_from_name(world_configuration_name="hexagonal",
                                                       occlusions_name=world_name)
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
        self.arena = Polygon.regular(center=arena_center,
                                     diameter=arena_transformation.size,
                                     angle=arena_transformation.rotation,
                                     sides=6)
        self.occlusions = [Polygon.regular(center=cell.location.get_values(),
                                           diameter=cell_transformation.size,
                                           angle=arena_transformation.rotation + cell_transformation.rotation,
                                           sides=self.world.configuration.cell_shape.sides)
                           for cell
                           in self.world.cells.occluded_cells()]
        spawn_cells = cw.Cell_group_builder.get_from_name("hexagonal",
                                                          world_name,
                                                          "spawn_locations")
        self.robot_start_locations = [tuple(self.world.cells[sc].location.get_values()) for sc in spawn_cells]
        if not self.robot_start_locations:
            self.robot_start_locations = [(.5, .5)]
        self.full_action_list = self.open_locations

        try:
            self.lppo = cw.Cell_group_builder.get_from_name("hexagonal",
                                                            world_name,
                                                            "lppo")
            self.tlppo_action_list = [tuple(self.world.cells[sc].location.get_values()) for sc in self.lppo]
        except:
            self.lppo = []
            self.tlppo_action_list = []

        cell_visibility = cw.get_resource("graph", "hexagonal", world_name, "cell_visibility")

        self.navigation = Navigation(locations=self.locations,
                                     paths=self.paths,
                                     visibility=cell_visibility)
        self._options_graph = None
    @property
    def options_graph(self) -> cw.Graph:
        if self._options_graph is None:
            graph_builder = cw.get_resource("graph",
                                            "hexagonal",
                                            self.world_name,
                                            "options")
            # options_graph = cw.Graph(builder=graph_builder,
            #                          cells=self.world.cells)
            cell_index = {cell.id: index for index, cell in enumerate(self.world.cells.free_cells())}
            self._options_graph = [[cell_index[cell_id] for cell_id in cnn] for cnn in graph_builder if cnn]
        return self._options_graph

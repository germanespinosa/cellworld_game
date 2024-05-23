from .lib import is_torch_available

if is_torch_available() is not None:
    from .torch import Polygon
else:
    from .shapely import Polygon

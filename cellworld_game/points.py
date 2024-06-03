from .lib import is_torch_available

if is_torch_available() is not None:
    from .torch import Points
else:
    from .shapely import Points

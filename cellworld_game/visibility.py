from .lib import is_torch_available

if is_torch_available():
    from .torch import Visibility
else:
    from .shapely import Visibility

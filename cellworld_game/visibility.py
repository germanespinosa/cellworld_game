import importlib.util

module_spec = importlib.util.find_spec("torch")

if module_spec is not None:
    from .visibility_torch import Visibility
else:
    from .visibility_shapely import Visibility

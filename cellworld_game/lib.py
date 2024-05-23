def is_torch_available() -> bool:
    import importlib.util
    module_spec = importlib.util.find_spec("torch")
    if module_spec is not None:
        return True
    else:
        return False

CONDITION_FN_REGISTRY = {}

def register_condition_fn(name):
    def decorator(fn):
        CONDITION_FN_REGISTRY[name] = fn
        return fn
    return decorator

def get_condition_fn(name):
    if name not in CONDITION_FN_REGISTRY:
        raise KeyError(f"Unknown conditioning function {name}")
    return CONDITION_FN_REGISTRY[name]
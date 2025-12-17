import numpy as np

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

# ------------------------------------------------------------------ #
# Registry

@register_condition_fn("relative_viewpoint_rotation")
def relative_viewpoint_rotation(val0, val1, meta, r=0.5):
    """
    Zero-1-to-3 encoding for the relative angle between two orthogonal cameras,
    after applying a global rotation (0, 90, 180, 270 degrees).

    Output (in radians):
        [ Δθ, sin(φ), cos(φ), r ]
    """

    rot_deg = meta.get("rotation_deg", 0) % 360
    rot_rad = np.deg2rad(rot_deg)

    # Initial relative direction: left → vector (-1, 0)
    v = np.array([-1.0, 0.0], dtype=np.float32)

    # 2D rotation matrix
    R = np.array([
        [np.cos(rot_rad), -np.sin(rot_rad)],
        [np.sin(rot_rad),  np.cos(rot_rad)]
    ], dtype=np.float32)

    # rotated direction
    v_rot = R @ v   # (x', y')
    x, y = v_rot

    # Classify direction: horizontal vs vertical
    if abs(x) > abs(y):
        # left / right bordering
        phi = np.sign(x) * (np.pi / 2)   # -π/2 (left) or +π/2 (right)
        theta = 0.0                      # Δθ = 0 for horizontal
    else:
        # top / bottom bordering
        phi = 0.0                        # φ = 0 for vertical
        theta = np.sign(y) * (np.pi / 2) # -π/2 (top) or +π/2 (bottom)

    # image0 → always default
    out0 = np.array([0.0, np.sin(0.0), np.cos(0.0),  r], dtype=np.float32)

    # image1 → rotated value
    out1 = np.array([theta, np.sin(phi), np.cos(phi), r], dtype=np.float32)

    return out0, out1
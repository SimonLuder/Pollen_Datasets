import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# # ---------------------------------------------------------
# # Stable, non-flipping, non-rotating basis
# # ---------------------------------------------------------
def orientation_to_basis(orientation):
    """
    orientation = (dtheta, sin_dphi, cos_dphi, dr)

    Returns:
        C  : camera center (on sphere, scaled by dr)
        T  : tangent    (width direction, stable w.r.t. theta)
        B  : bitangent  (height direction)
        N  : normal vector (view direction)
    """
    dtheta, sin_dphi, cos_dphi, dr = orientation
    dphi = np.arctan2(sin_dphi, cos_dphi)

    # Normal direction from spherical coords (unit)
    N = np.array([
        np.cos(dtheta) * np.cos(dphi),
        np.cos(dtheta) * np.sin(dphi),
        np.sin(dtheta)
    ])

    # N /= np.linalg.norm(N)

    # Center on the sphere
    C = dr * N

    # --- Width direction: horizontal, independent of theta ---
    # This is a unit vector tangent to the sphere at azimuth dphi,
    # pointing in the +phi direction (around Z axis).
    T = np.array([
        -np.sin(dphi),
        np.cos(dphi),
        0.0
    ])
    T /= np.linalg.norm(T)

    # --- Height direction: perpendicular to both N and T ---
    B = np.cross(N, T)
    B /= np.linalg.norm(B)

    return C, T, B, N




# ---------------------------------------------------------
# PLOT SPHERE WITH POINTS + WIDTH+HEIGHT ARROWS
# ---------------------------------------------------------
def plot_unit_sphere_with_orientation_arrows(
    orientation0, orientation1,
    rotation0=0.0, rotation1=0.0,
    color0="blue", color1="red",
    height_color="orange",
    width_color="green",
    arrow_scale=0.25,
    sphere_alpha=0.12,
    invert_z=False,
    ):
    """
    Plot:
      • unit sphere
      • point for each camera center
      • width (T) and height (B) direction arrows at each point

    rotation0, rotation1 (degrees):
        Rotate the width/height directions CCW around C-axis.
    """

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    # ---------------------------------------------------------
    # UNIT SPHERE
    # ---------------------------------------------------------
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 120)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="gray", alpha=sphere_alpha, linewidth=0)

    # ---------------------------------------------------------
    # ORIENTATION 0
    # ---------------------------------------------------------
    C0, T0, B0, _ = orientation_to_basis(orientation0)

    # rotate T0 and B0 around C0 by rotation0 degrees
    T0r = rotate_vector(T0, C0, rotation0)
    B0r = rotate_vector(B0, C0, rotation0)

    # arrows
    ax.quiver(*C0, *T0r, length=arrow_scale, color=width_color, linewidth=2, label="image width")
    ax.quiver(*C0, *B0r, length=arrow_scale, color=height_color, linewidth=2, linestyle="dashed", label="image height")

    ax.scatter(*C0, s=80, color=color0, label="Image 0")
    ax.plot([0, C0[0]], [0, C0[1]], [0, C0[2]], color=color0, linewidth=2)

    # ---------------------------------------------------------
    # ORIENTATION 1
    # ---------------------------------------------------------
    C1, T1, B1, _ = orientation_to_basis(orientation1)

    T1r = rotate_vector(T1, C1, rotation1)
    B1r = rotate_vector(B1, C1, rotation1)

    ax.quiver(*C1, *T1r, length=arrow_scale, color=width_color, linewidth=2)
    ax.quiver(*C1, *B1r, length=arrow_scale, color=height_color, linewidth=2, linestyle="dashed")

    ax.scatter(*C1, s=80, color=color1, label="Image 1")
    ax.plot([0, C1[0]], [0, C1[1]], [0, C1[2]], color=color1, linewidth=2)

    # ---------------------------------------------------------
    # AXES
    # ---------------------------------------------------------
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])

    if invert_z:
        ax.invert_zaxis()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.legend()
    plt.title("Image Orientations on Unit Sphere")
    plt.show()


def rotate_vector(v, axis, degrees):
    """Rotate vector v around 'axis' by 'degrees' (Rodrigues)."""
    theta = np.deg2rad(degrees)
    axis = axis / np.linalg.norm(axis)
    v = np.asarray(v)

    return (
        v * np.cos(theta)
        + np.cross(axis, v) * np.sin(theta)
        + axis * np.dot(axis, v) * (1 - np.cos(theta))
    )


# ---------------------------------------------------------
# Create oriented image plane using the stable basis
# ---------------------------------------------------------
def make_oriented_plane(img, orientation, stride, rotation_deg=0.0):
    # IMPORTANT: use the same rotated basis as the arrows
    C, T, B, N = get_rotated_basis(orientation, rotation_deg)

    T = rotate_vector(T, C, rotation_deg)
    B = rotate_vector(B, C, rotation_deg)

    H, W = img.shape[:2]
    aspect = W / H

    # canonical plane coords
    x_lin = np.linspace(-aspect/2, aspect/2, W)  # width
    z_lin = np.linspace(-0.5,       0.5,       H)  # height
    x, z = np.meshgrid(x_lin, z_lin)

    # build plane by offsetting from C along rotated basis
    xr = C[0] + T[0]*x + B[0]*z
    yr = C[1] + T[1]*x + B[1]*z
    zr = C[2] + T[2]*x + B[2]*z

    # 4 corners
    corners = np.array([
        [xr[0,0],    yr[0,0],    zr[0,0]],
        [xr[-1,0],   yr[-1,0],   zr[-1,0]],
        [xr[-1,-1],  yr[-1,-1],  zr[-1,-1]],
        [xr[0,-1],   yr[0,-1],   zr[0,-1]],
        [xr[0,0],    yr[0,0],    zr[0,0]],
    ])

    return xr, yr, zr, corners


def get_rotated_basis(orientation, rotation_deg):
    C, T, B, N = orientation_to_basis(orientation)

    # Apply the same roll rotation used for the arrows
    T_rot = rotate_vector(T, C, rotation_deg)
    B_rot = rotate_vector(B, C, rotation_deg)

    return C, T_rot, B_rot, N


# ---------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------
def plot_dual_camera_planes(img0, img1,
                            orientation0,
                            orientation1,
                            rotation0=0.0, 
                            rotation1=0.0,
                            ax_lim=1.0,
                            border_color0="Blue", border_color1="Red",
                            stride=3,
                            invert_z=False):
    """
    Plot 2 image planes, each with their own stable spherical orientation.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    img0 = np.flip(img0, axis=0)
    img1 = np.flip(img1, axis=0)

    # ----------------------------------------------------------
    # Image 0
    # ----------------------------------------------------------
    x0, y0, z0, corners0 = make_oriented_plane(
        img0, orientation0, stride,
        rotation_deg=rotation0
    )
    ax.plot_surface(x0, y0, z0, facecolors=img0,
                    rstride=stride, cstride=stride, shade=False)

    ax.plot(corners0[:,0], corners0[:,1], corners0[:,2],
            color=border_color0, linewidth=2)
    



    # ----------------------------------------------------------
    # Image 1
    # ----------------------------------------------------------
    x1, y1, z1, corners1 = make_oriented_plane(
        img1, orientation1, stride,
        rotation_deg=rotation1
    )
    ax.plot_surface(x1, y1, z1, facecolors=img1,
                    rstride=stride, cstride=stride, shade=False)

    ax.plot(corners1[:,0], corners1[:,1], corners1[:,2],
            color=border_color1, linewidth=2)

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-ax_lim, ax_lim])
    ax.set_ylim([-ax_lim, ax_lim])
    ax.set_zlim([-ax_lim, ax_lim])

    if invert_z:
        ax.invert_zaxis()

    ax.legend(handles=[
        Line2D([0],[0], lw=4, color=border_color0, label="Image 0"),
        Line2D([0],[0], lw=4, color=border_color1, label="Image 1")
    ])

    plt.show()

def load_train_and_test_ids(path):
    """
    Load trainset (upper main table) and the four Swisens testsets 
    from the raw Excel file.

    Assumes column names are stored in row 0 for all blocks.

    Returns:
        trainset (DataFrame)
        df_test_combined (DataFrame)
    """

    # Load full Excel with header in row 0
    df_raw = pd.read_excel(path, header=0)

    categories = [
        "Unseen Species - Seen Poleno",
        "Unseen Species - Unseen Poleno",
        "Seen Species - Seen Poleno",
        "Seen Species - Unseen Poleno"
    ]

    # Find the first occurrence of any category row
    category_indices = {
        cat: df_raw.index[df_raw.eq(cat).any(axis=1)][0]
        for cat in categories
    }

    first_cat_idx = min(category_indices.values())

    # --------------------------
    # TRAINSET = everything above the first category row
    # --------------------------
    trainset = df_raw.iloc[:first_cat_idx].dropna(how="all")
    trainset = trainset.reset_index(drop=True)
    trainset = trainset.drop(columns="Unnamed: 0", errors="ignore")
    trainset = trainset.dropna()

    # --------------------------
    # TESTSETS
    # --------------------------
    testsets = {}

    sorted_cats = sorted(category_indices.items(), key=lambda x: x[1])

    for i, (cat, idx) in enumerate(sorted_cats):

        start = idx + 1

        # determine end index
        if i < len(sorted_cats) - 1:
            next_idx = sorted_cats[i + 1][1]
        else:
            next_idx = len(df_raw)

        block = df_raw.iloc[start:next_idx].dropna(how="all").reset_index(drop=True)

        testsets[cat] = block

    # --------------------------
    # COMBINE TESTSETS
    # --------------------------
    combined = []
    for cat, df in testsets.items():

        df2 = df.copy()
        df2["seen_species"] = int("Seen Species" in cat)
        df2["seen_poleno"] = int("Seen Poleno" in cat)

        combined.append(df2)

    df_test_combined = pd.concat(combined, ignore_index=True)
    df_test_combined = df_test_combined.drop(columns="Unnamed: 0", errors="ignore")

    return trainset, df_test_combined
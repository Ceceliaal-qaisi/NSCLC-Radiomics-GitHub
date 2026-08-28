
# ================================================================
# STEP 6 - 3-D SHAPE FEATURES
# ================================================================
# Features:
#   1. Volume
#   2. Surface Area
#   3. Sphericity
#   4. Surface-to-Volume Ratio
#
# Input:
#   GTV1_binary_mask.npy
#
# Output:
#   3D_SHAPE_FEATURES/
#
# Notes:
#   - Uses the complete 3-D GTV-1 mask.
#   - Uses the actual CT voxel spacing.
#   - Volume is reported in mm^3.
#   - Surface area is reported in mm^2.
#   - Sphericity is dimensionless.
#   - Surface-to-volume ratio is reported in 1/mm.
# ================================================================

import os
import numpy as np
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATIENT_DIR = os.path.join(
    BASE_DIR,
    "LUNG1-001",
    "69331"
)

MASK_PATH = os.path.join(
    PATIENT_DIR,
    "GTV1_MASK",
    "GTV1_binary_mask.npy"
)

OUTPUT_DIR = os.path.join(
    PATIENT_DIR,
    "3D_SHAPE_FEATURES"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# CT VOXEL SPACING
# ================================================================
# From the CT DICOM series:
#
# Row spacing    = 0.9765625 mm
# Column spacing = 0.9765625 mm
# Slice spacing  = 3.0 mm
#
# Mask array order:
#   [slice, row, column]
#
# Therefore:
#   spacing = [z, y, x]
# ================================================================

VOXEL_SPACING = np.array(
    [3.0, 0.9765625, 0.9765625],
    dtype=float
)

SZ = VOXEL_SPACING[0]
SY = VOXEL_SPACING[1]
SX = VOXEL_SPACING[2]


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("STEP 6 - 3-D SHAPE FEATURES")
print("         VOLUME + SURFACE AREA + SPHERICITY")
print("         SURFACE-TO-VOLUME RATIO")
print("=" * 70)

print()
print("Mask:")
print(MASK_PATH)

print()
print("Output:")
print(OUTPUT_DIR)

print()
print("-" * 70)
print("Voxel spacing")
print("-" * 70)

print(f"X spacing: {SX:.7f} mm")
print(f"Y spacing: {SY:.7f} mm")
print(f"Z spacing: {SZ:.7f} mm")


# ================================================================
# LOAD MASK
# ================================================================

print()
print("-" * 70)
print("Loading binary tumor mask...")
print("-" * 70)

if not os.path.exists(MASK_PATH):
    raise FileNotFoundError(
        f"Binary mask not found:\n{MASK_PATH}"
    )

mask = np.load(MASK_PATH)

mask = mask.astype(bool)

print("Mask loaded successfully.")
print(f"Shape: {mask.shape}")

tumor_voxels = int(np.sum(mask))

print(f"Tumor voxels: {tumor_voxels}")


if tumor_voxels == 0:
    raise ValueError(
        "The GTV-1 mask contains zero tumor voxels."
    )


# ================================================================
# TUMOR SLICES
# ================================================================

tumor_slices = np.where(
    np.any(mask, axis=(1, 2))
)[0]

print(f"Tumor slices: {len(tumor_slices)}")

if len(tumor_slices) > 0:
    print(
        f"Tumor slice range: "
        f"{tumor_slices[0]} - {tumor_slices[-1]}"
    )


# ================================================================
# 1. VOLUME
# ================================================================

print()
print("=" * 70)
print("1. VOLUME")
print("=" * 70)

voxel_volume = SX * SY * SZ

volume_mm3 = tumor_voxels * voxel_volume

volume_cm3 = volume_mm3 / 1000.0

print(f"Voxel volume: {voxel_volume:.6f} mm^3")
print(f"Tumor voxels: {tumor_voxels}")
print(f"Volume: {volume_mm3:.3f} mm^3")
print(f"Volume: {volume_cm3:.3f} cm^3")


# ================================================================
# 2. SURFACE AREA
# ================================================================
#
# Marching Cubes is used to obtain a physical 3-D surface.
#
# The spacing is supplied directly so that the resulting
# vertices are represented in physical millimetres.
# ================================================================

print()
print("=" * 70)
print("2. SURFACE AREA")
print("=" * 70)

print("Constructing 3-D tumor surface...")

verts, faces, normals, values = measure.marching_cubes(
    mask.astype(np.float32),
    level=0.5,
    spacing=(SZ, SY, SX)
)

print(f"Surface vertices: {len(verts)}")
print(f"Surface triangles: {len(faces)}")


# ================================================================
# TRIANGLE SURFACE AREA
# ================================================================

triangle_points_1 = verts[faces[:, 0]]
triangle_points_2 = verts[faces[:, 1]]
triangle_points_3 = verts[faces[:, 2]]

vector_1 = triangle_points_2 - triangle_points_1
vector_2 = triangle_points_3 - triangle_points_1

cross_products = np.cross(
    vector_1,
    vector_2
)

triangle_areas = (
    0.5 *
    np.linalg.norm(
        cross_products,
        axis=1
    )
)

surface_area_mm2 = float(
    np.sum(triangle_areas)
)

print(f"Surface area: {surface_area_mm2:.3f} mm^2")


# ================================================================
# 3. SPHERICITY
# ================================================================
#
# Sphericity:
#
#       (36*pi*V^2)^(1/3)
# Psi = -------------------
#               A
#
# where:
#   V = volume
#   A = surface area
#
# A perfect sphere has sphericity = 1.
# ================================================================

print()
print("=" * 70)
print("3. SPHERICITY")
print("=" * 70)

if surface_area_mm2 > 0:

    sphericity = (
        (36.0 * np.pi * (volume_mm3 ** 2.0))
        ** (1.0 / 3.0)
    ) / surface_area_mm2

else:

    sphericity = np.nan

print(f"Sphericity: {sphericity:.6f}")


# ================================================================
# 4. SURFACE-TO-VOLUME RATIO
# ================================================================

print()
print("=" * 70)
print("4. SURFACE-TO-VOLUME RATIO")
print("=" * 70)

if volume_mm3 > 0:

    surface_to_volume_ratio = (
        surface_area_mm2 / volume_mm3
    )

else:

    surface_to_volume_ratio = np.nan

print(
    f"Surface-to-volume ratio: "
    f"{surface_to_volume_ratio:.6f} 1/mm"
)


# ================================================================
# BASIC VALIDATION
# ================================================================

print()
print("=" * 70)
print("3-D SHAPE VALIDATION")
print("=" * 70)

print(
    f"Volume > 0: "
    f"{volume_mm3 > 0}"
)

print(
    f"Surface area > 0: "
    f"{surface_area_mm2 > 0}"
)

print(
    f"Sphericity > 0: "
    f"{sphericity > 0}"
)

print(
    f"Sphericity <= 1: "
    f"{sphericity <= 1.0}"
)

print(
    f"Surface-to-volume ratio > 0: "
    f"{surface_to_volume_ratio > 0}"
)


# ================================================================
# SAVE NUMERICAL RESULTS
# ================================================================

results = {

    "patient": "LUNG1-001",

    "study_id": "69331",

    "mask_shape": tuple(mask.shape),

    "tumor_voxels": tumor_voxels,

    "tumor_slices": tumor_slices,

    "voxel_spacing_mm": VOXEL_SPACING,

    "voxel_volume_mm3": voxel_volume,

    "volume_mm3": volume_mm3,

    "volume_cm3": volume_cm3,

    "surface_area_mm2": surface_area_mm2,

    "sphericity": sphericity,

    "surface_to_volume_ratio_per_mm":
        surface_to_volume_ratio,

    "surface_vertices": len(verts),

    "surface_triangles": len(faces)
}


RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "3d_shape_features_results.npy"
)

np.save(
    RESULTS_PATH,
    results,
    allow_pickle=True
)


# ================================================================
# SAVE SUMMARY
# ================================================================

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "3d_shape_features_summary.txt"
)

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 70 + "\n")
    f.write("STEP 6 - 3-D SHAPE FEATURES\n")
    f.write("=" * 70 + "\n\n")

    f.write("Patient: LUNG1-001\n")
    f.write("Study ID: 69331\n\n")

    f.write("INPUT MASK\n")
    f.write("-" * 70 + "\n")
    f.write(f"{MASK_PATH}\n\n")

    f.write("MASK INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Mask shape: {mask.shape}\n")
    f.write(f"Tumor voxels: {tumor_voxels}\n")
    f.write(f"Tumor slices: {len(tumor_slices)}\n")

    if len(tumor_slices) > 0:
        f.write(
            f"Tumor slice range: "
            f"{tumor_slices[0]} - "
            f"{tumor_slices[-1]}\n"
        )

    f.write("\n")

    f.write("VOXEL SPACING\n")
    f.write("-" * 70 + "\n")
    f.write(f"X: {SX:.7f} mm\n")
    f.write(f"Y: {SY:.7f} mm\n")
    f.write(f"Z: {SZ:.7f} mm\n")
    f.write(
        f"Voxel volume: "
        f"{voxel_volume:.6f} mm^3\n"
    )

    f.write("\n")

    f.write("3-D SHAPE FEATURES\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Volume: "
        f"{volume_mm3:.6f} mm^3\n"
    )

    f.write(
        f"Volume: "
        f"{volume_cm3:.6f} cm^3\n"
    )

    f.write(
        f"Surface area: "
        f"{surface_area_mm2:.6f} mm^2\n"
    )

    f.write(
        f"Sphericity: "
        f"{sphericity:.8f}\n"
    )

    f.write(
        f"Surface-to-volume ratio: "
        f"{surface_to_volume_ratio:.8f} 1/mm\n"
    )

    f.write("\n")

    f.write("SURFACE INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Surface vertices: "
        f"{len(verts)}\n"
    )

    f.write(
        f"Surface triangles: "
        f"{len(faces)}\n"
    )


# ================================================================
# SAVE 3-D VISUALIZATION
# ================================================================

print()
print("=" * 70)
print("SAVING 3-D VISUALIZATION")
print("=" * 70)

fig = plt.figure(
    figsize=(9, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

# Plot a subset of the surface triangles for visualization.
# The complete surface is still used for the numerical
# surface-area calculation.

ax.plot_trisurf(
    verts[:, 0],
    verts[:, 1],
    verts[:, 2],
    triangles=faces,
    linewidth=0.05,
    antialiased=True,
    alpha=0.8
)

ax.set_title(
    "GTV-1 3-D Surface"
)

ax.set_xlabel(
    "Z position (mm)"
)

ax.set_ylabel(
    "Y position (mm)"
)

ax.set_zlabel(
    "X position (mm)"
)

plt.tight_layout()

VIS_PATH = os.path.join(
    OUTPUT_DIR,
    "gtv1_3d_surface_visualization.png"
)

plt.savefig(
    VIS_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print("3-D Shape Features: DONE")
print(f"Volume: {volume_mm3:.3f} mm^3")
print(f"Surface Area: {surface_area_mm2:.3f} mm^2")
print(f"Sphericity: {sphericity:.6f}")
print(
    f"Surface-to-Volume Ratio: "
    f"{surface_to_volume_ratio:.6f} 1/mm"
)

print()
print("Results saved:")
print(RESULTS_PATH)

print()
print("Summary saved:")
print(SUMMARY_PATH)

print()
print("Visualization saved:")
print(VIS_PATH)

print()
print("Output folder:")
print(OUTPUT_DIR)

print()
print("=" * 70)
print("NEXT STEP:")
print("TEXTURE / FEATURE INTEGRATION")
print("=" * 70)

input("\nPress ENTER to close...")


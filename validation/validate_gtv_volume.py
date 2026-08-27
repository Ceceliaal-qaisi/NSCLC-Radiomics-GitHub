import SimpleITK as sitk
import numpy as np


# ============================================================
# FILE PATHS
# ============================================================

MASK_PATH = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\GTV1_MASK.nrrd"


# ============================================================
# READ MASK
# ============================================================

print("=" * 70)
print("GTV-1 VOLUME VALIDATION")
print("=" * 70)

mask_img = sitk.ReadImage(MASK_PATH)
mask = sitk.GetArrayFromImage(mask_img)

mask = mask > 0

spacing = mask_img.GetSpacing()
origin = mask_img.GetOrigin()
size = mask_img.GetSize()

sx, sy, sz = spacing

print("\nMASK INFORMATION")
print("-" * 70)

print("Size (X,Y,Z):", size)
print("Array shape (Z,Y,X):", mask.shape)

print("Spacing:")
print("  X =", sx, "mm")
print("  Y =", sy, "mm")
print("  Z =", sz, "mm")

print("Origin:", origin)


# ============================================================
# VOXEL COUNT
# ============================================================

voxel_count = int(np.count_nonzero(mask))

print("\nVOXEL COUNT")
print("-" * 70)
print("Tumor voxels:", voxel_count)


# ============================================================
# VOLUME
# ============================================================

voxel_volume = sx * sy * sz

volume_mm3 = voxel_count * voxel_volume
volume_cm3 = volume_mm3 / 1000.0

print("\nVOLUME")
print("-" * 70)

print("Single voxel volume:", voxel_volume, "mm^3")
print("Tumor volume:", volume_mm3, "mm^3")
print("Tumor volume:", volume_cm3, "cm^3")


# ============================================================
# TUMOR SLICES
# ============================================================

tumor_slice_indices = np.where(
    np.any(mask, axis=(1, 2))
)[0]

print("\nTUMOR SLICE RANGE")
print("-" * 70)

print("Number of tumor slices:", len(tumor_slice_indices))

if len(tumor_slice_indices) > 0:

    first_slice = int(tumor_slice_indices[0])
    last_slice = int(tumor_slice_indices[-1])

    print("First tumor slice:", first_slice)
    print("Last tumor slice:", last_slice)

    physical_z_extent = (last_slice - first_slice + 1) * sz

    print(
        "Approximate Z extent:",
        physical_z_extent,
        "mm"
    )


# ============================================================
# BOUNDING BOX
# ============================================================

coords = np.argwhere(mask)

z_min, y_min, x_min = coords.min(axis=0)
z_max, y_max, x_max = coords.max(axis=0)

print("\nBOUNDING BOX")
print("-" * 70)

print("X range:", x_min, "to", x_max)
print("Y range:", y_min, "to", y_max)
print("Z range:", z_min, "to", z_max)

width_x = (x_max - x_min + 1) * sx
height_y = (y_max - y_min + 1) * sy
depth_z = (z_max - z_min + 1) * sz

print("\nApproximate tumor dimensions:")
print("Width  X:", width_x, "mm")
print("Height Y:", height_y, "mm")
print("Depth  Z:", depth_z, "mm")


# ============================================================
# PHYSICAL BOUNDING BOX
# ============================================================

physical_x_min = origin[0] + x_min * sx
physical_x_max = origin[0] + (x_max + 1) * sx

physical_y_min = origin[1] + y_min * sy
physical_y_max = origin[1] + (y_max + 1) * sy

physical_z_min = origin[2] + z_min * sz
physical_z_max = origin[2] + (z_max + 1) * sz

print("\nPHYSICAL COORDINATES")
print("-" * 70)

print("X:", physical_x_min, "to", physical_x_max, "mm")
print("Y:", physical_y_min, "to", physical_y_max, "mm")
print("Z:", physical_z_min, "to", physical_z_max, "mm")


# ============================================================
# SANITY CHECK
# ============================================================

print("\nSANITY CHECK")
print("-" * 70)

if voxel_count == 54639:
    print("Voxel count matches previous result: PASS")
else:
    print("Voxel count changed: CHECK")

if volume_cm3 > 0:
    print("Mask is non-empty: PASS")
else:
    print("Mask is empty: FAIL")

if set(np.unique(mask.astype(np.uint8))).issubset({0, 1}):
    print("Binary mask: PASS")
else:
    print("Binary mask: FAIL")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"Volume       : {volume_cm3:.4f} cm^3")
print(f"Voxels       : {voxel_count}")
print(f"Slices       : {len(tumor_slice_indices)}")
print(f"Dimensions   : {width_x:.2f} x {height_y:.2f} x {depth_z:.2f} mm")

print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
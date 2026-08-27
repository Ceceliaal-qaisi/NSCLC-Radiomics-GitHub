import SimpleITK as sitk
import numpy as np


# ============================================================
# FILE
# ============================================================

MASK_PATH = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\GTV1_MASK.nrrd"


# ============================================================
# READ MASK
# ============================================================

img = sitk.ReadImage(MASK_PATH)
mask = sitk.GetArrayFromImage(img) > 0

sx, sy, sz = img.GetSpacing()

coords = np.argwhere(mask)

z = coords[:, 0]
y = coords[:, 1]
x = coords[:, 2]

voxel_count = int(mask.sum())
voxel_volume = sx * sy * sz

volume_mm3 = voxel_count * voxel_volume
volume_cm3 = volume_mm3 / 1000

tumor_slices = np.where(np.any(mask, axis=(1, 2)))[0]

slice_voxels = np.sum(mask, axis=(1, 2))
slice_voxels = slice_voxels[slice_voxels > 0]


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("GTV-1 SUMMARY")
print("=" * 70)

print("\nMASK")
print("-" * 70)
print("Size (X,Y,Z):", img.GetSize())
print("Array shape (Z,Y,X):", mask.shape)
print("Spacing (mm):", img.GetSpacing())
print("Origin:", img.GetOrigin())

print("\nTUMOR")
print("-" * 70)
print("Tumor voxels:", voxel_count)
print("Tumor slices:", len(tumor_slices))

print(
    "Slice range:",
    int(tumor_slices[0]),
    "to",
    int(tumor_slices[-1])
)

print(
    "Voxel volume:",
    f"{voxel_volume:.6f}",
    "mm^3"
)

print(
    "Tumor volume:",
    f"{volume_mm3:.4f}",
    "mm^3"
)

print(
    "Tumor volume:",
    f"{volume_cm3:.4f}",
    "cm^3"
)


print("\nSLICE STATISTICS")
print("-" * 70)

print(
    "Maximum voxels/slice:",
    int(slice_voxels.max())
)

print(
    "Mean voxels/tumor slice:",
    f"{slice_voxels.mean():.2f}"
)

print(
    "Median voxels/tumor slice:",
    f"{np.median(slice_voxels):.2f}"
)


# ============================================================
# BOUNDING BOX
# ============================================================

print("\nBOUNDING BOX")
print("-" * 70)

xmin, ymin, zmin = x.min(), y.min(), z.min()
xmax, ymax, zmax = x.max(), y.max(), z.max()

width = (xmax - xmin + 1) * sx
height = (ymax - ymin + 1) * sy
depth = (zmax - zmin + 1) * sz

print(f"X: {xmin} to {xmax}")
print(f"Y: {ymin} to {ymax}")
print(f"Z: {zmin} to {zmax}")

print(
    "Physical dimensions:",
    f"{width:.2f} × {height:.2f} × {depth:.2f} mm"
)


# ============================================================
# BINARY CHECK
# ============================================================

print("\nQUALITY CHECK")
print("-" * 70)

unique_values = np.unique(mask.astype(np.uint8))

print("Unique mask values:", unique_values)

if voxel_count > 0:
    print("Mask non-empty: PASS")
else:
    print("Mask non-empty: FAIL")

if set(unique_values).issubset({0, 1}):
    print("Binary mask: PASS")
else:
    print("Binary mask: FAIL")


print("\n" + "=" * 70)
print("SUMMARY COMPLETE")
print("=" * 70)
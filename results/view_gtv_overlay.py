import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\82046"

MASK_PATH = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\GTV1_MASK.nrrd"

OUTPUT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\GTV1_QC_IMAGES"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 1. READ CT DICOM
# ============================================================

print("=" * 70)
print("READING CT")
print("=" * 70)

reader = sitk.ImageSeriesReader()

series_ids = reader.GetGDCMSeriesIDs(CT_FOLDER)

if not series_ids:
    raise RuntimeError("No DICOM CT series found.")

print("CT Series found:", len(series_ids))

# Use the first series
series_uid = series_ids[0]

file_names = reader.GetGDCMSeriesFileNames(
    CT_FOLDER,
    series_uid
)

print("CT files:", len(file_names))

reader.SetFileNames(file_names)

ct_img = reader.Execute()

ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)

print("CT shape (Z,Y,X):", ct.shape)
print("CT spacing:", ct_img.GetSpacing())


# ============================================================
# 2. READ GTV-1 MASK
# ============================================================

print("\n" + "=" * 70)
print("READING GTV-1 MASK")
print("=" * 70)

mask_img = sitk.ReadImage(MASK_PATH)

mask = sitk.GetArrayFromImage(mask_img)

mask = mask > 0

print("Mask shape (Z,Y,X):", mask.shape)
print("Mask voxels:", np.count_nonzero(mask))


# ============================================================
# 3. CHECK CT AND MASK SIZE
# ============================================================

print("\n" + "=" * 70)
print("SIZE CHECK")
print("=" * 70)

if ct.shape != mask.shape:
    raise RuntimeError(
        f"CT and mask dimensions do not match!\n"
        f"CT: {ct.shape}\n"
        f"Mask: {mask.shape}"
    )

print("CT and mask dimensions match: PASS")


# ============================================================
# 4. FIND TUMOR CENTER
# ============================================================

coords = np.argwhere(mask)

z_indices = coords[:, 0]
y_indices = coords[:, 1]
x_indices = coords[:, 2]

z_center = int(np.mean(z_indices))
y_center = int(np.mean(y_indices))
x_center = int(np.mean(x_indices))

print("\nTumor center:")
print("Z:", z_center)
print("Y:", y_center)
print("X:", x_center)

print("Tumor slices:",
      int(z_indices.min()),
      "to",
      int(z_indices.max()))


# ============================================================
# 5. AXIAL VIEW
# ============================================================

axial_z = z_center

plt.figure(figsize=(8, 8))

plt.imshow(
    ct[axial_z],
    cmap="gray"
)

overlay = np.ma.masked_where(
    ~mask[axial_z],
    mask[axial_z]
)

plt.imshow(
    overlay,
    cmap="autumn",
    alpha=0.55
)

plt.title(
    f"GTV-1 Overlay - Axial Slice {axial_z}"
)

plt.axis("off")

axial_path = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_axial.png"
)

plt.savefig(
    axial_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. CORONAL VIEW
# ============================================================

coronal_y = y_center

plt.figure(figsize=(10, 8))

plt.imshow(
    ct[:, coronal_y, :],
    cmap="gray",
    aspect="auto"
)

overlay = np.ma.masked_where(
    ~mask[:, coronal_y, :],
    mask[:, coronal_y, :]
)

plt.imshow(
    overlay,
    cmap="autumn",
    alpha=0.55,
    aspect="auto"
)

plt.title(
    f"GTV-1 Overlay - Coronal"
)

plt.axis("off")

coronal_path = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_coronal.png"
)

plt.savefig(
    coronal_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. SAGITTAL VIEW
# ============================================================

sagittal_x = x_center

plt.figure(figsize=(10, 8))

plt.imshow(
    ct[:, :, sagittal_x],
    cmap="gray",
    aspect="auto"
)

overlay = np.ma.masked_where(
    ~mask[:, :, sagittal_x],
    mask[:, :, sagittal_x]
)

plt.imshow(
    overlay,
    cmap="autumn",
    alpha=0.55,
    aspect="auto"
)

plt.title(
    f"GTV-1 Overlay - Sagittal"
)

plt.axis("off")

sagittal_path = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_sagittal.png"
)

plt.savefig(
    sagittal_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. SAVE CENTER SLICE
# ============================================================

with open(
    os.path.join(OUTPUT_FOLDER, "QC_info.txt"),
    "w"
) as f:

    f.write("GTV-1 QUALITY CHECK\n")
    f.write("===================\n")
    f.write(f"CT shape: {ct.shape}\n")
    f.write(f"Mask shape: {mask.shape}\n")
    f.write(f"Mask voxels: {np.count_nonzero(mask)}\n")
    f.write(f"Tumor slices: {z_indices.min()} - {z_indices.max()}\n")
    f.write(f"Tumor center Z: {z_center}\n")
    f.write(f"Tumor center Y: {y_center}\n")
    f.write(f"Tumor center X: {x_center}\n")


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("QUALITY CHECK IMAGES CREATED")
print("=" * 70)

print("Saved folder:")
print(OUTPUT_FOLDER)

print("\nImages:")
print(axial_path)
print(coronal_path)
print(sagittal_path)

print("\nSUCCESS")
print("=" * 70)
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom

# ============================================================
# VISUALIZATION - GTV-1 SEGMENTATION + BOUNDARY
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_FOLDER = os.path.join(PATIENT_FOLDER, "82046")
RTSTRUCT_FOLDER = os.path.join(PATIENT_FOLDER, "78236")

MASK_FOLDER = os.path.join(PATIENT_FOLDER, "GTV1_MASK")
BOUNDARY_FOLDER = os.path.join(PATIENT_FOLDER, "BOUNDARY")

MASK_FILE = os.path.join(
    MASK_FOLDER,
    "GTV1_binary_mask.npy"
)

BOUNDARY_FILE = os.path.join(
    BOUNDARY_FOLDER,
    "ordered_boundary_points.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "VISUALIZATION"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


print("=" * 70)
print("VISUALIZATION OF GTV-1 SEGMENTATION AND BOUNDARY")
print("=" * 70)


# ============================================================
# 1. LOAD CT
# ============================================================

print("\nLoading CT images...")

ct_files = []

for file in os.listdir(CT_FOLDER):
    if file.lower().endswith(".dcm"):
        path = os.path.join(CT_FOLDER, file)

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=False)

            if getattr(ds, "Modality", "") == "CT":
                ct_files.append(ds)

        except Exception:
            pass


# Sort CT slices
ct_files.sort(
    key=lambda x: float(
        getattr(x, "ImagePositionPatient", [0, 0, 0])[2]
    )
)

print("CT slices:", len(ct_files))


# ============================================================
# 2. CREATE CT VOLUME
# ============================================================

ct_volume = np.stack(
    [ds.pixel_array for ds in ct_files],
    axis=0
)

print("CT volume shape:", ct_volume.shape)


# ============================================================
# 3. LOAD BINARY MASK
# ============================================================

print("\nLoading GTV-1 binary mask...")

mask = np.load(MASK_FILE)

print("Mask shape:", mask.shape)
print("Tumor pixels:", np.sum(mask))


# ============================================================
# 4. LOAD BOUNDARIES
# ============================================================

print("\nLoading boundary points...")

boundary_data = np.load(
    BOUNDARY_FILE,
    allow_pickle=True
).item()

print("Boundary slices:", len(boundary_data))


# ============================================================
# 5. FIND TUMOR SLICES
# ============================================================

tumor_slices = np.where(
    np.sum(mask, axis=(1, 2)) > 0
)[0]

print("\nTumor slices:")
print(tumor_slices)


# ============================================================
# 6. CREATE SEGMENTATION IMAGES
# ============================================================

print("\nCreating segmentation images...")

for z in tumor_slices:

    ct = ct_volume[z]
    tumor = mask[z]

    plt.figure(figsize=(7, 7))

    plt.imshow(
        ct,
        cmap="gray"
    )

    plt.imshow(
        np.ma.masked_where(
            tumor == 0,
            tumor
        ),
        cmap="Reds",
        alpha=0.45
    )

    plt.title(
        f"GTV-1 Segmentation - Slice {z}"
    )

    plt.axis("off")

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"segmentation_slice_{z:03d}.png"
    )

    plt.savefig(
        filename,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()


# ============================================================
# 7. CREATE BOUNDARY IMAGES
# ============================================================

print("\nCreating boundary images...")

for z in tumor_slices:

    ct = ct_volume[z]

    plt.figure(figsize=(7, 7))

    plt.imshow(
        ct,
        cmap="gray"
    )

    if z in boundary_data:

        points = np.asarray(
            boundary_data[z]
        )

        if len(points) > 0:

            y = points[:, 0]
            x = points[:, 1]

            # Close boundary
            x_closed = np.append(x, x[0])
            y_closed = np.append(y, y[0])

            plt.plot(
                x_closed,
                y_closed,
                linewidth=1.5
            )

            plt.scatter(
                x,
                y,
                s=3
            )

    plt.title(
        f"GTV-1 Boundary - Slice {z}"
    )

    plt.axis("off")

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"boundary_slice_{z:03d}.png"
    )

    plt.savefig(
        filename,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()


# ============================================================
# 8. CREATE COMBINED IMAGE
# ============================================================

print("\nCreating combined images...")

for z in tumor_slices:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6)
    )

    # --------------------------------------------------------
    # LEFT: SEGMENTATION
    # --------------------------------------------------------

    axes[0].imshow(
        ct_volume[z],
        cmap="gray"
    )

    axes[0].imshow(
        np.ma.masked_where(
            mask[z] == 0,
            mask[z]
        ),
        cmap="Reds",
        alpha=0.45
    )

    axes[0].set_title(
        f"GTV-1 Segmentation\nSlice {z}"
    )

    axes[0].axis("off")


    # --------------------------------------------------------
    # RIGHT: BOUNDARY
    # --------------------------------------------------------

    axes[1].imshow(
        ct_volume[z],
        cmap="gray"
    )

    if z in boundary_data:

        points = np.asarray(
            boundary_data[z]
        )

        if len(points) > 0:

            y = points[:, 0]
            x = points[:, 1]

            x_closed = np.append(x, x[0])
            y_closed = np.append(y, y[0])

            axes[1].plot(
                x_closed,
                y_closed,
                linewidth=2
            )

            axes[1].scatter(
                x,
                y,
                s=3
            )

    axes[1].set_title(
        f"Moore Boundary\nSlice {z}"
    )

    axes[1].axis("off")


    plt.tight_layout()

    filename = os.path.join(
        OUTPUT_FOLDER,
        f"combined_slice_{z:03d}.png"
    )

    plt.savefig(
        filename,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()


# ============================================================
# 9. SAVE MONTAGE OF ALL TUMOR SLICES
# ============================================================

print("\nCreating tumor montage...")

n = len(tumor_slices)

cols = 5
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(15, 3 * rows)
)

axes = np.array(axes).reshape(-1)

for i, z in enumerate(tumor_slices):

    axes[i].imshow(
        ct_volume[z],
        cmap="gray"
    )

    axes[i].imshow(
        np.ma.masked_where(
            mask[z] == 0,
            mask[z]
        ),
        cmap="Reds",
        alpha=0.45
    )

    axes[i].set_title(
        f"Slice {z}"
    )

    axes[i].axis("off")


for i in range(n, len(axes)):
    axes[i].axis("off")


plt.tight_layout()

montage_file = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_segmentation_montage.png"
)

plt.savefig(
    montage_file,
    bbox_inches="tight",
    dpi=200
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print("\nImages saved in:")

print(OUTPUT_FOLDER)

print("\nYou should find:")

print("1. segmentation_slice_XXX.png")
print("2. boundary_slice_XXX.png")
print("3. combined_slice_XXX.png")
print("4. GTV1_segmentation_montage.png")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

input("\nPress ENTER to close...")
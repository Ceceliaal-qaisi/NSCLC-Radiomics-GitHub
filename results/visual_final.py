import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom

# ============================================================
# FINAL VISUAL CHECK
# CT + GTV-1 + ORIGINAL BOUNDARY + SAMPLED BOUNDARY
# ============================================================

PATIENT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_DIR = os.path.join(PATIENT_DIR, "82046")
MASK_DIR = os.path.join(PATIENT_DIR, "GTV1_MASK")
BOUNDARY_DIR = os.path.join(PATIENT_DIR, "BOUNDARY")
CHAIN_DIR = os.path.join(PATIENT_DIR, "CHAIN_CODE")

OUTPUT_DIR = os.path.join(PATIENT_DIR, "FINAL_VISUAL_CHECK")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FINAL VISUAL CHECK")
print("CT + GTV-1 + BOUNDARY + SAMPLING")
print("=" * 70)

# ------------------------------------------------------------
# LOAD CT
# ------------------------------------------------------------

print("\nLoading CT...")

ct_files = []

for filename in os.listdir(CT_DIR):

    if filename.lower().endswith(".dcm"):

        path = os.path.join(CT_DIR, filename)

        try:
            ds = pydicom.dcmread(path)

            if getattr(ds, "Modality", "") == "CT":
                ct_files.append(ds)

        except:
            pass


def z_position(ds):

    try:
        return float(ds.ImagePositionPatient[2])
    except:
        return float(getattr(ds, "InstanceNumber", 0))


ct_files.sort(key=z_position)

print("CT slices:", len(ct_files))

# ------------------------------------------------------------
# LOAD MASK
# ------------------------------------------------------------

mask_path = os.path.join(
    MASK_DIR,
    "GTV1_binary_mask.npy"
)

mask = np.load(mask_path)

print("Mask shape:", mask.shape)

# ------------------------------------------------------------
# LOAD BOUNDARY
# ------------------------------------------------------------

boundary_path = os.path.join(
    BOUNDARY_DIR,
    "ordered_boundary_points.npy"
)

boundary_data = np.load(
    boundary_path,
    allow_pickle=True
)

if boundary_data.shape == ():
    boundary = boundary_data.item()
else:
    boundary = boundary_data

print("Boundary slices:", len(boundary))

# ------------------------------------------------------------
# SELECT THREE SLICES
# ------------------------------------------------------------

slices = sorted(boundary.keys())

selected = [
    slices[0],
    slices[len(slices) // 2],
    slices[-1]
]

print("\nSelected slices:", selected)

# ------------------------------------------------------------
# PROCESS
# ------------------------------------------------------------

for slice_number in selected:

    idx = int(slice_number)

    print("\nProcessing slice:", slice_number)

    ds = ct_files[idx]

    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    image = image * slope + intercept

    # Lung window
    center = -600
    width = 1500

    low = center - width / 2
    high = center + width / 2

    image_display = np.clip(
        image,
        low,
        high
    )

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    tumor_mask = mask[idx]

    # --------------------------------------------------------
    # ORIGINAL BOUNDARY
    # --------------------------------------------------------

    points = np.asarray(
        boundary[slice_number]
    )

    y = points[:, 0]
    x = points[:, 1]

    # --------------------------------------------------------
    # SAMPLING
    # --------------------------------------------------------

    # Same sampling idea used in Step 3
    sampling_step = max(
        1,
        len(points) // 50
    )

    sampled_points = points[::sampling_step]

    sy = sampled_points[:, 0]
    sx = sampled_points[:, 1]

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # ========================================================
    # 1 - CT + GTV-1
    # ========================================================

    axes[0].imshow(
        image_display,
        cmap="gray"
    )

    mask_overlay = np.ma.masked_where(
        tumor_mask == 0,
        tumor_mask
    )

    axes[0].imshow(
        mask_overlay,
        cmap="Reds",
        alpha=0.50
    )

    axes[0].set_title(
        f"CT + GTV-1 | Slice {slice_number}",
        fontsize=13
    )

    axes[0].axis("off")

    # ========================================================
    # 2 - ORIGINAL BOUNDARY
    # ========================================================

    axes[1].imshow(
        image_display,
        cmap="gray"
    )

    axes[1].plot(
        x,
        y,
        "r-",
        linewidth=2
    )

    axes[1].plot(
        x,
        y,
        "r.",
        markersize=2
    )

    axes[1].set_title(
        f"Original Ordered Boundary\n{len(points)} points",
        fontsize=13
    )

    axes[1].axis("off")

    # ========================================================
    # 3 - SAMPLED BOUNDARY
    # ========================================================

    axes[2].imshow(
        image_display,
        cmap="gray"
    )

    axes[2].plot(
        x,
        y,
        "r-",
        linewidth=1,
        alpha=0.35
    )

    axes[2].plot(
        sx,
        sy,
        "bo-",
        markersize=4,
        linewidth=1
    )

    axes[2].set_title(
        f"Boundary Sampling\n"
        f"{len(points)} → {len(sampled_points)} points",
        fontsize=13
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output = os.path.join(
        OUTPUT_DIR,
        f"slice_{slice_number}_visual.png"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output)

# ------------------------------------------------------------
# DONE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VISUAL CHECK COMPLETED")
print("=" * 70)

print("\nOpen this folder:")
print(OUTPUT_DIR)

print("\nThe images contain:")
print("1. CT + GTV-1")
print("2. Original boundary")
print("3. Sampled boundary")

print("\nNo project files were modified.")

input("\nPress ENTER to close...")
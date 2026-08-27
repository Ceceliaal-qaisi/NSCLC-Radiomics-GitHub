import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom

# ============================================================
# CHECK VISUALIZATION
# CT + GTV-1 + BOUNDARY + SAMPLING
# ============================================================

PATIENT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_DIR = os.path.join(PATIENT_DIR, "82046")
RTSTRUCT_DIR = os.path.join(PATIENT_DIR, "78236")
MASK_DIR = os.path.join(PATIENT_DIR, "GTV1_MASK")
BOUNDARY_DIR = os.path.join(PATIENT_DIR, "BOUNDARY")
CHAIN_DIR = os.path.join(PATIENT_DIR, "CHAIN_CODE")

OUTPUT_DIR = os.path.join(PATIENT_DIR, "VISUAL_CHECK")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("VISUAL CHECK - GTV-1 / BOUNDARY / SAMPLING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD CT
# ------------------------------------------------------------

print("\nLoading CT images...")

ct_files = []

for f in os.listdir(CT_DIR):
    if f.lower().endswith(".dcm"):
        path = os.path.join(CT_DIR, f)

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=False)

            if getattr(ds, "Modality", "") == "CT":
                ct_files.append(ds)

        except Exception:
            pass

if len(ct_files) == 0:
    print("ERROR: No CT files found.")
    input("Press ENTER to close...")
    raise SystemExit

# Sort using ImagePositionPatient when available
def get_z(ds):
    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0))

ct_files.sort(key=get_z)

print("CT slices:", len(ct_files))

first_ct = ct_files[0]

rows = int(first_ct.Rows)
cols = int(first_ct.Columns)

print("CT size:", rows, "x", cols)

# ------------------------------------------------------------
# 2. LOAD MASK
# ------------------------------------------------------------

print("\nLoading binary tumor mask...")

mask_path = os.path.join(
    MASK_DIR,
    "GTV1_binary_mask.npy"
)

if not os.path.exists(mask_path):
    print("ERROR: Mask file not found:")
    print(mask_path)
    input("Press ENTER to close...")
    raise SystemExit

mask = np.load(mask_path)

print("Mask shape:", mask.shape)
print("Tumor pixels:", int(np.sum(mask)))

# ------------------------------------------------------------
# 3. LOAD BOUNDARY
# ------------------------------------------------------------

print("\nLoading ordered boundary...")

boundary_path = os.path.join(
    BOUNDARY_DIR,
    "ordered_boundary_points.npy"
)

if not os.path.exists(boundary_path):
    print("ERROR: Boundary file not found:")
    print(boundary_path)
    input("Press ENTER to close...")
    raise SystemExit

boundary_data = np.load(
    boundary_path,
    allow_pickle=True
)

# The previous code saved a dictionary inside a 0-D numpy array
if boundary_data.shape == ():
    boundary = boundary_data.item()
else:
    boundary = boundary_data

print("Boundary loaded.")

if not isinstance(boundary, dict):
    print("ERROR: Boundary format is not a dictionary.")
    print(type(boundary))
    input("Press ENTER to close...")
    raise SystemExit

print("Boundary slices:", len(boundary))

# ------------------------------------------------------------
# 4. FIND TUMOR SLICES
# ------------------------------------------------------------

tumor_slices = sorted(boundary.keys())

print("\nTumor slices:")
print(tumor_slices)

# ------------------------------------------------------------
# 5. VISUALIZE SELECTED SLICES
# ------------------------------------------------------------

print("\nCreating visual checks...")

# We don't need all 21 images.
# We select beginning, middle and end.
selected = [
    tumor_slices[0],
    tumor_slices[len(tumor_slices) // 2],
    tumor_slices[-1]
]

for slice_number in selected:

    print("\nProcessing slice:", slice_number)

    # --------------------------------------------------------
    # IMPORTANT:
    # Dictionary keys are CT slice indices used by our previous
    # boundary extraction.
    # --------------------------------------------------------

    idx = int(slice_number)

    if idx < 0 or idx >= len(ct_files):
        print("Skipping invalid CT index:", idx)
        continue

    ds = ct_files[idx]

    image = ds.pixel_array.astype(np.float32)

    # Apply DICOM rescale if available
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    image = image * slope + intercept

    # Window for lung visualization
    window_center = -600
    window_width = 1500

    low = window_center - window_width / 2
    high = window_center + window_width / 2

    display_image = np.clip(image, low, high)

    # Normalize for matplotlib
    display_image = (
        display_image - low
    ) / (high - low)

    # --------------------------------------------------------
    # GET MASK
    # --------------------------------------------------------

    if idx >= mask.shape[0]:
        print("Mask index unavailable:", idx)
        continue

    slice_mask = mask[idx]

    # --------------------------------------------------------
    # GET BOUNDARY
    # --------------------------------------------------------

    points = boundary[slice_number]

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 2:
        print("Invalid boundary format.")
        continue

    # --------------------------------------------------------
    # CREATE FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # ========================================================
    # IMAGE 1 - ORIGINAL CT
    # ========================================================

    axes[0].imshow(
        display_image,
        cmap="gray"
    )

    axes[0].set_title(
        f"CT - Slice {slice_number}"
    )

    axes[0].axis("off")

    # ========================================================
    # IMAGE 2 - CT + MASK
    # ========================================================

    axes[1].imshow(
        display_image,
        cmap="gray"
    )

    mask_overlay = np.ma.masked_where(
        slice_mask == 0,
        slice_mask
    )

    axes[1].imshow(
        mask_overlay,
        cmap="Reds",
        alpha=0.45
    )

    axes[1].set_title(
        "CT + GTV-1 Mask"
    )

    axes[1].axis("off")

    # ========================================================
    # IMAGE 3 - CT + ORIGINAL BOUNDARY
    # ========================================================

    axes[2].imshow(
        display_image,
        cmap="gray"
    )

    # Boundary format is usually [row, col]
    y = points[:, 0]
    x = points[:, 1]

    axes[2].plot(
        x,
        y,
        "r-",
        linewidth=1.5
    )

    axes[2].plot(
        x,
        y,
        "r.",
        markersize=2
    )

    axes[2].set_title(
        "CT + Ordered Boundary"
    )

    axes[2].axis("off")

    # ========================================================
    # IMAGE 4 - MASK + BOUNDARY + POINT NUMBERS
    # ========================================================

    axes[3].imshow(
        slice_mask,
        cmap="gray"
    )

    axes[3].plot(
        x,
        y,
        "r-",
        linewidth=1.5
    )

    # Show every few points to avoid clutter
    step = max(1, len(points) // 20)

    for i in range(0, len(points), step):

        axes[3].text(
            x[i],
            y[i],
            str(i),
            fontsize=7
        )

    axes[3].set_title(
        "Boundary + Ordered Points"
    )

    axes[3].axis("off")

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"visual_check_slice_{slice_number}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:")
    print(output_file)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VISUAL CHECK COMPLETED")
print("=" * 70)

print("\nImages saved in:")

print(OUTPUT_DIR)

print("\nOpen this folder and you should find:")
print("  visual_check_slice_*.png")

print("\nThese images show:")
print("  1. Original CT")
print("  2. CT + GTV-1 mask")
print("  3. CT + ordered boundary")
print("  4. Boundary + ordered points")

print("\nIMPORTANT:")
print("This script does NOT modify:")
print("  GTV1_binary_mask.npy")
print("  ordered_boundary_points.npy")
print("  chain_code_results.npy")

print("\n" + "=" * 70)

input("\nPress ENTER to close...")
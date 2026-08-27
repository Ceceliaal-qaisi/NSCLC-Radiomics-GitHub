import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom

# ============================================================
# STEP 3 VISUALIZATION
# GTV-1 SEGMENTATION + BOUNDARY + RESAMPLING + GRID
# ============================================================

PATIENT = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_FOLDER = os.path.join(PATIENT, "82046")
MASK_FILE = os.path.join(
    PATIENT,
    "GTV1_MASK",
    "GTV1_binary_mask.npy"
)

BOUNDARY_FILE = os.path.join(
    PATIENT,
    "BOUNDARY",
    "ordered_boundary_points.npy"
)

OUTPUT = os.path.join(
    PATIENT,
    "VISUALIZATION"
)

os.makedirs(OUTPUT, exist_ok=True)

print("=" * 70)
print("GTV-1 VISUALIZATION")
print("SEGMENTATION + BOUNDARY + RESAMPLING + GRID")
print("=" * 70)


# ============================================================
# LOAD CT
# ============================================================

print("\nLoading CT slices...")

ct_files = []

for filename in os.listdir(CT_FOLDER):

    if filename.lower().endswith(".dcm"):

        path = os.path.join(CT_FOLDER, filename)

        try:
            ds = pydicom.dcmread(path)

            if getattr(ds, "Modality", "") == "CT":
                ct_files.append(ds)

        except:
            pass


# Sort according to ImagePositionPatient
ct_files.sort(
    key=lambda ds: float(
        ds.ImagePositionPatient[2]
    )
)

print("CT slices:", len(ct_files))


# ============================================================
# CT VOLUME
# ============================================================

ct_volume = np.stack(
    [ds.pixel_array for ds in ct_files]
)

print("CT dimensions:", ct_volume.shape)


# ============================================================
# LOAD MASK
# ============================================================

print("\nLoading GTV-1 mask...")

mask = np.load(MASK_FILE)

print("Mask shape:", mask.shape)


# ============================================================
# LOAD BOUNDARY
# ============================================================

print("\nLoading boundary...")

boundary_data = np.load(
    BOUNDARY_FILE,
    allow_pickle=True
).item()

print("Boundary slices:", len(boundary_data))


# ============================================================
# RESAMPLING FUNCTION
# ============================================================

def resample_boundary(points, target_points=50):

    points = np.asarray(points)

    if len(points) < 3:
        return points

    # Remove consecutive duplicates
    clean = [points[0]]

    for p in points[1:]:

        if not np.array_equal(p, clean[-1]):
            clean.append(p)

    points = np.asarray(clean)

    if len(points) < 3:
        return points

    # Close boundary
    closed = np.vstack([
        points,
        points[0]
    ])

    # Distance between consecutive points
    distances = np.sqrt(
        np.sum(
            np.diff(closed, axis=0) ** 2,
            axis=1
        )
    )

    cumulative = np.concatenate([
        [0],
        np.cumsum(distances)
    ])

    total_length = cumulative[-1]

    if total_length == 0:
        return points

    # Uniform sampling
    new_distances = np.linspace(
        0,
        total_length,
        target_points,
        endpoint=False
    )

    resampled = []

    for d in new_distances:

        idx = np.searchsorted(
            cumulative,
            d,
            side="right"
        ) - 1

        idx = min(
            idx,
            len(closed) - 2
        )

        segment_length = (
            cumulative[idx + 1]
            - cumulative[idx]
        )

        if segment_length == 0:

            alpha = 0

        else:

            alpha = (
                d - cumulative[idx]
            ) / segment_length

        p = (
            closed[idx]
            + alpha *
            (
                closed[idx + 1]
                - closed[idx]
            )
        )

        resampled.append(p)

    return np.asarray(resampled)


# ============================================================
# TUMOR SLICES
# ============================================================

tumor_slices = sorted(
    boundary_data.keys()
)

print("\nTumor slices:", tumor_slices)


# ============================================================
# CREATE VISUALIZATIONS
# ============================================================

for z in tumor_slices:

    print(
        "\nProcessing slice:",
        z
    )

    ct = ct_volume[z]
    tumor = mask[z]

    original_boundary = np.asarray(
        boundary_data[z]
    )

    # --------------------------------------------------------
    # RESAMPLE
    # --------------------------------------------------------

    resampled = resample_boundary(
        original_boundary,
        target_points=50
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )


    # ========================================================
    # IMAGE 1 - SEGMENTATION
    # ========================================================

    axes[0].imshow(
        ct,
        cmap="gray"
    )

    axes[0].imshow(
        np.ma.masked_where(
            tumor == 0,
            tumor
        ),
        cmap="Reds",
        alpha=0.45
    )

    axes[0].set_title(
        "GTV-1 Segmentation"
    )

    axes[0].axis("off")


    # ========================================================
    # IMAGE 2 - ORIGINAL BOUNDARY
    # ========================================================

    axes[1].imshow(
        ct,
        cmap="gray"
    )

    if len(original_boundary) > 0:

        y = original_boundary[:, 0]
        x = original_boundary[:, 1]

        axes[1].plot(
            np.append(x, x[0]),
            np.append(y, y[0]),
            linewidth=2
        )

    axes[1].set_title(
        f"Original Boundary\n{len(original_boundary)} points"
    )

    axes[1].axis("off")


    # ========================================================
    # IMAGE 3 - RESAMPLED BOUNDARY
    # ========================================================

    axes[2].imshow(
        ct,
        cmap="gray"
    )

    if len(resampled) > 0:

        y = resampled[:, 0]
        x = resampled[:, 1]

        axes[2].plot(
            np.append(x, x[0]),
            np.append(y, y[0]),
            linewidth=2
        )

        axes[2].scatter(
            x,
            y,
            s=15
        )

    axes[2].set_title(
        f"Resampled Boundary\n{len(resampled)} points"
    )

    axes[2].axis("off")


    # ========================================================
    # IMAGE 4 - GRID
    # ========================================================

    axes[3].imshow(
        ct,
        cmap="gray"
    )

    if len(resampled) > 0:

        y = resampled[:, 0]
        x = resampled[:, 1]

        axes[3].scatter(
            x,
            y,
            s=20
        )

        # Connect points
        axes[3].plot(
            np.append(x, x[0]),
            np.append(y, y[0]),
            linewidth=1
        )

    axes[3].set_title(
        "Resampled Boundary + Grid Points"
    )

    axes[3].axis("off")


    plt.suptitle(
        f"GTV-1 Processing - Slice {z}",
        fontsize=16
    )

    plt.tight_layout()


    # ========================================================
    # SAVE
    # ========================================================

    filename = os.path.join(
        OUTPUT,
        f"slice_{z:03d}_visualization.png"
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Original:",
        len(original_boundary),
        "| Resampled:",
        len(resampled)
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED")
print("=" * 70)

print("\nImages saved here:")
print(OUTPUT)

print("\nEach image contains:")

print("1. GTV-1 segmentation in RED")
print("2. Original boundary")
print("3. Resampled boundary")
print("4. Resampled boundary + grid points")

print("\n" + "=" * 70)

input("\nPress ENTER to close...")
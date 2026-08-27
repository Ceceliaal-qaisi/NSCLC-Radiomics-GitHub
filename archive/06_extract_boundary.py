import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "82046"
)

SEG_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "9.554"
)


# ============================================================
# LOAD CT
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING CT")
print("=" * 70)

reader = sitk.ImageSeriesReader()

series_ids = reader.GetGDCMSeriesIDs(
    CT_FOLDER
)

series_data = []

for series_id in series_ids:

    files = reader.GetGDCMSeriesFileNames(
        CT_FOLDER,
        series_id
    )

    series_data.append(
        (
            series_id,
            len(files),
            files
        )
    )

series_data.sort(
    key=lambda x: x[1],
    reverse=True
)

reader.SetFileNames(
    series_data[0][2]
)

ct_image = reader.Execute()

ct_array = sitk.GetArrayFromImage(
    ct_image
)

print(
    "CT shape:",
    ct_array.shape
)


# ============================================================
# LOAD SEG
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - LOADING GTV-1")
print("=" * 70)

seg_file = None

for root, dirs, files in os.walk(SEG_FOLDER):

    for file in files:

        if file.lower().endswith(".dcm"):

            seg_file = os.path.join(
                root,
                file
            )

            break

    if seg_file:
        break


seg_ds = pydicom.dcmread(
    seg_file,
    force=True
)

pixel_array = seg_ds.pixel_array


# ============================================================
# GET GTV-1
# ============================================================

gtv_segment_number = 1

frame_segment_numbers = []

for frame in seg_ds.PerFrameFunctionalGroupsSequence:

    number = int(
        frame.SegmentIdentificationSequence[
            0
        ].ReferencedSegmentNumber
    )

    frame_segment_numbers.append(
        number
    )

frame_segment_numbers = np.array(
    frame_segment_numbers
)

gtv_frames = np.where(
    frame_segment_numbers ==
    gtv_segment_number
)[0]


gtv_mask = np.zeros(
    (
        len(gtv_frames),
        512,
        512
    ),
    dtype=np.uint8
)


for i, frame_index in enumerate(
    gtv_frames
):

    gtv_mask[i] = (
        pixel_array[frame_index] > 0
    ).astype(np.uint8)


print(
    "GTV-1 mask shape:",
    gtv_mask.shape
)


# ============================================================
# SELECT SLICE
# ============================================================

slice_index = 74

mask = gtv_mask[
    slice_index
]

ct_slice = ct_array[
    slice_index
]


print(
    "Selected slice:",
    slice_index
)

print(
    "Tumor pixels:",
    np.sum(mask)
)


# ============================================================
# BOUNDARY EXTRACTION FROM SCRATCH
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - EXTRACTING BOUNDARY")
print("=" * 70)


boundary = np.zeros_like(
    mask,
    dtype=np.uint8
)


rows, cols = mask.shape


# Check every tumor pixel

for r in range(rows):

    for c in range(cols):

        # Only process tumor pixels
        if mask[r, c] != 1:
            continue


        # ----------------------------------------------------
        # Check 4-neighbourhood
        # ----------------------------------------------------

        neighbors = []


        if r > 0:
            neighbors.append(
                mask[r - 1, c]
            )
        else:
            neighbors.append(0)


        if r < rows - 1:
            neighbors.append(
                mask[r + 1, c]
            )
        else:
            neighbors.append(0)


        if c > 0:
            neighbors.append(
                mask[r, c - 1]
            )
        else:
            neighbors.append(0)


        if c < cols - 1:
            neighbors.append(
                mask[r, c + 1]
            )
        else:
            neighbors.append(0)


        # ----------------------------------------------------
        # If at least one neighbour is background,
        # this pixel belongs to the boundary.
        # ----------------------------------------------------

        if 0 in neighbors:

            boundary[r, c] = 1


# ============================================================
# BOUNDARY POINTS
# ============================================================

boundary_points = np.column_stack(
    np.where(boundary == 1)
)


print(
    "Number of boundary pixels:",
    len(boundary_points)
)


print(
    "\nFirst 10 boundary points:"
)

for point in boundary_points[:10]:

    print(
        "(row, column) =",
        tuple(point)
    )


# ============================================================
# DISPLAY BINARY MASK
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    mask,
    cmap="gray"
)

plt.title(
    "GTV-1 Binary Mask - Slice 74"
)

plt.axis("off")

plt.show()


# ============================================================
# DISPLAY BOUNDARY
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    mask,
    cmap="gray"
)

boundary_y, boundary_x = np.where(
    boundary == 1
)

plt.scatter(
    boundary_x,
    boundary_y,
    s=2
)

plt.title(
    "GTV-1 Boundary - Slice 74"
)

plt.axis("off")

plt.show()


# ============================================================
# CT + BOUNDARY
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    ct_slice,
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.scatter(
    boundary_x,
    boundary_y,
    s=2
)

plt.title(
    "CT + GTV-1 Boundary - Slice 74"
)

plt.axis("off")

plt.show()


# ============================================================
# SAVE BOUNDARY
# ============================================================

output_file = os.path.join(
    PATIENT_FOLDER,
    "GTV1_boundary_slice74.npy"
)

np.save(
    output_file,
    boundary_points
)

print(
    "\nBoundary points saved to:"
)

print(
    output_file
)


print("\n" + "=" * 70)
print("BOUNDARY EXTRACTION COMPLETE")
print("=" * 70)

input(
    "\nPress Enter to close..."
)
import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
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

BOUNDARY_FILE = os.path.join(
    PATIENT_FOLDER,
    "GTV1_boundary_slice74.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_2_BOUNDARY_RESULTS"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# STEP 1 - LOAD CT
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING CT")
print("=" * 70)

reader = sitk.ImageSeriesReader()

series_ids = reader.GetGDCMSeriesIDs(
    CT_FOLDER
)

if series_ids is None:
    raise RuntimeError(
        "No CT DICOM series found."
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
# STEP 2 - LOAD DICOM SEG
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - LOADING GTV-1")
print("=" * 70)

seg_file = None

for root, dirs, files in os.walk(
    SEG_FOLDER
):

    for file in files:

        if file.lower().endswith(".dcm"):

            seg_file = os.path.join(
                root,
                file
            )

            break

    if seg_file:
        break


if seg_file is None:

    raise RuntimeError(
        "No SEG DICOM file found."
    )


seg_ds = pydicom.dcmread(
    seg_file,
    force=True
)

pixel_array = seg_ds.pixel_array


# ============================================================
# FIND GTV-1
# ============================================================

gtv_segment_number = None

for segment in seg_ds.SegmentSequence:

    number = int(
        segment.SegmentNumber
    )

    description = getattr(
        segment,
        "SegmentDescription",
        ""
    )

    if (
        description.upper() == "GTV-1"
        or
        "GTV-1" in description.upper()
    ):

        gtv_segment_number = number


if gtv_segment_number is None:

    raise RuntimeError(
        "GTV-1 was not found."
    )


print(
    "GTV-1 segment number:",
    gtv_segment_number
)


# ============================================================
# EXTRACT GTV-1 MASK
# ============================================================

frame_segment_numbers = []

for frame in (
    seg_ds.PerFrameFunctionalGroupsSequence
):

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
    frame_segment_numbers
    ==
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


# ============================================================
# SELECT SLICE 74
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
# STEP 3 - LOAD BOUNDARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - LOADING BOUNDARY")
print("=" * 70)

boundary_points = np.load(
    BOUNDARY_FILE
)

print(
    "Boundary points:",
    len(boundary_points)
)


# ============================================================
# CREATE BOUNDARY IMAGE
# ============================================================

boundary_image = np.zeros_like(
    mask,
    dtype=np.uint8
)

for r, c in boundary_points:

    boundary_image[
        int(r),
        int(c)
    ] = 1


# ============================================================
# MOORE NEIGHBOR DIRECTIONS
# ============================================================

directions = [

    (-1, -1),   # upper-left
    (-1,  0),   # up
    (-1,  1),   # upper-right
    ( 0,  1),   # right
    ( 1,  1),   # lower-right
    ( 1,  0),   # down
    ( 1, -1),   # lower-left
    ( 0, -1)    # left

]


# ============================================================
# FIND STARTING POINT
# ============================================================

start_index = np.lexsort(
    (
        boundary_points[:, 1],
        boundary_points[:, 0]
    )
)[0]


start = tuple(
    boundary_points[start_index]
)


print(
    "Starting point:",
    start
)


# ============================================================
# MOORE BOUNDARY TRACKING
# ============================================================

def moore_boundary_trace(
    binary_boundary,
    start_point
):

    rows, cols = binary_boundary.shape

    start_r, start_c = start_point

    current = (
        int(start_r),
        int(start_c)
    )

    previous = (
        int(start_r),
        int(start_c - 1)
    )

    ordered = [
        current
    ]

    max_iterations = (
        rows * cols * 2
    )

    iterations = 0


    while iterations < max_iterations:

        iterations += 1

        current_r, current_c = current

        previous_r, previous_c = previous


        dr = previous_r - current_r
        dc = previous_c - current_c


        try:

            previous_direction = directions.index(
                (dr, dc)
            )

        except ValueError:

            previous_direction = 7


        search_start = (
            previous_direction + 1
        ) % 8


        found_next = False


        for k in range(8):

            direction_index = (
                search_start + k
            ) % 8


            dr, dc = directions[
                direction_index
            ]


            nr = current_r + dr
            nc = current_c + dc


            if nr < 0 or nr >= rows:
                continue

            if nc < 0 or nc >= cols:
                continue


            if binary_boundary[
                nr,
                nc
            ] == 1:

                previous = (
                    current_r,
                    current_c
                )

                current = (
                    nr,
                    nc
                )

                found_next = True

                break


        if not found_next:

            break


        if (
            current == (
                int(start_r),
                int(start_c)
            )
            and
            len(ordered) > 2
        ):

            break


        ordered.append(
            current
        )


    return np.array(
        ordered,
        dtype=int
    )


# ============================================================
# RUN MOORE TRACKING
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - MOORE BOUNDARY TRACKING")
print("=" * 70)

ordered_boundary = moore_boundary_trace(
    boundary_image,
    start
)


print(
    "Ordered boundary points:",
    len(ordered_boundary)
)


unique_points = np.unique(
    ordered_boundary,
    axis=0
)


print(
    "Unique points:",
    len(unique_points)
)

print(
    "Repeated points:",
    len(ordered_boundary)
    -
    len(unique_points)
)


# ============================================================
# SAVE ORDERED BOUNDARY
# ============================================================

ordered_file = os.path.join(
    PATIENT_FOLDER,
    "GTV1_ordered_boundary_slice74.npy"
)

np.save(
    ordered_file,
    ordered_boundary
)


# ============================================================
# FIGURE 1 - BINARY MASK
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

plt.title(
    "GTV-1 Binary Tumor Mask - Slice 74"
)

plt.xlabel(
    "Column (pixel)"
)

plt.ylabel(
    "Row (pixel)"
)

plt.tight_layout()


binary_image_file = os.path.join(
    OUTPUT_FOLDER,
    "01_GTV1_Binary_Mask_Slice74.png"
)

plt.savefig(
    binary_image_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 2 - WHITE TUMOR + BLUE BOUNDARY
# ============================================================

plt.figure(figsize=(8, 8))

# Create an RGB image
# Background = BLACK
# Tumor = WHITE
# Boundary = BLUE
visual = np.zeros(
    (mask.shape[0], mask.shape[1], 3),
    dtype=np.uint8
)

# ------------------------------------------------------------
# Tumor -> WHITE
# ------------------------------------------------------------

visual[mask == 1] = [255, 255, 255]


# ------------------------------------------------------------
# Boundary -> BLUE
# ------------------------------------------------------------

boundary_y, boundary_x = np.where(
    boundary_image == 1
)

visual[
    boundary_y,
    boundary_x
] = [0, 0, 255]


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

plt.imshow(
    visual
)

plt.title(
    "GTV-1 Tumor Mask and Boundary - Slice 74"
)

plt.xlabel(
    "Column (pixel)"
)

plt.ylabel(
    "Row (pixel)"
)

plt.tight_layout()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

boundary_image_file = os.path.join(
    OUTPUT_FOLDER,
    "02_GTV1_WHITE_TUMOR_BLUE_BOUNDARY_Slice74.png"
)

plt.savefig(
    boundary_image_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FIGURE 3 - ORDERED BOUNDARY
# ============================================================

plt.figure(
    figsize=(9, 9)
)

# White tumor + black background

plt.imshow(
    mask,
    cmap="gray",
    vmin=0,
    vmax=1
)


# Ordered boundary

ordered_x = ordered_boundary[:, 1]
ordered_y = ordered_boundary[:, 0]


plt.plot(
    ordered_x,
    ordered_y,
    linewidth=2
)


# ------------------------------------------------------------
# Mark START
# ------------------------------------------------------------

plt.scatter(
    start[1],
    start[0],
    s=100,
    marker="o"
)


plt.text(
    start[1] + 6,
    start[0] - 8,
    "START",
    fontsize=12,
    fontweight="bold"
)


# ------------------------------------------------------------
# Number selected points
# ------------------------------------------------------------

number_step = 20


for i in range(
    0,
    len(ordered_boundary),
    number_step
):

    r = ordered_boundary[i, 0]
    c = ordered_boundary[i, 1]


    plt.scatter(
        c,
        r,
        s=30
    )


    plt.text(
        c + 5,
        r - 5,
        str(i + 1),
        fontsize=10,
        fontweight="bold"
    )


plt.title(
    "GTV-1 Ordered Boundary - Moore Boundary Tracking"
)

plt.xlabel(
    "Column (pixel)"
)

plt.ylabel(
    "Row (pixel)"
)

plt.tight_layout()


ordered_image_file = os.path.join(
    OUTPUT_FOLDER,
    "03_GTV1_Ordered_Boundary_Moore_Slice74.png"
)

plt.savefig(
    ordered_image_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FIGURE 4 - CT + ORDERED BOUNDARY
# ============================================================

plt.figure(
    figsize=(8, 8)
)


plt.imshow(
    ct_slice,
    cmap="gray",
    vmin=-1000,
    vmax=400
)


plt.plot(
    ordered_x,
    ordered_y,
    linewidth=2
)


plt.scatter(
    start[1],
    start[0],
    s=100,
    marker="o"
)


plt.text(
    start[1] + 6,
    start[0] - 8,
    "START",
    fontsize=12,
    fontweight="bold"
)


plt.title(
    "CT + GTV-1 Ordered Boundary - Slice 74"
)

plt.xlabel(
    "Column (pixel)"
)

plt.ylabel(
    "Row (pixel)"
)

plt.tight_layout()


ct_boundary_file = os.path.join(
    OUTPUT_FOLDER,
    "04_CT_With_GTV1_Ordered_Boundary_Slice74.png"
)

plt.savefig(
    ct_boundary_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# SAVE POINT NUMBERS + COORDINATES
# ============================================================

txt_file = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_Ordered_Boundary_Points.txt"
)


with open(
    txt_file,
    "w"
) as f:

    f.write(
        "GTV-1 Ordered Boundary Points\n"
    )

    f.write(
        "Slice: 74\n"
    )

    f.write(
        "Method: Moore Boundary Tracking\n\n"
    )

    f.write(
        "Point_Number,Row,Column\n"
    )

    for i, point in enumerate(
        ordered_boundary
    ):

        f.write(
            f"{i + 1},{point[0]},{point[1]}\n"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)

print(
    "\nBinary tumor pixels:",
    np.sum(mask)
)

print(
    "Boundary points:",
    len(boundary_points)
)

print(
    "Ordered boundary points:",
    len(ordered_boundary)
)

print(
    "Unique ordered points:",
    len(unique_points)
)

print(
    "Repeated points:",
    len(ordered_boundary)
    -
    len(unique_points)
)

print(
    "\nFiles saved in:"
)

print(
    OUTPUT_FOLDER
)

print(
    "\n1.",
    binary_image_file
)

print(
    "2.",
    boundary_image_file
)

print(
    "3.",
    ordered_image_file
)

print(
    "4.",
    ct_boundary_file
)

print(
    "5.",
    txt_file
)

print(
    "\nOrdered boundary array:"
)

print(
    ordered_file
)

input(
    "\nPress Enter to close..."
)
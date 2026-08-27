import os
import numpy as np

# ============================================================
# PROJECT 7
# Radiomic Feature Extraction and Outcome Classification
# Lung1 NSCLC-Radiomics
#
# STEP 2:
# Binary Tumor Mask
#        ↓
# Moore Boundary Tracking
#        ↓
# Ordered Boundary Points
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

MASK_FILE = os.path.join(
    PATIENT_FOLDER,
    "GTV1_MASK",
    "GTV1_binary_mask.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "BOUNDARY"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# 2. HEADER
# ------------------------------------------------------------

print("=" * 60)
print("STEP 2 - MOORE BOUNDARY TRACKING")
print("=" * 60)

print("\nBinary mask:")
print(MASK_FILE)

print("\nOutput folder:")
print(OUTPUT_FOLDER)


# ------------------------------------------------------------
# 3. CHECK MASK
# ------------------------------------------------------------

if not os.path.exists(MASK_FILE):

    print("\nERROR: Binary mask not found!")

    input("\nPress ENTER to close...")
    exit()


# ------------------------------------------------------------
# 4. LOAD MASK
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Loading binary tumor mask...")
print("-" * 60)

mask = np.load(MASK_FILE)

print("Mask loaded successfully.")

print("\nMask shape:")
print(mask.shape)

print("Mask data type:")
print(mask.dtype)

print("Tumor pixels:")
print(np.sum(mask))


# ------------------------------------------------------------
# 5. MOORE NEIGHBORHOOD
# ------------------------------------------------------------

# 8-neighborhood
#
#       0 1 2
#       7 P 3
#       6 5 4
#
# Clockwise order

NEIGHBORS = [
    (-1, -1),   # 0
    (-1,  0),   # 1
    (-1,  1),   # 2
    ( 0,  1),   # 3
    ( 1,  1),   # 4
    ( 1,  0),   # 5
    ( 1, -1),   # 6
    ( 0, -1)    # 7
]


# ------------------------------------------------------------
# 6. FIND FIRST BOUNDARY PIXEL
# ------------------------------------------------------------

def find_starting_pixel(binary_image):

    rows, cols = binary_image.shape

    for r in range(rows):

        for c in range(cols):

            if binary_image[r, c] == 1:

                # Check whether this pixel touches background

                for dr, dc in NEIGHBORS:

                    nr = r + dr
                    nc = c + dc

                    if (
                        nr < 0
                        or nr >= rows
                        or nc < 0
                        or nc >= cols
                    ):
                        return (r, c)

                    if binary_image[nr, nc] == 0:
                        return (r, c)

    return None


# ------------------------------------------------------------
# 7. MOORE BOUNDARY TRACKING
# ------------------------------------------------------------

def moore_boundary(binary_image):

    start = find_starting_pixel(binary_image)

    if start is None:
        return []

    boundary = []

    current = start

    # Backtracking pixel
    backtrack = (
        start[0],
        start[1] - 1
    )

    start_backtrack = backtrack

    max_iterations = binary_image.size * 8

    iterations = 0

    while iterations < max_iterations:

        boundary.append(current)

        current_r, current_c = current

        back_r, back_c = backtrack

        # ----------------------------------------------------
        # Find direction from current pixel to backtrack pixel
        # ----------------------------------------------------

        dr = back_r - current_r
        dc = back_c - current_c

        direction = None

        for i, (ndr, ndc) in enumerate(NEIGHBORS):

            if ndr == dr and ndc == dc:

                direction = i
                break

        # If backtrack is not an immediate neighbor
        if direction is None:

            direction = 7

        # ----------------------------------------------------
        # Search clockwise starting after backtrack
        # ----------------------------------------------------

        found_next = False

        for k in range(8):

            search_direction = (
                direction + 1 + k
            ) % 8

            ndr, ndc = NEIGHBORS[
                search_direction
            ]

            nr = current_r + ndr
            nc = current_c + ndc

            # Outside image
            if (
                nr < 0
                or nr >= binary_image.shape[0]
                or nc < 0
                or nc >= binary_image.shape[1]
            ):
                continue

            # Foreground pixel
            if binary_image[nr, nc] == 1:

                # The pixel before the next pixel
                previous_direction = (
                    search_direction - 1
                ) % 8

                pdr, pdc = NEIGHBORS[
                    previous_direction
                ]

                backtrack = (
                    current_r + pdr,
                    current_c + pdc
                )

                next_pixel = (
                    nr,
                    nc
                )

                found_next = True

                break

        if not found_next:
            break

        # ----------------------------------------------------
        # Stopping condition
        # ----------------------------------------------------

        if (
            next_pixel == start
            and backtrack == start_backtrack
            and len(boundary) > 2
        ):
            break

        current = next_pixel

        iterations += 1

    return boundary


# ------------------------------------------------------------
# 8. PROCESS EACH TUMOR SLICE
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Tracking tumor boundaries...")
print("-" * 60)

all_boundaries = {}

tumor_slices = np.where(
    np.sum(mask, axis=(1, 2)) > 0
)[0]

print(
    "\nNumber of tumor slices:",
    len(tumor_slices)
)


for slice_index in tumor_slices:

    print(
        f"\nProcessing slice "
        f"{slice_index + 1}/{mask.shape[0]}"
    )

    binary_slice = mask[
        slice_index
    ]

    boundary = moore_boundary(
        binary_slice
    )

    if len(boundary) == 0:

        print("Boundary NOT found.")

        continue

    all_boundaries[
        int(slice_index)
    ] = boundary

    print(
        "Boundary points:",
        len(boundary)
    )


# ------------------------------------------------------------
# 9. SAVE BOUNDARY POINTS
# ------------------------------------------------------------

boundary_file = os.path.join(
    OUTPUT_FOLDER,
    "ordered_boundary_points.npy"
)

np.save(
    boundary_file,
    all_boundaries,
    allow_pickle=True
)


print("\n" + "-" * 60)
print("Saving results...")
print("-" * 60)

print("\nBoundary file:")
print(boundary_file)


# ------------------------------------------------------------
# 10. SAVE HUMAN-READABLE TEXT
# ------------------------------------------------------------

text_file = os.path.join(
    OUTPUT_FOLDER,
    "boundary_info.txt"
)

with open(text_file, "w") as f:

    f.write(
        "LUNG1-001 Moore Boundary Tracking\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(
        f"Mask shape: {mask.shape}\n"
    )

    f.write(
        f"Tumor slices: {len(tumor_slices)}\n\n"
    )

    for slice_index in tumor_slices:

        if slice_index in all_boundaries:

            number_of_points = len(
                all_boundaries[slice_index]
            )

            f.write(
                f"Slice {slice_index}: "
                f"{number_of_points} boundary points\n"
            )


print("\nInformation file:")
print(text_file)


# ------------------------------------------------------------
# 11. FINAL CHECK
# ------------------------------------------------------------

total_boundaries = len(
    all_boundaries
)

total_points = sum(
    len(points)
    for points in all_boundaries.values()
)


print("\n" + "=" * 60)
print("STEP 2 COMPLETED")
print("=" * 60)

print("\nTumor slices:")
print(len(tumor_slices))

print("\nBoundaries successfully extracted:")
print(total_boundaries)

print("\nTotal ordered boundary points:")
print(total_points)

print("\nOutput folder:")
print(OUTPUT_FOLDER)

print("\nNext step:")
print("Chain Code + First Difference + Shape Number")

print("\n" + "=" * 60)

input("\nPress ENTER to close...")
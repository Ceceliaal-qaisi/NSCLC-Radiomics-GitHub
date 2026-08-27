```python
# ================================================================
# STEP 3 - BOUNDARY PROCESSING
# Sampling + Grid Quantization
# Chain Code + First Difference
# Normalized First Difference + Shape Number
# Boundary Signature + Boundary Straightness
# Visualization
# ================================================================

import os
import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

BOUNDARY_FILE = os.path.join(
    PATIENT_FOLDER,
    "BOUNDARY",
    "ordered_boundary_points.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "CHAIN_CODE"
)

FIGURE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "FIGURES"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

# Sampling interval.
# Every 5th boundary point is initially sampled.
SAMPLING_STEP = 5

# Grid resolution.
# Coordinates are quantized to integer pixel locations.
GRID_SIZE = 1


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def remove_consecutive_duplicates(points):
    """
    Remove consecutive duplicate boundary points.
    """
    points = np.asarray(points)

    if len(points) == 0:
        return points

    cleaned = [points[0]]

    for p in points[1:]:
        if not np.array_equal(p, cleaned[-1]):
            cleaned.append(p)

    return np.asarray(cleaned)


# ----------------------------------------------------------------

def sample_boundary(points, step=5):
    """
    Uniform boundary sampling.

    Keeps every 'step'-th point and always attempts
    to retain the final point.
    """
    points = np.asarray(points)

    if len(points) <= 2:
        return points.copy()

    sampled = points[::step]

    # Make sure the final point is represented.
    if not np.array_equal(sampled[-1], points[-1]):
        sampled = np.vstack((sampled, points[-1]))

    return sampled


# ----------------------------------------------------------------

def grid_quantization(points, grid_size=1):
    """
    Quantize coordinates onto a rectangular pixel grid.
    """
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return points

    grid_points = np.round(points / grid_size) * grid_size

    return grid_points.astype(int)


# ----------------------------------------------------------------

def direction_code(dr, dc):
    """
    8-connected Freeman chain-code direction.

             2 1 0
             3   7
             4 5 6

    Coordinates are treated as (row, column).
    """

    if dr == 0 and dc > 0:
        return 0       # right

    if dr < 0 and dc > 0:
        return 1       # up-right

    if dr < 0 and dc == 0:
        return 2       # up

    if dr < 0 and dc < 0:
        return 3       # up-left

    if dr == 0 and dc < 0:
        return 4       # left

    if dr > 0 and dc < 0:
        return 5       # down-left

    if dr > 0 and dc == 0:
        return 6       # down

    if dr > 0 and dc > 0:
        return 7       # down-right

    return None


# ----------------------------------------------------------------

def compute_chain_code(points):
    """
    Calculate 8-connected Freeman chain code.
    """

    points = np.asarray(points)

    if len(points) < 2:
        return np.array([], dtype=int)

    chain = []

    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]

        dr = int(p2[0] - p1[0])
        dc = int(p2[1] - p1[1])

        # For grid quantization, a displacement may be larger
        # than one pixel. Reduce it to its direction.
        dr = np.sign(dr)
        dc = np.sign(dc)

        code = direction_code(dr, dc)

        if code is not None:
            chain.append(code)

    return np.asarray(chain, dtype=int)


# ----------------------------------------------------------------

def compute_first_difference(chain_code):
    """
    First difference of a circular chain code.

    d(i) = (c(i+1) - c(i)) mod 8
    """

    chain_code = np.asarray(chain_code, dtype=int)

    if len(chain_code) == 0:
        return np.array([], dtype=int)

    next_code = np.roll(chain_code, -1)

    first_difference = (next_code - chain_code) % 8

    return first_difference.astype(int)


# ----------------------------------------------------------------

def normalize_first_difference(first_difference):
    """
    Rotation normalization of the first difference.

    The sequence is circularly shifted so that its
    lexicographically smallest rotation is selected.
    """

    fd = np.asarray(first_difference, dtype=int)

    if len(fd) == 0:
        return fd

    rotations = np.array([
        np.roll(fd, -i)
        for i in range(len(fd))
    ])

    normalized = rotations[
        np.lexsort(rotations.T[::-1])[0]
    ]

    return normalized.astype(int)


# ----------------------------------------------------------------

def compute_shape_number(first_difference):
    """
    Shape number is represented by the normalized
    first-difference sequence.
    """

    return np.asarray(first_difference, dtype=int).copy()


# ----------------------------------------------------------------

def compute_boundary_signature(points):
    """
    Boundary signature based on radial distance from centroid.

    Signature:
        r(i) = sqrt((x-xc)^2 + (y-yc)^2)
    """

    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.array([], dtype=float)

    centroid = np.mean(points, axis=0)

    distances = np.sqrt(
        (points[:, 0] - centroid[0]) ** 2 +
        (points[:, 1] - centroid[1]) ** 2
    )

    return distances


# ----------------------------------------------------------------

def compute_boundary_straightness(points):
    """
    Boundary straightness:

        direct distance between first and last point
        ------------------------------------------------
        total boundary path length

    The value is between 0 and 1 for normal boundaries.
    """

    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        return 0.0

    differences = np.diff(points, axis=0)

    segment_lengths = np.sqrt(
        differences[:, 0] ** 2 +
        differences[:, 1] ** 2
    )

    total_length = np.sum(segment_lengths)

    if total_length == 0:
        return 0.0

    direct_distance = np.linalg.norm(
        points[-1] - points[0]
    )

    return float(direct_distance / total_length)


# ----------------------------------------------------------------

def basic_rectangle(points):
    """
    Calculate the axis-aligned bounding rectangle.
    """

    points = np.asarray(points)

    if len(points) == 0:
        return {
            "min_row": 0,
            "max_row": 0,
            "min_col": 0,
            "max_col": 0,
            "height": 0,
            "width": 0
        }

    min_row = int(np.min(points[:, 0]))
    max_row = int(np.max(points[:, 0]))

    min_col = int(np.min(points[:, 1]))
    max_col = int(np.max(points[:, 1]))

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    return {
        "min_row": min_row,
        "max_row": max_row,
        "min_col": min_col,
        "max_col": max_col,
        "height": height,
        "width": width
    }


# ================================================================
# VISUALIZATION
# ================================================================

def save_boundary_figure(
    original,
    sampled,
    grid_points,
    rectangle,
    signature,
    slice_number
):
    """
    Save visual verification figure.

    The figure contains:
      1. Original boundary
      2. Sampled boundary
      3. Grid-quantized boundary
      4. Bounding rectangle
      5. Boundary signature
    """

    fig = plt.figure(figsize=(12, 9))

    # ------------------------------------------------------------
    # Original + sampling
    # ------------------------------------------------------------

    ax1 = fig.add_subplot(2, 2, 1)

    ax1.plot(
        original[:, 1],
        original[:, 0],
        linewidth=1
    )

    ax1.scatter(
        sampled[:, 1],
        sampled[:, 0],
        s=12
    )

    ax1.set_title(
        f"Slice {slice_number} - Original + Sampling"
    )

    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    ax1.invert_yaxis()
    ax1.axis("equal")
    ax1.grid(True)


    # ------------------------------------------------------------
    # Grid boundary
    # ------------------------------------------------------------

    ax2 = fig.add_subplot(2, 2, 2)

    ax2.plot(
        grid_points[:, 1],
        grid_points[:, 0],
        marker="o",
        markersize=3,
        linewidth=1
    )

    ax2.set_title(
        f"Slice {slice_number} - Grid Quantized Boundary"
    )

    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")

    ax2.invert_yaxis()
    ax2.axis("equal")
    ax2.grid(True)


    # ------------------------------------------------------------
    # Bounding rectangle
    # ------------------------------------------------------------

    ax3 = fig.add_subplot(2, 2, 3)

    ax3.plot(
        original[:, 1],
        original[:, 0],
        linewidth=1
    )

    min_row = rectangle["min_row"]
    max_row = rectangle["max_row"]

    min_col = rectangle["min_col"]
    max_col = rectangle["max_col"]

    rectangle_x = [
        min_col,
        max_col,
        max_col,
        min_col,
        min_col
    ]

    rectangle_y = [
        min_row,
        min_row,
        max_row,
        max_row,
        min_row
    ]

    ax3.plot(
        rectangle_x,
        rectangle_y,
        linewidth=2
    )

    ax3.set_title(
        f"Slice {slice_number} - Basic Rectangle"
    )

    ax3.set_xlabel("Column")
    ax3.set_ylabel("Row")

    ax3.invert_yaxis()
    ax3.axis("equal")
    ax3.grid(True)


    # ------------------------------------------------------------
    # Boundary signature
    # ------------------------------------------------------------

    ax4 = fig.add_subplot(2, 2, 4)

    ax4.plot(
        signature,
        linewidth=1
    )

    ax4.set_title(
        f"Slice {slice_number} - Boundary Signature"
    )

    ax4.set_xlabel("Boundary Point")
    ax4.set_ylabel("Radial Distance")

    ax4.grid(True)


    # ------------------------------------------------------------

    fig.suptitle(
        f"GTV-1 Boundary Processing - Slice {slice_number}",
        fontsize=14
    )

    fig.tight_layout()

    output_file = os.path.join(
        FIGURE_FOLDER,
        f"slice_{slice_number:03d}.png"
    )

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ================================================================
# LOAD BOUNDARY DATA
# ================================================================

print("=" * 70)
print("STEP 3 - BOUNDARY PROCESSING")
print("SAMPLING + GRID + CHAIN CODE")
print("FIRST DIFFERENCE + NORMALIZATION")
print("BASIC RECTANGLE + SHAPE NUMBER")
print("BOUNDARY SIGNATURE + STRAIGHTNESS")
print("VISUALIZATION")
print("=" * 70)

print()
print("Boundary file:")
print(BOUNDARY_FILE)

print()
print("-" * 60)
print("Loading boundary...")
print("-" * 60)

if not os.path.exists(BOUNDARY_FILE):
    print("ERROR: Boundary file not found!")
    input("Press ENTER to close...")
    raise SystemExit

loaded = np.load(
    BOUNDARY_FILE,
    allow_pickle=True
)

# The previous Step 2 saved the dictionary as a
# zero-dimensional NumPy object.
if loaded.shape == ():
    boundary_data = loaded.item()
else:
    boundary_data = loaded

print("Boundary loaded successfully.")

if isinstance(boundary_data, dict):
    print("Boundary container: dictionary")
else:
    print("Boundary container:", type(boundary_data))

print("Number of slices:", len(boundary_data))


# ================================================================
# PROCESS
# ================================================================

results = {}

total_original_points = 0
total_sampled_points = 0
total_grid_points = 0

print()
print("=" * 70)
print("PROCESSING")
print("=" * 70)


for slice_number, boundary in sorted(
    boundary_data.items(),
    key=lambda x: int(x[0])
):

    print()
    print("-" * 60)
    print(f"Slice: {slice_number}")
    print("-" * 60)

    original = np.asarray(boundary)

    original = remove_consecutive_duplicates(original)

    print(
        "Original boundary points:",
        len(original)
    )

    total_original_points += len(original)


    # ------------------------------------------------------------
    # SAMPLING
    # ------------------------------------------------------------

    sampled = sample_boundary(
        original,
        SAMPLING_STEP
    )

    print(
        "Sampled points:",
        len(sampled)
    )

    total_sampled_points += len(sampled)


    # ------------------------------------------------------------
    # GRID
    # ------------------------------------------------------------

    grid_points = grid_quantization(
        sampled,
        GRID_SIZE
    )

    # Remove consecutive duplicates introduced by
    # grid quantization.
    grid_points = remove_consecutive_duplicates(
        grid_points
    )

    print(
        "Grid points:",
        len(grid_points)
    )

    total_grid_points += len(grid_points)


    # ------------------------------------------------------------
    # BASIC RECTANGLE
    # ------------------------------------------------------------

    rectangle = basic_rectangle(
        grid_points
    )

    print(
        f"Basic rectangle: "
        f"{rectangle['height']} x "
        f"{rectangle['width']}"
    )


    # ------------------------------------------------------------
    # CHAIN CODE
    # ------------------------------------------------------------

    chain_code = compute_chain_code(
        grid_points
    )

    print(
        "Chain code length:",
        len(chain_code)
    )


    # ------------------------------------------------------------
    # FIRST DIFFERENCE
    # ------------------------------------------------------------

    first_difference = compute_first_difference(
        chain_code
    )

    print(
        "First difference length:",
        len(first_difference)
    )


    # ------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------

    normalized_first_difference = (
        normalize_first_difference(
            first_difference
        )
    )

    print(
        "Normalized first difference calculated."
    )


    # ------------------------------------------------------------
    # SHAPE NUMBER
    # ------------------------------------------------------------

    shape_number = compute_shape_number(
        normalized_first_difference
    )

    print(
        "Shape number calculated."
    )


    # ------------------------------------------------------------
    # BOUNDARY SIGNATURE
    # ------------------------------------------------------------

    boundary_signature = (
        compute_boundary_signature(
            grid_points
        )
    )

    print(
        "Boundary signature calculated."
    )


    # ------------------------------------------------------------
    # BOUNDARY STRAIGHTNESS
    # ------------------------------------------------------------

    straightness = compute_boundary_straightness(
        grid_points
    )

    print(
        f"Boundary straightness: "
        f"{straightness:.6f}"
    )


    # ------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------

    save_boundary_figure(
        original,
        sampled,
        grid_points,
        rectangle,
        boundary_signature,
        slice_number
    )

    print(
        "Visualization saved."
    )


    # ------------------------------------------------------------
    # STORE RESULTS
    # ------------------------------------------------------------

    results[int(slice_number)] = {

        "original_boundary": original,

        "sampled_boundary": sampled,

        "grid_boundary": grid_points,

        "basic_rectangle": rectangle,

        "chain_code": chain_code,

        "first_difference": first_difference,

        "normalized_first_difference":
            normalized_first_difference,

        "shape_number": shape_number,

        "boundary_signature":
            boundary_signature,

        "boundary_straightness":
            straightness
    }


# ================================================================
# SAVE RESULTS
# ================================================================

print()
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)


results_file = os.path.join(
    OUTPUT_FOLDER,
    "chain_code_results.npy"
)

np.save(
    results_file,
    results,
    allow_pickle=True
)


# ================================================================
# INFORMATION FILE
# ================================================================

info_file = os.path.join(
    OUTPUT_FOLDER,
    "chain_code_info.txt"
)

with open(
    info_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 3 - BOUNDARY PROCESSING\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Input boundary file:\n"
    )

    f.write(
        BOUNDARY_FILE + "\n\n"
    )

    f.write(
        f"Number of slices: "
        f"{len(results)}\n"
    )

    f.write(
        f"Sampling step: "
        f"{SAMPLING_STEP}\n"
    )

    f.write(
        f"Grid size: "
        f"{GRID_SIZE}\n\n"
    )

    f.write(
        f"Total original boundary points: "
        f"{total_original_points}\n"
    )

    f.write(
        f"Total sampled points: "
        f"{total_sampled_points}\n"
    )

    f.write(
        f"Total grid points: "
        f"{total_grid_points}\n\n"
    )

    f.write(
        "Features calculated:\n"
    )

    f.write(
        "1. Sampling\n"
    )

    f.write(
        "2. Grid quantization\n"
    )

    f.write(
        "3. Basic rectangular bounding box\n"
    )

    f.write(
        "4. Freeman chain code\n"
    )

    f.write(
        "5. First difference\n"
    )

    f.write(
        "6. Normalized first difference\n"
    )

    f.write(
        "7. Shape number\n"
    )

    f.write(
        "8. Boundary signature\n"
    )

    f.write(
        "9. Boundary straightness\n"
    )

    f.write(
        "\nVisualization folder:\n"
    )

    f.write(
        FIGURE_FOLDER + "\n"
    )


# ================================================================
# FINAL MESSAGE
# ================================================================

print()
print("=" * 70)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Slices processed:",
    len(results)
)

print(
    "Sampling: DONE"
)

print(
    "Grid: DONE"
)

print(
    "Basic rectangle: DONE"
)

print(
    "Chain code: DONE"
)

print(
    "First difference: DONE"
)

print(
    "Normalization: DONE"
)

print(
    "Shape number: DONE"
)

print(
    "Boundary signature: DONE"
)

print(
    "Boundary straightness: DONE"
)

print(
    "Visualization: DONE"
)

print()
print("Results:")
print(results_file)

print()
print("Information:")
print(info_file)

print()
print("Figures:")
print(FIGURE_FOLDER)

print()
print("=" * 70)
print("NEXT STEP: FOURIER DESCRIPTORS")
print("=" * 70)

input("Press ENTER to close...")
```

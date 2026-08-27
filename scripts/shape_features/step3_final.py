import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 3 - FINAL BOUNDARY PREPROCESSING + SHAPE DESCRIPTORS
#
# 1. Uniform boundary resampling
# 2. Grid quantization
# 3. 8-direction chain code
# 4. First difference
# 5. Normalized first difference
# 6. Shape number
# 7. Boundary signature
# 8. Boundary straightness
# 9. Visualization
# ============================================================

PATIENT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

BOUNDARY_DIR = os.path.join(PATIENT_DIR, "BOUNDARY")
OUTPUT_DIR = os.path.join(PATIENT_DIR, "STEP3_FINAL")

os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_FILE = os.path.join(
    BOUNDARY_DIR,
    "ordered_boundary_points.npy"
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

# Number of uniformly resampled points.
# This is a processing parameter, not a claim from the book.
TARGET_POINTS = 64

# Grid size used for coordinate quantization.
GRID_SIZE = 64

# ------------------------------------------------------------
# PRINT HEADER
# ------------------------------------------------------------

print("=" * 70)
print("STEP 3 - FINAL BOUNDARY PROCESSING")
print("=" * 70)

print("Uniform Resampling")
print("Grid Quantization")
print("Chain Code")
print("First Difference")
print("Normalized First Difference")
print("Shape Number")
print("Boundary Signature")
print("Boundary Straightness")
print("=" * 70)

# ------------------------------------------------------------
# LOAD BOUNDARY
# ------------------------------------------------------------

print("\nLoading boundary...")

data = np.load(
    BOUNDARY_FILE,
    allow_pickle=True
)

if data.shape == ():
    boundaries = data.item()
else:
    boundaries = data

print("Boundary loaded successfully.")
print("Number of slices:", len(boundaries))

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def remove_duplicate_points(points):
    """
    Remove consecutive duplicate points.
    """

    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return points

    cleaned = [points[0]]

    for p in points[1:]:

        if not np.array_equal(p, cleaned[-1]):
            cleaned.append(p)

    return np.asarray(cleaned)


def close_boundary(points):
    """
    Make sure the boundary is treated as closed.
    """

    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return points

    if not np.array_equal(points[0], points[-1]):

        points = np.vstack(
            [points, points[0]]
        )

    return points


def cumulative_distance(points):
    """
    Calculate cumulative Euclidean distance
    along a closed boundary.
    """

    points = close_boundary(points)

    diffs = np.diff(points, axis=0)

    distances = np.sqrt(
        np.sum(diffs ** 2, axis=1)
    )

    cumulative = np.concatenate(
        ([0.0], np.cumsum(distances))
    )

    return points, cumulative


def uniform_resample(points, n_points):
    """
    Uniform resampling along boundary arc length.

    The new points are equally spaced according
    to the cumulative boundary distance.
    """

    points = remove_duplicate_points(points)

    if len(points) < 3:
        return points

    closed_points, cumulative = cumulative_distance(points)

    total_length = cumulative[-1]

    if total_length == 0:
        return points

    target_distances = np.linspace(
        0,
        total_length,
        n_points,
        endpoint=False
    )

    resampled = []

    for d in target_distances:

        idx = np.searchsorted(
            cumulative,
            d,
            side="right"
        ) - 1

        idx = max(
            0,
            min(
                idx,
                len(closed_points) - 2
            )
        )

        d0 = cumulative[idx]
        d1 = cumulative[idx + 1]

        p0 = closed_points[idx]
        p1 = closed_points[idx + 1]

        if d1 == d0:

            point = p0.copy()

        else:

            alpha = (
                d - d0
            ) / (
                d1 - d0
            )

            point = (
                p0 +
                alpha * (p1 - p0)
            )

        resampled.append(point)

    return np.asarray(resampled)


def grid_quantize(points, grid_size):
    """
    Map coordinates to a normalized square grid.
    """

    points = np.asarray(points, dtype=float)

    y = points[:, 0]
    x = points[:, 1]

    min_y = np.min(y)
    max_y = np.max(y)

    min_x = np.min(x)
    max_x = np.max(x)

    if max_y == min_y:
        gy = np.zeros(len(y), dtype=int)

    else:

        gy = np.round(
            (y - min_y) /
            (max_y - min_y) *
            (grid_size - 1)
        ).astype(int)

    if max_x == min_x:
        gx = np.zeros(len(x), dtype=int)

    else:

        gx = np.round(
            (x - min_x) /
            (max_x - min_x) *
            (grid_size - 1)
        ).astype(int)

    return np.column_stack(
        (gy, gx)
    )


def direction_code(dy, dx):
    """
    8-connected Freeman chain code.

          2 1 0
          3   7
          4 5 6

    Coordinates are image coordinates:
    row increases downward.
    """

    mapping = {

        (0, 1): 0,
        (-1, 1): 1,
        (-1, 0): 2,
        (-1, -1): 3,

        (0, -1): 4,
        (1, -1): 5,
        (1, 0): 6,
        (1, 1): 7
    }

    return mapping.get(
        (dy, dx),
        None
    )


def chain_code(points):
    """
    Calculate 8-direction chain code.
    """

    codes = []

    n = len(points)

    for i in range(n):

        p1 = points[i]

        p2 = points[
            (i + 1) % n
        ]

        dy = int(
            p2[0] - p1[0]
        )

        dx = int(
            p2[1] - p1[1]
        )

        # Handle possible jumps after quantization
        if abs(dy) > 1:
            dy = int(np.sign(dy))

        if abs(dx) > 1:
            dx = int(np.sign(dx))

        code = direction_code(
            dy,
            dx
        )

        if code is not None:
            codes.append(code)

    return np.asarray(
        codes,
        dtype=int
    )


def first_difference(chain):

    chain = np.asarray(
        chain,
        dtype=int
    )

    if len(chain) == 0:
        return chain

    diff = np.zeros_like(chain)

    for i in range(len(chain)):

        diff[i] = (
            chain[
                (i + 1) % len(chain)
            ]
            -
            chain[i]
        ) % 8

    return diff


def normalize_first_difference(fd):
    """
    Rotation normalization.

    Canonical representation is obtained by
    cyclically rotating the first-difference sequence
    to its lexicographically smallest rotation.
    """

    fd = np.asarray(
        fd,
        dtype=int
    )

    if len(fd) == 0:
        return fd

    rotations = [
        np.roll(fd, -i)
        for i in range(len(fd))
    ]

    normalized = min(
        rotations,
        key=lambda x: tuple(x)
    )

    return np.asarray(
        normalized,
        dtype=int
    )


def shape_number(fd):
    """
    Shape number = lexicographically smallest
    cyclic rotation of the first difference.

    The normalized first difference is used
    as the rotation-invariant representation.
    """

    fd = np.asarray(
        fd,
        dtype=int
    )

    if len(fd) == 0:
        return fd

    rotations = [
        np.roll(fd, -i)
        for i in range(len(fd))
    ]

    return np.asarray(
        min(
            rotations,
            key=lambda x: tuple(x)
        ),
        dtype=int
    )


def boundary_signature(points):

    points = np.asarray(
        points,
        dtype=float
    )

    center = np.mean(
        points,
        axis=0
    )

    distances = np.sqrt(
        np.sum(
            (points - center) ** 2,
            axis=1
        )
    )

    if np.max(distances) > 0:

        distances = (
            distances /
            np.max(distances)
        )

    return distances


def boundary_straightness(points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) < 2:
        return 0.0

    closed = np.vstack(
        [
            points,
            points[0]
        ]
    )

    segment_lengths = np.sqrt(
        np.sum(
            np.diff(
                closed,
                axis=0
            ) ** 2,
            axis=1
        )
    )

    perimeter = np.sum(
        segment_lengths
    )

    start = points[0]
    end = points[len(points) // 2]

    chord = np.linalg.norm(
        end - start
    )

    if perimeter == 0:
        return 0.0

    return chord / (
        perimeter / 2.0
    )


def basic_rectangle(points):

    y = points[:, 0]
    x = points[:, 1]

    height = (
        np.max(y) -
        np.min(y)
    )

    width = (
        np.max(x) -
        np.min(x)
    )

    return (
        float(height),
        float(width)
    )

# ------------------------------------------------------------
# PROCESS ALL SLICES
# ------------------------------------------------------------

results = {}

print("\n" + "=" * 70)
print("PROCESSING")
print("=" * 70)

for slice_number in sorted(
    boundaries.keys(),
    key=lambda x: int(x)
):

    print("\n" + "-" * 60)
    print("Slice:", slice_number)
    print("-" * 60)

    original = np.asarray(
        boundaries[slice_number],
        dtype=float
    )

    original = remove_duplicate_points(
        original
    )

    print(
        "Original boundary points:",
        len(original)
    )

    # --------------------------------------------------------
    # RESAMPLING
    # --------------------------------------------------------

    sampled = uniform_resample(
        original,
        TARGET_POINTS
    )

    print(
        "Uniformly resampled points:",
        len(sampled)
    )

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    grid_points = grid_quantize(
        sampled,
        GRID_SIZE
    )

    print(
        "Grid points:",
        len(grid_points)
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES CREATED BY GRID
    # --------------------------------------------------------

    unique_grid = []

    for p in grid_points:

        if len(unique_grid) == 0:

            unique_grid.append(
                p
            )

        elif not np.array_equal(
            p,
            unique_grid[-1]
        ):

            unique_grid.append(
                p
            )

    grid_points = np.asarray(
        unique_grid,
        dtype=int
    )

    # --------------------------------------------------------
    # CHAIN CODE
    # --------------------------------------------------------

    cc = chain_code(
        grid_points
    )

    print(
        "Chain code length:",
        len(cc)
    )

    # --------------------------------------------------------
    # FIRST DIFFERENCE
    # --------------------------------------------------------

    fd = first_difference(
        cc
    )

    print(
        "First difference length:",
        len(fd)
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    normalized_fd = (
        normalize_first_difference(fd)
    )

    print(
        "Normalized first difference: DONE"
    )

    # --------------------------------------------------------
    # SHAPE NUMBER
    # --------------------------------------------------------

    sn = shape_number(
        normalized_fd
    )

    print(
        "Shape number: DONE"
    )

    # --------------------------------------------------------
    # BOUNDARY SIGNATURE
    # --------------------------------------------------------

    signature = boundary_signature(
        sampled
    )

    print(
        "Boundary signature: DONE"
    )

    # --------------------------------------------------------
    # STRAIGHTNESS
    # --------------------------------------------------------

    straightness = boundary_straightness(
        sampled
    )

    print(
        "Boundary straightness:",
        round(
            straightness,
            6
        )
    )

    # --------------------------------------------------------
    # BASIC RECTANGLE
    # --------------------------------------------------------

    height, width = basic_rectangle(
        sampled
    )

    print(
        "Basic rectangle:",
        round(height, 2),
        "x",
        round(width, 2)
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results[int(slice_number)] = {

        "original_boundary": original,

        "resampled_boundary": sampled,

        "grid_points": grid_points,

        "chain_code": cc,

        "first_difference": fd,

        "normalized_first_difference":
            normalized_fd,

        "shape_number": sn,

        "boundary_signature":
            signature,

        "boundary_straightness":
            straightness,

        "basic_rectangle_height":
            height,

        "basic_rectangle_width":
            width
    }

# ------------------------------------------------------------
# SAVE NUMPY RESULTS
# ------------------------------------------------------------

results_file = os.path.join(
    OUTPUT_DIR,
    "step3_final_results.npy"
)

np.save(
    results_file,
    results,
    allow_pickle=True
)

# ------------------------------------------------------------
# SAVE INFORMATION
# ------------------------------------------------------------

info_file = os.path.join(
    OUTPUT_DIR,
    "step3_final_info.txt"
)

with open(
    info_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 3 - FINAL BOUNDARY PROCESSING\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        f"Target resampling points: "
        f"{TARGET_POINTS}\n"
    )

    f.write(
        f"Grid size: {GRID_SIZE} x "
        f"{GRID_SIZE}\n\n"
    )

    f.write(
        f"Number of slices: "
        f"{len(results)}\n\n"
    )

    for s in sorted(results):

        r = results[s]

        f.write(
            f"Slice {s}\n"
        )

        f.write(
            f"Original points: "
            f"{len(r['original_boundary'])}\n"
        )

        f.write(
            f"Resampled points: "
            f"{len(r['resampled_boundary'])}\n"
        )

        f.write(
            f"Grid points: "
            f"{len(r['grid_points'])}\n"
        )

        f.write(
            f"Chain code length: "
            f"{len(r['chain_code'])}\n"
        )

        f.write(
            f"First difference length: "
            f"{len(r['first_difference'])}\n"
        )

        f.write(
            f"Straightness: "
            f"{r['boundary_straightness']}\n"
        )

        f.write("\n")

# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

visual_slices = sorted(
    results.keys()
)

if len(visual_slices) >= 3:

    selected = [

        visual_slices[0],

        visual_slices[
            len(visual_slices) // 2
        ],

        visual_slices[-1]
    ]

else:

    selected = visual_slices


for s in selected:

    r = results[s]

    original = r[
        "original_boundary"
    ]

    sampled = r[
        "resampled_boundary"
    ]

    grid = r[
        "grid_points"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    axes[0].plot(
        original[:, 1],
        original[:, 0],
        "r-",
        linewidth=1.5
    )

    axes[0].plot(
        original[:, 1],
        original[:, 0],
        "r.",
        markersize=2
    )

    axes[0].set_title(
        f"Original Boundary\n"
        f"Slice {s} | "
        f"{len(original)} points"
    )

    axes[0].set_aspect(
        "equal"
    )

    axes[0].invert_yaxis()

    # --------------------------------------------------------
    # RESAMPLED
    # --------------------------------------------------------

    axes[1].plot(
        sampled[:, 1],
        sampled[:, 0],
        "b-",
        linewidth=1.5
    )

    axes[1].plot(
        sampled[:, 1],
        sampled[:, 0],
        "bo",
        markersize=3
    )

    axes[1].set_title(
        f"Uniform Resampling\n"
        f"{len(original)} → "
        f"{len(sampled)} points"
    )

    axes[1].set_aspect(
        "equal"
    )

    axes[1].invert_yaxis()

    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    axes[2].plot(
        grid[:, 1],
        grid[:, 0],
        "ko-",
        markersize=3,
        linewidth=1
    )

    axes[2].set_xlim(
        -2,
        GRID_SIZE + 1
    )

    axes[2].set_ylim(
        -2,
        GRID_SIZE + 1
    )

    axes[2].set_aspect(
        "equal"
    )

    axes[2].invert_yaxis()

    axes[2].set_title(
        f"Grid Representation\n"
        f"{GRID_SIZE} × {GRID_SIZE}"
    )

    plt.tight_layout()

    image_file = os.path.join(
        OUTPUT_DIR,
        f"slice_{s}_step3_visual.png"
    )

    plt.savefig(
        image_file,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        image_file
    )

# ------------------------------------------------------------
# FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 COMPLETED")
print("=" * 70)

print(
    "\nUniform resampling: DONE"
)

print(
    "Grid representation: DONE"
)

print(
    "Chain code: DONE"
)

print(
    "First difference: DONE"
)

print(
    "Normalized first difference: DONE"
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

print("\nResults:")
print(results_file)

print("\nInformation:")
print(info_file)

print("\nVisualizations:")
print(OUTPUT_DIR)

print("\nNEXT STEP:")
print("FOURIER DESCRIPTORS + RECONSTRUCTION")

print("=" * 70)

input("\nPress ENTER to close...")
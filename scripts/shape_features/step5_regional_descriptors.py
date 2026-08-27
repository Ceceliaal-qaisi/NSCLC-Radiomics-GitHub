
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 5 - REGIONAL DESCRIPTORS + HU INVARIANT MOMENTS
#
# Required:
# Area
# Perimeter
# Compactness
# Circularity
# Eccentricity
# Solidity
# Seven Hu invariant moments
#
# Invariance verification:
# Translation
# Rotation
# Scaling
#
# IMPORTANT:
# Solidity = Region Area / Convex Hull Area
# Convex hull area is calculated using the actual tumor
# pixel coordinates and the shoelace formula.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PATIENT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

MASK_DIR = os.path.join(
    PATIENT_DIR,
    "GTV1_MASK"
)

MASK_FILE = os.path.join(
    MASK_DIR,
    "GTV1_binary_mask.npy"
)

OUTPUT_DIR = os.path.join(
    PATIENT_DIR,
    "REGIONAL_DESCRIPTORS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 5 - REGIONAL DESCRIPTORS")
print("         + 7 HU INVARIANT MOMENTS")
print("         + INVARIANCE VERIFICATION")
print("=" * 70)

print("\nMask:")
print(MASK_FILE)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================
# LOAD MASK
# ============================================================

print("\n" + "-" * 70)
print("Loading binary tumor mask...")
print("-" * 70)

if not os.path.exists(MASK_FILE):

    print("\nERROR:")
    print("Mask file not found!")

    print("\nExpected:")
    print(MASK_FILE)

    input("\nPress ENTER to close...")
    raise SystemExit

mask = np.load(
    MASK_FILE
)

mask = mask.astype(
    np.uint8
)

print("Mask loaded successfully.")
print("Shape:", mask.shape)
print("Tumor pixels:", int(np.sum(mask)))


# ============================================================
# FUNCTIONS
# ============================================================

def get_boundary(mask2d):
    """
    Extract boundary pixels from a binary region.

    A pixel belongs to the boundary if it is a tumor pixel
    and at least one of its 4-neighbours is background.
    """

    m = mask2d.astype(bool)

    if not np.any(m):

        return np.empty(
            (0, 2),
            dtype=int
        )

    padded = np.pad(
        m,
        1,
        mode="constant",
        constant_values=False
    )

    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    boundary = (
        m &
        ~(up & down & left & right)
    )

    points = np.argwhere(
        boundary
    )

    return points


# ============================================================
# PERIMETER
# ============================================================

def perimeter_from_mask(mask2d):
    """
    Perimeter approximation using exposed
    horizontal/vertical pixel edges.
    """

    m = mask2d.astype(bool)

    if not np.any(m):

        return 0.0

    horizontal = np.sum(
        m[:, 1:] != m[:, :-1]
    )

    vertical = np.sum(
        m[1:, :] != m[:-1, :]
    )

    border_top_bottom = (
        np.sum(m[0, :]) +
        np.sum(m[-1, :])
    )

    border_left_right = (
        np.sum(m[:, 0]) +
        np.sum(m[:, -1])
    )

    return float(
        horizontal +
        vertical +
        border_top_bottom +
        border_left_right
    )


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def connected_components(mask2d):
    """
    Simple 8-connected component labeling implemented
    from scratch.
    """

    m = mask2d.astype(bool)

    rows, cols = m.shape

    visited = np.zeros_like(
        m,
        dtype=bool
    )

    components = []

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for r in range(rows):

        for c in range(cols):

            if not m[r, c]:
                continue

            if visited[r, c]:
                continue

            stack = [(r, c)]

            visited[r, c] = True

            component = []

            while stack:

                rr, cc = stack.pop()

                component.append(
                    (rr, cc)
                )

                for dr, dc in neighbors:

                    nr = rr + dr
                    nc = cc + dc

                    if (
                        0 <= nr < rows
                        and
                        0 <= nc < cols
                        and
                        m[nr, nc]
                        and
                        not visited[nr, nc]
                    ):

                        visited[nr, nc] = True

                        stack.append(
                            (nr, nc)
                        )

            components.append(
                component
            )

    return components


# ============================================================
# LARGEST COMPONENT
# ============================================================

def largest_component(mask2d):
    """
    Keep the largest connected tumor component.
    """

    components = connected_components(
        mask2d
    )

    if not components:

        return np.zeros_like(
            mask2d,
            dtype=np.uint8
        )

    largest = max(
        components,
        key=len
    )

    output = np.zeros_like(
        mask2d,
        dtype=np.uint8
    )

    for r, c in largest:

        output[r, c] = 1

    return output


# ============================================================
# RAW MOMENTS
# ============================================================

def region_moments(mask2d):
    """
    Compute raw moments.
    """

    y, x = np.nonzero(
        mask2d
    )

    if len(x) == 0:

        return None

    moments = {}

    for p in range(4):

        for q in range(4 - p):

            moments[(p, q)] = np.sum(
                (x.astype(float) ** p) *
                (y.astype(float) ** q)
            )

    return moments


# ============================================================
# HU MOMENTS
# ============================================================

def hu_moments(mask2d):
    """
    Compute the seven Hu invariant moments
    from central normalized moments.
    """

    y, x = np.nonzero(
        mask2d
    )

    if len(x) == 0:

        return np.zeros(7)

    x = x.astype(float)
    y = y.astype(float)

    m00 = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    eta = {}

    for p in range(4):

        for q in range(4 - p):

            if p + q < 2:
                continue

            mu = np.sum(
                ((x - x_bar) ** p) *
                ((y - y_bar) ** q)
            )

            gamma = 1 + (
                (p + q) / 2
            )

            eta[(p, q)] = (
                mu /
                (m00 ** gamma)
            )

    n20 = eta[(2, 0)]
    n02 = eta[(0, 2)]
    n11 = eta[(1, 1)]
    n30 = eta[(3, 0)]
    n03 = eta[(0, 3)]
    n12 = eta[(1, 2)]
    n21 = eta[(2, 1)]

    h1 = (
        n20 + n02
    )

    h2 = (
        (n20 - n02) ** 2
        +
        4 * n11 ** 2
    )

    h3 = (
        (n30 - 3 * n12) ** 2
        +
        (3 * n21 - n03) ** 2
    )

    h4 = (
        (n30 + n12) ** 2
        +
        (n21 + n03) ** 2
    )

    h5 = (
        (n30 - 3 * n12)
        *
        (n30 + n12)
        *
        (
            (n30 + n12) ** 2
            -
            3 * (n21 + n03) ** 2
        )
        +
        (3 * n21 - n03)
        *
        (n21 + n03)
        *
        (
            3 * (n30 + n12) ** 2
            -
            (n21 + n03) ** 2
        )
    )

    h6 = (
        (n20 - n02)
        *
        (
            (n30 + n12) ** 2
            -
            (n21 + n03) ** 2
        )
        +
        4 * n11
        *
        (n30 + n12)
        *
        (n21 + n03)
    )

    h7 = (
        (3 * n21 - n03)
        *
        (n30 + n12)
        *
        (
            (n30 + n12) ** 2
            -
            3 * (n21 + n03) ** 2
        )
        -
        (n30 - 3 * n12)
        *
        (n21 + n03)
        *
        (
            3 * (n30 + n12) ** 2
            -
            (n21 + n03) ** 2
        )
    )

    return np.array([
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7
    ])


# ============================================================
# ECCENTRICITY
# ============================================================

def eccentricity(mask2d):
    """
    Eccentricity from eigenvalues of
    the covariance matrix.
    """

    y, x = np.nonzero(
        mask2d
    )

    if len(x) < 2:

        return 0.0

    x = x.astype(float)
    y = y.astype(float)

    x -= np.mean(x)
    y -= np.mean(y)

    covariance = np.cov(
        np.vstack((x, y)),
        bias=True
    )

    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    eigenvalues = np.sort(
        eigenvalues
    )

    major = eigenvalues[-1]
    minor = eigenvalues[0]

    if major <= 0:

        return 0.0

    value = np.sqrt(
        max(
            0.0,
            1 - minor / major
        )
    )

    return float(value)


# ============================================================
# CONVEX HULL
# ============================================================

def convex_hull(points):
    """
    Convex hull using the monotonic chain algorithm.

    Points are represented as:
    (row, column)
    """

    if len(points) <= 1:

        return points

    pts = sorted(
        set(
            map(
                tuple,
                points
            )
        )
    )

    def cross(o, a, b):

        return (
            (a[0] - o[0]) *
            (b[1] - o[1])
            -
            (a[1] - o[1]) *
            (b[0] - o[0])
        )

    lower = []

    for p in pts:

        while (
            len(lower) >= 2
            and
            cross(
                lower[-2],
                lower[-1],
                p
            ) <= 0
        ):

            lower.pop()

        lower.append(p)

    upper = []

    for p in reversed(pts):

        while (
            len(upper) >= 2
            and
            cross(
                upper[-2],
                upper[-1],
                p
            ) <= 0
        ):

            upper.pop()

        upper.append(p)

    hull = (
        lower[:-1] +
        upper[:-1]
    )

    return np.array(
        hull,
        dtype=float
    )


# ============================================================
# POLYGON AREA
# ============================================================

def polygon_area(points):
    """
    Polygon area using the shoelace formula.
    """

    if len(points) < 3:

        return 0.0

    x = points[:, 1]
    y = points[:, 0]

    return float(
        0.5 *
        abs(
            np.sum(
                x * np.roll(y, -1)
                -
                y * np.roll(x, -1)
            )
        )
    )


# ============================================================
# CONVEX HULL AREA
# ============================================================

def convex_hull_area(mask2d):
    """
    Calculate the area of the convex hull of the tumor.

    The convex hull is constructed from the tumor pixel
    coordinates. The shoelace formula is then used to
    calculate the hull polygon area.
    """

    y, x = np.nonzero(
        mask2d
    )

    if len(x) < 3:

        return 0.0

    points = np.column_stack(
        (
            y,
            x
        )
    ).astype(float)

    hull = convex_hull(
        points
    )

    return polygon_area(
        hull
    )


# ============================================================
# REGIONAL DESCRIPTORS
# ============================================================

def calculate_regional_descriptors(mask2d):

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    area = float(
        np.sum(mask2d)
    )

    # --------------------------------------------------------
    # PERIMETER
    # --------------------------------------------------------

    perimeter = perimeter_from_mask(
        mask2d
    )

    # --------------------------------------------------------
    # COMPACTNESS + CIRCULARITY
    # --------------------------------------------------------

    if perimeter > 0:

        compactness = (
            (perimeter ** 2)
            /
            (4 * np.pi * area)
        )

        circularity = (
            4 * np.pi * area
            /
            (perimeter ** 2)
        )

    else:

        compactness = 0.0
        circularity = 0.0

    # --------------------------------------------------------
    # ECCENTRICITY
    # --------------------------------------------------------

    ecc = eccentricity(
        mask2d
    )

    # --------------------------------------------------------
    # SOLIDITY
    #
    # Solidity = Area / Convex Hull Area
    # --------------------------------------------------------

    hull_area = convex_hull_area(
        mask2d
    )

    if hull_area > 0:

        solidity = (
            area /
            hull_area
        )

        # Numerical protection
        solidity = min(
            1.0,
            solidity
        )

    else:

        solidity = 0.0

    # --------------------------------------------------------
    # HU MOMENTS
    # --------------------------------------------------------

    hu = hu_moments(
        mask2d
    )

    return {

        "area": area,

        "perimeter": perimeter,

        "compactness": compactness,

        "circularity": circularity,

        "eccentricity": ecc,

        "solidity": solidity,

        "convex_hull_area": hull_area,

        "hu_moments": hu
    }


# ============================================================
# PROCESS ALL TUMOR SLICES
# ============================================================

print("\n" + "=" * 70)
print("PROCESSING REGIONAL DESCRIPTORS")
print("=" * 70)

regional_results = {}

tumor_slices = np.where(
    np.any(
        mask > 0,
        axis=(1, 2)
    )
)[0]

print(
    "\nTumor slices:",
    len(tumor_slices)
)

for index in tumor_slices:

    slice_number = int(index)

    print("\n" + "-" * 60)
    print(
        "Slice:",
        slice_number
    )
    print("-" * 60)

    slice_mask = mask[
        slice_number
    ]

    # Keep largest connected region
    clean_mask = largest_component(
        slice_mask
    )

    descriptors = calculate_regional_descriptors(
        clean_mask
    )

    regional_results[
        slice_number
    ] = descriptors

    print(
        f"Area: "
        f"{descriptors['area']:.3f}"
    )

    print(
        f"Perimeter: "
        f"{descriptors['perimeter']:.3f}"
    )

    print(
        f"Compactness: "
        f"{descriptors['compactness']:.6f}"
    )

    print(
        f"Circularity: "
        f"{descriptors['circularity']:.6f}"
    )

    print(
        f"Eccentricity: "
        f"{descriptors['eccentricity']:.6f}"
    )

    print(
        f"Convex Hull Area: "
        f"{descriptors['convex_hull_area']:.3f}"
    )

    print(
        f"Solidity: "
        f"{descriptors['solidity']:.6f}"
    )

    print(
        "7 Hu moments: DONE"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "regional_descriptors_results.npy"
)

np.save(
    results_file,
    regional_results,
    allow_pickle=True
)

print("\n" + "=" * 70)
print("REGIONAL RESULTS SAVED")
print("=" * 70)

print(results_file)


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "regional_descriptors_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 5 - REGIONAL DESCRIPTORS\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "Descriptors:\n"
    )

    f.write(
        "Area\n"
        "Perimeter\n"
        "Compactness\n"
        "Circularity\n"
        "Eccentricity\n"
        "Convex Hull Area\n"
        "Solidity\n"
        "Hu Moments 1-7\n\n"
    )

    f.write(
        "Solidity = Area / Convex Hull Area\n\n"
    )

    for s in sorted(
        regional_results.keys()
    ):

        d = regional_results[s]

        f.write(
            f"Slice {s}\n"
        )

        f.write(
            f"Area = {d['area']}\n"
        )

        f.write(
            f"Perimeter = {d['perimeter']}\n"
        )

        f.write(
            f"Compactness = "
            f"{d['compactness']}\n"
        )

        f.write(
            f"Circularity = "
            f"{d['circularity']}\n"
        )

        f.write(
            f"Eccentricity = "
            f"{d['eccentricity']}\n"
        )

        f.write(
            f"Convex Hull Area = "
            f"{d['convex_hull_area']}\n"
        )

        f.write(
            f"Solidity = "
            f"{d['solidity']}\n"
        )

        for i, value in enumerate(
            d["hu_moments"],
            start=1
        ):

            f.write(
                f"Hu{i} = {value:.12e}\n"
            )

        f.write("\n")

print(
    "\nSummary saved:"
)

print(
    summary_file
)


# ============================================================
# HU MOMENT INVARIANCE VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("HU MOMENT INVARIANCE VERIFICATION")
print("=" * 70)

# Select central tumor slice
reference_slice = int(
    tumor_slices[
        len(tumor_slices) // 2
    ]
)

reference_mask = largest_component(
    mask[reference_slice]
)


# ============================================================
# BOUNDING BOX
# ============================================================

ys, xs = np.nonzero(
    reference_mask
)

min_y = max(
    0,
    ys.min() - 20
)

max_y = min(
    mask.shape[1],
    ys.max() + 21
)

min_x = max(
    0,
    xs.min() - 20
)

max_x = min(
    mask.shape[2],
    xs.max() + 21
)

crop = reference_mask[
    min_y:max_y,
    min_x:max_x
]


# ============================================================
# TRANSFORMATION FUNCTIONS
# ============================================================

def translate_mask(
    original,
    dy,
    dx
):

    rows, cols = original.shape

    output = np.zeros_like(
        original
    )

    y1 = max(
        0,
        dy
    )

    y2 = min(
        rows,
        rows + dy
    )

    x1 = max(
        0,
        dx
    )

    x2 = min(
        cols,
        cols + dx
    )

    sy1 = max(
        0,
        -dy
    )

    sy2 = sy1 + (
        y2 - y1
    )

    sx1 = max(
        0,
        -dx
    )

    sx2 = sx1 + (
        x2 - x1
    )

    if (
        y2 > y1
        and
        x2 > x1
    ):

        output[
            y1:y2,
            x1:x2
        ] = original[
            sy1:sy2,
            sx1:sx2
        ]

    return output


def rotate_mask(
    original,
    angle
):

    rows, cols = original.shape

    cy = (
        rows - 1
    ) / 2

    cx = (
        cols - 1
    ) / 2

    y, x = np.nonzero(
        original
    )

    y = y - cy
    x = x - cx

    theta = np.deg2rad(
        angle
    )

    xr = (
        x * np.cos(theta)
        -
        y * np.sin(theta)
    )

    yr = (
        x * np.sin(theta)
        +
        y * np.cos(theta)
    )

    xr = np.round(
        xr + cx
    ).astype(int)

    yr = np.round(
        yr + cy
    ).astype(int)

    output = np.zeros_like(
        original
    )

    valid = (
        (yr >= 0)
        &
        (yr < rows)
        &
        (xr >= 0)
        &
        (xr < cols)
    )

    output[
        yr[valid],
        xr[valid]
    ] = 1

    return output


def scale_mask(
    original,
    factor
):

    y, x = np.nonzero(
        original
    )

    if len(x) == 0:

        return original.copy()

    cy = np.mean(y)
    cx = np.mean(x)

    ys = np.round(
        (y - cy) * factor + cy
    ).astype(int)

    xs = np.round(
        (x - cx) * factor + cx
    ).astype(int)

    output = np.zeros_like(
        original
    )

    valid = (
        (ys >= 0)
        &
        (ys < original.shape[0])
        &
        (xs >= 0)
        &
        (xs < original.shape[1])
    )

    output[
        ys[valid],
        xs[valid]
    ] = 1

    return output


# ============================================================
# ORIGINAL
# ============================================================

original_hu = hu_moments(
    crop
)

translated = translate_mask(
    crop,
    dy=8,
    dx=12
)

rotated = rotate_mask(
    crop,
    angle=30
)

scaled = scale_mask(
    crop,
    factor=1.25
)

translated_hu = hu_moments(
    translated
)

rotated_hu = hu_moments(
    rotated
)

scaled_hu = hu_moments(
    scaled
)


# ============================================================
# SAVE INVARIANCE RESULTS
# ============================================================

invariance_file = os.path.join(
    OUTPUT_DIR,
    "hu_invariance_results.txt"
)

with open(
    invariance_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "HU MOMENT INVARIANCE VERIFICATION\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        f"Reference slice: "
        f"{reference_slice}\n\n"
    )

    f.write(
        "Transformation:\n"
        "Translation: dy=8, dx=12\n"
        "Rotation: 30 degrees\n"
        "Scaling: factor=1.25\n\n"
    )

    header = (
        "Hu Moment | Original | "
        "Translation | Rotation | Scaling\n"
    )

    f.write(header)

    f.write(
        "-" * 90 +
        "\n"
    )

    for i in range(7):

        f.write(
            f"Hu{i+1} | "
            f"{original_hu[i]:.12e} | "
            f"{translated_hu[i]:.12e} | "
            f"{rotated_hu[i]:.12e} | "
            f"{scaled_hu[i]:.12e}\n"
        )

print(
    "\nReference slice:",
    reference_slice
)

print(
    "Translation test: DONE"
)

print(
    "Rotation test: DONE"
)

print(
    "Scaling test: DONE"
)

print(
    "\nInvariance results saved:"
)

print(
    invariance_file
)


# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4)
)

axes[0].imshow(
    crop,
    cmap="gray"
)

axes[0].set_title(
    "Original"
)

axes[1].imshow(
    translated,
    cmap="gray"
)

axes[1].set_title(
    "Translation"
)

axes[2].imshow(
    rotated,
    cmap="gray"
)

axes[2].set_title(
    "Rotation"
)

axes[3].imshow(
    scaled,
    cmap="gray"
)

axes[3].set_title(
    "Scaling"
)

for ax in axes:

    ax.axis("off")

fig.suptitle(
    "Hu Moment Invariance Verification"
)

plt.tight_layout()

invariance_image = os.path.join(
    OUTPUT_DIR,
    "hu_invariance_visualization.png"
)

plt.savefig(
    invariance_image,
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print(
    "\nVisualization saved:"
)

print(
    invariance_image
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nRegional descriptors: DONE"
)

print(
    "Area: DONE"
)

print(
    "Perimeter: DONE"
)

print(
    "Compactness: DONE"
)

print(
    "Circularity: DONE"
)

print(
    "Eccentricity: DONE"
)

print(
    "Convex Hull Area: DONE"
)

print(
    "Solidity: DONE"
)

print(
    "Seven Hu moments: DONE"
)

print(
    "Translation invariance test: DONE"
)

print(
    "Rotation invariance test: DONE"
)

print(
    "Scaling invariance test: DONE"
)

print(
    "\nOutput folder:"
)

print(
    OUTPUT_DIR
)

print(
    "\nNEXT STEP:"
)

print(
    "3-D SHAPE FEATURES"
)

print(
    "VOLUME + SURFACE AREA + SPHERICITY"
)

print(
    "SURFACE-TO-VOLUME RATIO"
)

print("=" * 70)

input(
    "\nPress ENTER to close..."
)


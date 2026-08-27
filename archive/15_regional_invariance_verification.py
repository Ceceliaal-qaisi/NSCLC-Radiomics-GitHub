# ============================================================
# 15_regional_invariance_verification.py
#
# REGIONAL DESCRIPTORS INVARIANCE VERIFICATION
#
# FROM-SCRATCH IMPLEMENTATION
#
# Regional descriptors:
#   Area
#   Perimeter
#   Compactness
#   Circularity
#   Effective Diameter
#   Centroid
#   Eccentricity
#   Solidity
#   Central Moments
#   Normalized Central Moments
#   Hu Moments 1-7
#
# NO OpenCV
# NO skimage
#
# IMPORTANT:
# The verification uses continuous geometric transformations.
#
# Translation:
#   invariant -> Area, Perimeter, Compactness, Circularity,
#                Effective Diameter, Eccentricity, Solidity,
#                Hu moments
#
# Rotation:
#   invariant -> same descriptors
#
# Scaling:
#   Area            -> scale^2
#   Perimeter       -> scale
#   Effective Diam. -> scale
#   Compactness     -> invariant
#   Circularity     -> invariant
#   Eccentricity    -> invariant
#   Solidity        -> invariant
#   Hu moments      -> invariant
#
# ============================================================

import os
import csv
import math

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    r"C:\Users\CeCe\Downloads\nsclc_radiomics"
    r"\LUNG1-001\69331"
)

BOUNDARY_FILE = os.path.join(
    BASE_DIR,
    "GTV1_ordered_boundary_slice74.npy"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_8_REGIONAL_MOMENT_DESCRIPTORS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

TRANSLATION_X = 200.0
TRANSLATION_Y = 150.0

ROTATION_ANGLE = 37.0

SCALE_FACTOR = 1.5


# ============================================================
# TOLERANCES
# ============================================================

RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-12


# ============================================================
# PRINT UTILITY
# ============================================================

def print_separator(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# ERROR CALCULATION
# ============================================================

def calculate_error(original, transformed):

    original = float(original)
    transformed = float(transformed)

    absolute_error = abs(
        transformed - original
    )

    denominator = max(
        abs(original),
        1e-15
    )

    relative_error = (
        absolute_error
        /
        denominator
    )

    return (
        absolute_error,
        relative_error
    )


def invariance_status(original, transformed):

    absolute_error, relative_error = (
        calculate_error(
            original,
            transformed
        )
    )

    if (
        absolute_error <= ABSOLUTE_TOLERANCE
        or
        relative_error <= RELATIVE_TOLERANCE
    ):
        return "PASS"

    return "CHECK"


# ============================================================
# LOAD BOUNDARY
# ============================================================

if not os.path.exists(
    BOUNDARY_FILE
):

    raise FileNotFoundError(
        "\nBoundary file was not found:\n"
        + BOUNDARY_FILE
    )


boundary = np.load(
    BOUNDARY_FILE
)


if (
    boundary.ndim != 2
    or
    boundary.shape[1] != 2
):

    raise ValueError(
        "Boundary must have shape (N,2). "
        f"Current shape = {boundary.shape}"
    )


boundary = boundary.astype(float)


print_separator(
    "REGIONAL DESCRIPTORS INVARIANCE VERIFICATION"
)

print(
    f"\nBoundary shape: {boundary.shape}"
)

print(
    f"Boundary points: {len(boundary)}"
)


# ============================================================
# REMOVE DUPLICATE LAST POINT
# ============================================================

if len(boundary) > 1:

    if np.allclose(
        boundary[0],
        boundary[-1]
    ):

        boundary = boundary[:-1]


# ============================================================
# POLYGON AREA
# ============================================================

def polygon_area(points):

    x = points[:, 0]
    y = points[:, 1]

    return float(
        0.5
        *
        abs(
            np.sum(
                x * np.roll(y, -1)
            )
            -
            np.sum(
                y * np.roll(x, -1)
            )
        )
    )


# ============================================================
# POLYGON PERIMETER
# ============================================================

def polygon_perimeter(points):

    differences = (
        np.roll(
            points,
            -1,
            axis=0
        )
        -
        points
    )

    lengths = np.sqrt(
        np.sum(
            differences ** 2,
            axis=1
        )
    )

    return float(
        np.sum(lengths)
    )


# ============================================================
# POLYGON CENTROID
# ============================================================

def polygon_centroid(points):

    x = points[:, 0]
    y = points[:, 1]

    x_next = np.roll(
        x,
        -1
    )

    y_next = np.roll(
        y,
        -1
    )

    cross = (
        x * y_next
        -
        x_next * y
    )

    signed_area = (
        0.5
        *
        np.sum(cross)
    )

    if abs(signed_area) < 1e-15:

        return (
            float(np.mean(x)),
            float(np.mean(y))
        )

    cx = (
        np.sum(
            (
                x + x_next
            )
            *
            cross
        )
        /
        (
            6.0
            *
            signed_area
        )
    )

    cy = (
        np.sum(
            (
                y + y_next
            )
            *
            cross
        )
        /
        (
            6.0
            *
            signed_area
        )
    )

    return (
        float(cx),
        float(cy)
    )


# ============================================================
# POINT IN POLYGON
# ============================================================

def point_in_polygon(
    x,
    y,
    polygon
):

    inside = False

    n = len(polygon)

    j = n - 1

    for i in range(n):

        xi = polygon[i, 0]
        yi = polygon[i, 1]

        xj = polygon[j, 0]
        yj = polygon[j, 1]

        if (
            (yi > y)
            !=
            (yj > y)
        ):

            x_intersection = (

                (xj - xi)
                *
                (y - yi)
                /
                (
                    (yj - yi)
                    +
                    1e-15
                )
                +
                xi
            )

            if x < x_intersection:

                inside = not inside

        j = i

    return inside


# ============================================================
# CREATE REGION POINTS
# ============================================================

def create_region_points_from_polygon(
    polygon
):

    min_x = int(
        math.floor(
            np.min(
                polygon[:, 0]
            )
        )
    )

    max_x = int(
        math.ceil(
            np.max(
                polygon[:, 0]
            )
        )
    )

    min_y = int(
        math.floor(
            np.min(
                polygon[:, 1]
            )
        )
    )

    max_y = int(
        math.ceil(
            np.max(
                polygon[:, 1]
            )
        )
    )

    points = []

    for y in range(
        min_y,
        max_y + 1
    ):

        for x in range(
            min_x,
            max_x + 1
        ):

            px = x + 0.5
            py = y + 0.5

            if point_in_polygon(
                px,
                py,
                polygon
            ):

                points.append(
                    [px, py]
                )

    return np.asarray(
        points,
        dtype=float
    )


# ============================================================
# ORIGINAL REGION
# ============================================================

print_separator(
    "CREATING ORIGINAL REGION POINTS"
)

region_points = (
    create_region_points_from_polygon(
        boundary
    )
)

print(
    f"Original region points: "
    f"{len(region_points)}"
)

if len(region_points) == 0:

    raise ValueError(
        "No region points generated."
    )


# ============================================================
# REGION CENTROID
# ============================================================

def region_centroid(points):

    if len(points) == 0:

        raise ValueError(
            "Empty region."
        )

    return np.mean(
        points,
        axis=0
    )


original_region_centroid = (
    region_centroid(
        region_points
    )
)


print(
    "\nOriginal region centroid:"
)

print(
    f"X = "
    f"{original_region_centroid[0]:.12f}"
)

print(
    f"Y = "
    f"{original_region_centroid[1]:.12f}"
)


# ============================================================
# TRANSFORMATIONS
# ============================================================

def translate_points(
    points,
    dx,
    dy
):

    result = points.copy()

    result[:, 0] += dx
    result[:, 1] += dy

    return result


def rotate_points(
    points,
    angle_degrees,
    center
):

    theta = math.radians(
        angle_degrees
    )

    c = math.cos(theta)
    s = math.sin(theta)

    center = np.asarray(
        center,
        dtype=float
    )

    centered = (
        points
        -
        center
    )

    rotation_matrix = np.array([

        [
            c,
            -s
        ],

        [
            s,
            c
        ]

    ])

    rotated = (
        centered
        @
        rotation_matrix.T
    )

    rotated += center

    return rotated


def scale_points(
    points,
    scale,
    center
):

    center = np.asarray(
        center,
        dtype=float
    )

    return (
        center
        +
        scale
        *
        (
            points
            -
            center
        )
    )


print_separator(
    "CREATING TRANSFORMATIONS"
)


# ------------------------------------------------------------
# Translation
# ------------------------------------------------------------

translation_boundary = (
    translate_points(
        boundary,
        TRANSLATION_X,
        TRANSLATION_Y
    )
)

translation_region = (
    translate_points(
        region_points,
        TRANSLATION_X,
        TRANSLATION_Y
    )
)


# ------------------------------------------------------------
# Rotation
# IMPORTANT:
# rotate around REGION centroid
# ------------------------------------------------------------

rotation_boundary = (
    rotate_points(
        boundary,
        ROTATION_ANGLE,
        original_region_centroid
    )
)

rotation_region = (
    rotate_points(
        region_points,
        ROTATION_ANGLE,
        original_region_centroid
    )
)


# ------------------------------------------------------------
# Scaling
# IMPORTANT:
# scale around REGION centroid
# ------------------------------------------------------------

scaling_boundary = (
    scale_points(
        boundary,
        SCALE_FACTOR,
        original_region_centroid
    )
)

scaling_region = (
    scale_points(
        region_points,
        SCALE_FACTOR,
        original_region_centroid
    )
)


print(
    f"Translation region points: "
    f"{len(translation_region)}"
)

print(
    f"Rotation region points: "
    f"{len(rotation_region)}"
)

print(
    f"Scaling region points: "
    f"{len(scaling_region)}"
)


# ============================================================
# CONVEX HULL
# ============================================================

def cross_product(
    o,
    a,
    b
):

    return (

        (a[0] - o[0])
        *
        (b[1] - o[1])

        -

        (a[1] - o[1])
        *
        (b[0] - o[0])
    )


def convex_hull(points):

    pts = sorted(
        set(
            (
                float(x),
                float(y)
            )
            for x, y in points
        )
    )

    if len(pts) <= 1:

        return np.asarray(
            pts,
            dtype=float
        )

    lower = []

    for p in pts:

        while (
            len(lower) >= 2
            and
            cross_product(
                lower[-2],
                lower[-1],
                p
            )
            <= 0
        ):

            lower.pop()

        lower.append(p)


    upper = []

    for p in reversed(pts):

        while (
            len(upper) >= 2
            and
            cross_product(
                upper[-2],
                upper[-1],
                p
            )
            <= 0
        ):

            upper.pop()

        upper.append(p)


    hull = (
        lower[:-1]
        +
        upper[:-1]
    )

    return np.asarray(
        hull,
        dtype=float
    )


# ============================================================
# MOMENT FUNCTIONS
# ============================================================

def raw_moment(
    points,
    p,
    q,
    weight=1.0
):

    x = points[:, 0]
    y = points[:, 1]

    return float(
        weight
        *
        np.sum(
            (x ** p)
            *
            (y ** q)
        )
    )


# ============================================================
# CENTROID FROM MOMENTS
# ============================================================

def calculate_centroid_from_region(
    points,
    weight=1.0
):

    m00 = raw_moment(
        points,
        0,
        0,
        weight
    )

    if m00 <= 0:

        raise ValueError(
            "Region has zero mass."
        )

    m10 = raw_moment(
        points,
        1,
        0,
        weight
    )

    m01 = raw_moment(
        points,
        0,
        1,
        weight
    )

    return (
        float(m10 / m00),
        float(m01 / m00)
    )


# ============================================================
# CENTRAL MOMENT
# ============================================================

def central_moment(
    points,
    p,
    q,
    weight=1.0
):

    x_bar, y_bar = (
        calculate_centroid_from_region(
            points,
            weight
        )
    )

    x = (
        points[:, 0]
        -
        x_bar
    )

    y = (
        points[:, 1]
        -
        y_bar
    )

    return float(
        weight
        *
        np.sum(
            (x ** p)
            *
            (y ** q)
        )
    )


# ============================================================
# NORMALIZED CENTRAL MOMENT
# ============================================================

def normalized_central_moment(
    points,
    p,
    q,
    weight=1.0
):

    mu00 = central_moment(
        points,
        0,
        0,
        weight
    )

    mu_pq = central_moment(
        points,
        p,
        q,
        weight
    )

    gamma = (
        (p + q)
        /
        2.0
        +
        1.0
    )

    denominator = (
        mu00 ** gamma
    )

    if abs(denominator) < 1e-30:

        return 0.0

    return float(
        mu_pq
        /
        denominator
    )


# ============================================================
# HU MOMENTS
# ============================================================

def calculate_hu_moments(
    points,
    weight=1.0
):

    eta20 = normalized_central_moment(
        points,
        2,
        0,
        weight
    )

    eta02 = normalized_central_moment(
        points,
        0,
        2,
        weight
    )

    eta11 = normalized_central_moment(
        points,
        1,
        1,
        weight
    )

    eta30 = normalized_central_moment(
        points,
        3,
        0,
        weight
    )

    eta12 = normalized_central_moment(
        points,
        1,
        2,
        weight
    )

    eta21 = normalized_central_moment(
        points,
        2,
        1,
        weight
    )

    eta03 = normalized_central_moment(
        points,
        0,
        3,
        weight
    )


    phi1 = (
        eta20
        +
        eta02
    )


    phi2 = (

        (eta20 - eta02) ** 2

        +

        4.0
        *
        eta11 ** 2
    )


    phi3 = (

        (
            eta30
            -
            3.0 * eta12
        ) ** 2

        +

        (
            3.0 * eta21
            -
            eta03
        ) ** 2
    )


    phi4 = (

        (
            eta30
            +
            eta12
        ) ** 2

        +

        (
            eta21
            +
            eta03
        ) ** 2
    )


    phi5 = (

        (
            eta30
            -
            3.0 * eta12
        )

        *

        (
            eta30
            +
            eta12
        )

        *

        (
            (
                eta30
                +
                eta12
            ) ** 2

            -

            3.0
            *
            (
                eta21
                +
                eta03
            ) ** 2
        )

        +

        (
            3.0 * eta21
            -
            eta03
        )

        *

        (
            eta21
            +
            eta03
        )

        *

        (
            3.0
            *
            (
                eta30
                +
                eta12
            ) ** 2

            -

            (
                eta21
                +
                eta03
            ) ** 2
        )
    )


    phi6 = (

        (
            eta20
            -
            eta02
        )

        *

        (
            (
                eta30
                +
                eta12
            ) ** 2

            -

            (
                eta21
                +
                eta03
            ) ** 2
        )

        +

        4.0
        *
        eta11
        *
        (
            eta30
            +
            eta12
        )
        *
        (
            eta21
            +
            eta03
        )
    )


    phi7 = (

        (
            3.0 * eta21
            -
            eta03
        )

        *

        (
            eta30
            +
            eta12
        )

        *

        (
            (
                eta30
                +
                eta12
            ) ** 2

            -

            3.0
            *
            (
                eta21
                +
                eta03
            ) ** 2
        )

        +

        (
            3.0 * eta12
            -
            eta30
        )

        *

        (
            eta21
            +
            eta03
        )

        *

        (
            3.0
            *
            (
                eta30
                +
                eta12
            ) ** 2

            -

            (
                eta21
                +
                eta03
            ) ** 2
        )
    )


    return [

        float(phi1),
        float(phi2),
        float(phi3),
        float(phi4),
        float(phi5),
        float(phi6),
        float(phi7)

    ]


# ============================================================
# ECCENTRICITY
# ============================================================

def calculate_eccentricity(
    points,
    weight=1.0
):

    mu20 = central_moment(
        points,
        2,
        0,
        weight
    )

    mu02 = central_moment(
        points,
        0,
        2,
        weight
    )

    mu11 = central_moment(
        points,
        1,
        1,
        weight
    )

    mu00 = central_moment(
        points,
        0,
        0,
        weight
    )

    if mu00 <= 0:

        return 0.0


    covariance = np.array([

        [
            mu20 / mu00,
            mu11 / mu00
        ],

        [
            mu11 / mu00,
            mu02 / mu00
        ]

    ])


    eigenvalues = np.linalg.eigvalsh(
        covariance
    )

    eigenvalues = np.sort(
        np.maximum(
            eigenvalues,
            0.0
        )
    )


    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[1]


    if lambda_max <= 1e-30:

        return 0.0


    e_squared = (

        1.0
        -
        lambda_min
        /
        lambda_max
    )

    e_squared = max(
        0.0,
        min(
            1.0,
            e_squared
        )
    )

    return float(
        math.sqrt(
            e_squared
        )
    )


# ============================================================
# DESCRIPTOR CALCULATION
# ============================================================

def calculate_descriptors(
    boundary_points,
    region_points,
    moment_weight=1.0
):

    # --------------------------------------------------------
    # GEOMETRIC AREA
    # --------------------------------------------------------

    area = polygon_area(
        boundary_points
    )


    # --------------------------------------------------------
    # PERIMETER
    # --------------------------------------------------------

    perimeter = polygon_perimeter(
        boundary_points
    )


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    centroid_x, centroid_y = (
        calculate_centroid_from_region(
            region_points,
            moment_weight
        )
    )


    # --------------------------------------------------------
    # COMPACTNESS
    #
    # P^2 / A
    # --------------------------------------------------------

    if area > 0:

        compactness = (
            perimeter ** 2
            /
            area
        )

    else:

        compactness = 0.0


    # --------------------------------------------------------
    # CIRCULARITY
    #
    # 4*pi*A/P^2
    # --------------------------------------------------------

    if perimeter > 0:

        circularity = (

            4.0
            *
            math.pi
            *
            area
            /
            perimeter ** 2
        )

    else:

        circularity = 0.0


    # --------------------------------------------------------
    # EFFECTIVE DIAMETER
    # --------------------------------------------------------

    effective_diameter = math.sqrt(

        4.0
        *
        area
        /
        math.pi
    )


    # --------------------------------------------------------
    # ECCENTRICITY
    # --------------------------------------------------------

    eccentricity = (
        calculate_eccentricity(
            region_points,
            moment_weight
        )
    )


    # --------------------------------------------------------
    # SOLIDITY
    #
    # Area / Convex Hull Area
    # --------------------------------------------------------

    hull = convex_hull(
        boundary_points
    )

    if len(hull) >= 3:

        hull_area = polygon_area(
            hull
        )

        if hull_area > 0:

            solidity = (
                area
                /
                hull_area
            )

        else:

            solidity = 0.0

    else:

        solidity = 1.0


    solidity = min(
        max(
            solidity,
            0.0
        ),
        1.0
    )


    # --------------------------------------------------------
    # HU MOMENTS
    # --------------------------------------------------------

    hu = calculate_hu_moments(
        region_points,
        moment_weight
    )


    return {

        "Area":
            float(area),

        "Perimeter":
            float(perimeter),

        "Compactness":
            float(compactness),

        "Circularity":
            float(circularity),

        "Effective_Diameter":
            float(effective_diameter),

        "Centroid_X":
            float(centroid_x),

        "Centroid_Y":
            float(centroid_y),

        "Eccentricity":
            float(eccentricity),

        "Solidity":
            float(solidity),

        "Hu1":
            hu[0],

        "Hu2":
            hu[1],

        "Hu3":
            hu[2],

        "Hu4":
            hu[3],

        "Hu5":
            hu[4],

        "Hu6":
            hu[5],

        "Hu7":
            hu[6]

    }


# ============================================================
# CALCULATE ALL DESCRIPTORS
# ============================================================

print_separator(
    "CALCULATING DESCRIPTORS"
)


# Original
original = calculate_descriptors(
    boundary,
    region_points,
    1.0
)


# Translation
translation = calculate_descriptors(
    translation_boundary,
    translation_region,
    1.0
)


# Rotation
rotation = calculate_descriptors(
    rotation_boundary,
    rotation_region,
    1.0
)


# Scaling
#
# IMPORTANT:
# Every transformed sample represents an area element.
# Under scaling by s:
#
#     dA' = s^2 dA
#
# Therefore moment weight = s^2.
# ------------------------------------------------------------

scaling_weight = (
    SCALE_FACTOR ** 2
)

scaling = calculate_descriptors(
    scaling_boundary,
    scaling_region,
    scaling_weight
)


# ============================================================
# PRINT DESCRIPTORS
# ============================================================

def print_descriptors(
    name,
    descriptors
):

    print_separator(
        name
    )

    for key, value in (
        descriptors.items()
    ):

        print(
            f"{key}: "
            f"{value:.12e}"
        )


print_descriptors(
    "ORIGINAL",
    original
)

print_descriptors(
    "TRANSLATION",
    translation
)

print_descriptors(
    "ROTATION",
    rotation
)

print_descriptors(
    "SCALING",
    scaling
)


# ============================================================
# CENTRAL MOMENTS
# ============================================================

print_separator(
    "CENTRAL MOMENTS - ORIGINAL"
)


central_moment_names = [

    ("mu02", 0, 2),
    ("mu03", 0, 3),
    ("mu11", 1, 1),
    ("mu12", 1, 2),
    ("mu20", 2, 0),
    ("mu21", 2, 1),
    ("mu30", 3, 0)

]


original_central_moments = {}


for name, p, q in (
    central_moment_names
):

    value = central_moment(
        region_points,
        p,
        q,
        1.0
    )

    original_central_moments[
        name
    ] = value

    print(
        f"{name}: "
        f"{value:.12e}"
    )


# ============================================================
# NORMALIZED CENTRAL MOMENTS
# ============================================================

print_separator(
    "NORMALIZED CENTRAL MOMENTS - ORIGINAL"
)


original_normalized_moments = {}


for name, p, q in (
    central_moment_names
):

    value = normalized_central_moment(
        region_points,
        p,
        q,
        1.0
    )

    normalized_name = (
        name.replace(
            "mu",
            "eta"
        )
    )

    original_normalized_moments[
        normalized_name
    ] = value

    print(
        f"{normalized_name}: "
        f"{value:.12e}"
    )


# ============================================================
# INVARIANCE COMPARISON
# ============================================================

print_separator(
    "INVARIANCE COMPARISON"
)


invariant_features = [

    "Compactness",
    "Circularity",
    "Eccentricity",
    "Solidity",

    "Hu1",
    "Hu2",
    "Hu3",
    "Hu4",
    "Hu5",
    "Hu6",
    "Hu7"

]


csv_rows = []


for feature in invariant_features:

    original_value = original[
        feature
    ]

    translation_value = translation[
        feature
    ]

    rotation_value = rotation[
        feature
    ]

    scaling_value = scaling[
        feature
    ]


    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    (
        translation_abs_error,
        translation_rel_error
    ) = calculate_error(
        original_value,
        translation_value
    )

    translation_status = invariance_status(
        original_value,
        translation_value
    )


    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    (
        rotation_abs_error,
        rotation_rel_error
    ) = calculate_error(
        original_value,
        rotation_value
    )

    rotation_status = invariance_status(
        original_value,
        rotation_value
    )


    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    (
        scaling_abs_error,
        scaling_rel_error
    ) = calculate_error(
        original_value,
        scaling_value
    )

    scaling_status = invariance_status(
        original_value,
        scaling_value
    )


    print(
        "\n" + feature
    )


    print(

        f"  Translation = "
        f"{translation_value:.12e} | "

        f"Abs Error = "
        f"{translation_abs_error:.3e} | "

        f"Rel Error = "
        f"{translation_rel_error:.3e} | "

        f"{translation_status}"
    )


    print(

        f"  Rotation    = "
        f"{rotation_value:.12e} | "

        f"Abs Error = "
        f"{rotation_abs_error:.3e} | "

        f"Rel Error = "
        f"{rotation_rel_error:.3e} | "

        f"{rotation_status}"
    )


    print(

        f"  Scaling     = "
        f"{scaling_value:.12e} | "

        f"Abs Error = "
        f"{scaling_abs_error:.3e} | "

        f"Rel Error = "
        f"{scaling_rel_error:.3e} | "

        f"{scaling_status}"
    )


    csv_rows.append([

        feature,

        original_value,

        translation_value,
        translation_abs_error,
        translation_rel_error,
        translation_status,

        rotation_value,
        rotation_abs_error,
        rotation_rel_error,
        rotation_status,

        scaling_value,
        scaling_abs_error,
        scaling_rel_error,
        scaling_status

    ])


# ============================================================
# EXPECTED BEHAVIOR
# ============================================================

print_separator(
    "EXPECTED TRANSFORMATION BEHAVIOR"
)


print(
"""
AREA
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> scale^2

PERIMETER
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> scale

EFFECTIVE DIAMETER
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> scale

CENTROID
    Translation -> changes
    Rotation around centroid -> invariant
    Scaling around centroid  -> invariant

COMPACTNESS
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> invariant

CIRCULARITY
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> invariant

ECCENTRICITY
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> invariant

SOLIDITY
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> invariant

HU MOMENTS
    Translation -> invariant
    Rotation    -> invariant
    Scaling     -> invariant
"""
)


# ============================================================
# GEOMETRIC TRANSFORMATION CHECKS
# ============================================================

print_separator(
    "GEOMETRIC TRANSFORMATION CHECKS"
)


# ------------------------------------------------------------
# AREA
# ------------------------------------------------------------

original_polygon_area = polygon_area(
    boundary
)

scaled_polygon_area = polygon_area(
    scaling_boundary
)

expected_scaled_polygon_area = (

    original_polygon_area
    *
    SCALE_FACTOR ** 2
)


(
    area_abs_error,
    area_rel_error
) = calculate_error(
    expected_scaled_polygon_area,
    scaled_polygon_area
)


area_status = (

    "PASS"
    if area_rel_error
    <= RELATIVE_TOLERANCE
    else
    "CHECK"
)


print(
    f"\nOriginal polygon Area      = "
    f"{original_polygon_area:.12e}"
)

print(
    f"Expected scaled Area       = "
    f"{expected_scaled_polygon_area:.12e}"
)

print(
    f"Actual scaled Area         = "
    f"{scaled_polygon_area:.12e}"
)

print(
    f"Area relative error       = "
    f"{area_rel_error:.3e}"
)

print(
    f"Area check                = "
    f"{area_status}"
)


# ------------------------------------------------------------
# PERIMETER
# ------------------------------------------------------------

original_perimeter = polygon_perimeter(
    boundary
)

scaled_perimeter = polygon_perimeter(
    scaling_boundary
)

expected_scaled_perimeter = (

    original_perimeter
    *
    SCALE_FACTOR
)


(
    perimeter_abs_error,
    perimeter_rel_error
) = calculate_error(
    expected_scaled_perimeter,
    scaled_perimeter
)


perimeter_status = (

    "PASS"
    if perimeter_rel_error
    <= RELATIVE_TOLERANCE
    else
    "CHECK"
)


print(
    f"\nOriginal Perimeter         = "
    f"{original_perimeter:.12e}"
)

print(
    f"Expected scaled Perimeter = "
    f"{expected_scaled_perimeter:.12e}"
)

print(
    f"Actual scaled Perimeter   = "
    f"{scaled_perimeter:.12e}"
)

print(
    f"Perimeter relative error  = "
    f"{perimeter_rel_error:.3e}"
)

print(
    f"Perimeter check           = "
    f"{perimeter_status}"
)


# ------------------------------------------------------------
# EFFECTIVE DIAMETER
# ------------------------------------------------------------

original_diameter = math.sqrt(

    4.0
    *
    original_polygon_area
    /
    math.pi
)

scaled_diameter = math.sqrt(

    4.0
    *
    scaled_polygon_area
    /
    math.pi
)

expected_scaled_diameter = (

    original_diameter
    *
    SCALE_FACTOR
)


(
    diameter_abs_error,
    diameter_rel_error
) = calculate_error(
    expected_scaled_diameter,
    scaled_diameter
)


diameter_status = (

    "PASS"
    if diameter_rel_error
    <= RELATIVE_TOLERANCE
    else
    "CHECK"
)


print(
    f"\nOriginal Diameter         = "
    f"{original_diameter:.12e}"
)

print(
    f"Expected scaled Diameter = "
    f"{expected_scaled_diameter:.12e}"
)

print(
    f"Actual scaled Diameter   = "
    f"{scaled_diameter:.12e}"
)

print(
    f"Diameter relative error  = "
    f"{diameter_rel_error:.3e}"
)

print(
    f"Diameter check           = "
    f"{diameter_status}"
)


# ============================================================
# SAVE CSV
# ============================================================

csv_file = os.path.join(

    OUTPUT_DIR,

    "regional_invariance_verification.csv"
)


with open(

    csv_file,
    "w",
    newline="",
    encoding="utf-8"

) as f:

    writer = csv.writer(f)

    writer.writerow([

        "Feature",

        "Original",

        "Translation",
        "Translation_Absolute_Error",
        "Translation_Relative_Error",
        "Translation_Status",

        "Rotation",
        "Rotation_Absolute_Error",
        "Rotation_Relative_Error",
        "Rotation_Status",

        "Scaling",
        "Scaling_Absolute_Error",
        "Scaling_Relative_Error",
        "Scaling_Status"

    ])

    writer.writerows(
        csv_rows
    )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_file = os.path.join(

    OUTPUT_DIR,

    "regional_invariance_verification_report.txt"
)


with open(

    report_file,
    "w",
    encoding="utf-8"

) as f:

    f.write(
        "REGIONAL DESCRIPTORS "
        "INVARIANCE VERIFICATION\n"
    )

    f.write(
        "=" * 80
        +
        "\n\n"
    )

    f.write(
        "Boundary file:\n"
    )

    f.write(
        BOUNDARY_FILE
        +
        "\n\n"
    )

    f.write(
        f"Boundary points: "
        f"{len(boundary)}\n"
    )

    f.write(
        f"Region points: "
        f"{len(region_points)}\n\n"
    )

    f.write(
        f"Translation: "
        f"({TRANSLATION_X}, "
        f"{TRANSLATION_Y})\n"
    )

    f.write(
        f"Rotation angle: "
        f"{ROTATION_ANGLE}\n"
    )

    f.write(
        f"Scaling factor: "
        f"{SCALE_FACTOR}\n\n"
    )

    f.write(
        "Hu Moments follow the "
        "Gonzalez & Woods "
        "moment-invariant equations.\n\n"
    )

    for feature in invariant_features:

        f.write(
            f"{feature}\n"
        )

        f.write(
            f"  Original    = "
            f"{original[feature]:.12e}\n"
        )

        f.write(
            f"  Translation = "
            f"{translation[feature]:.12e}\n"
        )

        f.write(
            f"  Rotation    = "
            f"{rotation[feature]:.12e}\n"
        )

        f.write(
            f"  Scaling     = "
            f"{scaling[feature]:.12e}\n\n"
        )


# ============================================================
# SAVE TRANSFORMATION IMAGE
# ============================================================

print_separator(
    "CREATING TRANSFORMATION IMAGES"
)


def save_transformation_image(
    boundary_points,
    title,
    filename
):

    plt.figure(
        figsize=(7, 7)
    )

    closed = np.vstack([

        boundary_points,

        boundary_points[0]

    ])

    plt.plot(

        closed[:, 0],
        closed[:, 1],

        linewidth=1.5
    )

    plt.scatter(

        boundary_points[:, 0],
        boundary_points[:, 1],

        s=5
    )

    plt.gca().set_aspect(
        "equal",
        adjustable="box"
    )

    plt.gca().invert_yaxis()

    plt.title(
        title
    )

    plt.xlabel(
        "X"
    )

    plt.ylabel(
        "Y"
    )

    plt.tight_layout()

    output_path = os.path.join(

        OUTPUT_DIR,
        filename
    )

    plt.savefig(

        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


save_transformation_image(

    boundary,

    "Original GTV-1 Boundary",

    "01_original.png"
)


save_transformation_image(

    translation_boundary,

    "Translated GTV-1 Boundary",

    "02_translation.png"
)


save_transformation_image(

    rotation_boundary,

    f"Rotated GTV-1 Boundary "
    f"({ROTATION_ANGLE} degrees)",

    "03_rotation.png"
)


save_transformation_image(

    scaling_boundary,

    f"Scaled GTV-1 Boundary "
    f"(factor = {SCALE_FACTOR})",

    "04_scaling.png"
)


# ============================================================
# ALL TRANSFORMATIONS
# ============================================================

plt.figure(
    figsize=(10, 8)
)


original_closed = np.vstack([
    boundary,
    boundary[0]
])

translation_closed = np.vstack([
    translation_boundary,
    translation_boundary[0]
])

rotation_closed = np.vstack([
    rotation_boundary,
    rotation_boundary[0]
])

scaling_closed = np.vstack([
    scaling_boundary,
    scaling_boundary[0]
])


plt.plot(

    original_closed[:, 0],
    original_closed[:, 1],

    label="Original"
)


plt.plot(

    translation_closed[:, 0],
    translation_closed[:, 1],

    label="Translation"
)


plt.plot(

    rotation_closed[:, 0],
    rotation_closed[:, 1],

    label="Rotation"
)


plt.plot(

    scaling_closed[:, 0],
    scaling_closed[:, 1],

    label="Scaling"
)


plt.gca().set_aspect(
    "equal",
    adjustable="box"
)

plt.gca().invert_yaxis()

plt.title(
    "Regional Descriptor Transformation Verification"
)

plt.xlabel(
    "X"
)

plt.ylabel(
    "Y"
)

plt.legend()

plt.tight_layout()


all_transformations_file = os.path.join(

    OUTPUT_DIR,

    "05_all_transformations.png"
)


plt.savefig(

    all_transformations_file,

    dpi=200,

    bbox_inches="tight"
)

plt.close()


print(
    f"Saved: "
    f"{all_transformations_file}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_separator(
    "INVARIANCE VERIFICATION COMPLETE"
)


print(
    "\nImages created:"
)

print(
    "01_original.png"
)

print(
    "02_translation.png"
)

print(
    "03_rotation.png"
)

print(
    "04_scaling.png"
)

print(
    "05_all_transformations.png"
)

print(
    "\nCSV:"
)

print(
    csv_file
)

print(
    "\nReport:"
)

print(
    report_file
)

print(
    "\nAll files saved in:"
)

print(
    OUTPUT_DIR
)

print(
    "\nDONE."
)
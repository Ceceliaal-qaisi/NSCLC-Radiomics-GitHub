# ================================================================
# STEP 8 - REGIONAL AND MOMENT DESCRIPTORS
# ================================================================
# Based on:
# Digital Image Processing, 4th Edition
# Rafael C. Gonzalez & Richard E. Woods
#
# Region Feature Descriptors:
# Area
# Perimeter
# Compactness
# Circularity
# Effective Diameter
# Centroid
# Major Axis
# Minor Axis
# Eccentricity
# Solidity
# Central Moments
# Normalized Central Moments
# Seven Hu Invariant Moments
#
# IMPORTANT:
# Eccentricity is NOT Major Axis / Minor Axis.
# For a REGION, the book defines eccentricity using
# an ellipse having the same second central moments.
# Therefore covariance matrix + eigenvalues are used.
# ================================================================

import os
import numpy as np
import math
import csv
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

BOUNDARY_FILE = os.path.join(
    BASE_DIR,
    "GTV1_ordered_boundary_slice74.npy"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_8_REGIONAL_MOMENT_DESCRIPTORS"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# POLYGON TO BINARY MASK
# ================================================================

def polygon_to_mask(boundary):

    boundary = np.asarray(boundary, dtype=float)

    min_x = int(np.floor(np.min(boundary[:, 0])))
    max_x = int(np.ceil(np.max(boundary[:, 0])))

    min_y = int(np.floor(np.min(boundary[:, 1])))
    max_y = int(np.ceil(np.max(boundary[:, 1])))

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    x = boundary[:, 0] - min_x
    y = boundary[:, 1] - min_y

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    # ------------------------------------------------------------
    # Scanline polygon filling
    # ------------------------------------------------------------

    for row in range(height):

        yy_scan = row + 0.5

        intersections = []

        for i in range(len(x)):

            j = (i + 1) % len(x)

            x1 = x[i]
            y1 = y[i]

            x2 = x[j]
            y2 = y[j]

            if (y1 <= yy_scan < y2) or (
                y2 <= yy_scan < y1
            ):

                if y2 != y1:

                    xx = (
                        x1
                        + (yy_scan - y1)
                        * (x2 - x1)
                        / (y2 - y1)
                    )

                    intersections.append(xx)

        intersections.sort()

        for k in range(
            0,
            len(intersections) - 1,
            2
        ):

            x_start = int(
                math.ceil(intersections[k])
            )

            x_end = int(
                math.floor(intersections[k + 1])
            )

            if x_end >= x_start:

                x_start = max(
                    0,
                    x_start
                )

                x_end = min(
                    width - 1,
                    x_end
                )

                mask[
                    row,
                    x_start:x_end + 1
                ] = 1

    return mask, min_x, min_y


# ================================================================
# CONVEX HULL FROM SCRATCH
# ================================================================

def cross(o, a, b):

    return (
        (a[0] - o[0])
        * (b[1] - o[1])
        -
        (a[1] - o[1])
        * (b[0] - o[0])
    )


def convex_hull(points):

    points = sorted(
        set(
            map(
                tuple,
                points
            )
        )
    )

    if len(points) <= 1:

        return np.array(
            points,
            dtype=float
        )

    lower = []

    for p in points:

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

    for p in reversed(points):

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
        lower[:-1]
        +
        upper[:-1]
    )

    return np.array(
        hull,
        dtype=float
    )


def polygon_area(points):

    if len(points) < 3:

        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * abs(
        np.sum(
            x * np.roll(y, -1)
            -
            y * np.roll(x, -1)
        )
    )


# ================================================================
# STEP 1 - LOAD ORDERED BOUNDARY
# ================================================================

print("=" * 80)
print("STEP 1 - LOADING ORDERED GTV-1 BOUNDARY")
print("=" * 80)

boundary = np.load(
    BOUNDARY_FILE
)

boundary = np.asarray(
    boundary,
    dtype=float
)

print(
    "Boundary shape:",
    boundary.shape
)

print(
    "Boundary points:",
    len(boundary)
)


# ================================================================
# STEP 2 - CREATE BINARY REGION
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 2 - CREATING BINARY REGION "
    "FROM GTV-1 BOUNDARY"
)
print("=" * 80)

mask, offset_x, offset_y = polygon_to_mask(
    boundary
)

print(
    "Binary mask shape:",
    mask.shape
)

print(
    "Tumor pixels:",
    int(np.sum(mask))
)


# ================================================================
# STEP 3 - REGION PIXELS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 3 - EXTRACTING REGION PIXELS"
)
print("=" * 80)

local_y, local_x = np.where(
    mask == 1
)

global_x = (
    local_x
    + offset_x
)

global_y = (
    local_y
    + offset_y
)

N = len(global_x)

print(
    "Number of region pixels:",
    N
)


# ================================================================
# STEP 4 - AREA
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 4 - CALCULATING AREA"
)
print("=" * 80)

# Book:
# Area = number of pixels in the region

A = float(N)

print(
    "Area:",
    A,
    "pixels"
)


# ================================================================
# STEP 5 - PERIMETER
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 5 - CALCULATING PERIMETER"
)
print("=" * 80)

dx_boundary = np.diff(
    np.append(
        boundary[:, 0],
        boundary[0, 0]
    )
)

dy_boundary = np.diff(
    np.append(
        boundary[:, 1],
        boundary[0, 1]
    )
)

P = float(
    np.sum(
        np.sqrt(
            dx_boundary ** 2
            +
            dy_boundary ** 2
        )
    )
)

print(
    "Perimeter:",
    P,
    "pixels"
)


# ================================================================
# STEP 6 - COMPACTNESS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 6 - CALCULATING COMPACTNESS"
)
print("=" * 80)

# Book Eq. 11-18:
#
# Compactness = P^2 / A

compactness = (
    P ** 2
) / A

print(
    "Compactness:",
    compactness
)


# ================================================================
# STEP 7 - CIRCULARITY
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 7 - CALCULATING CIRCULARITY"
)
print("=" * 80)

# Book Eq. 11-19:
#
# Circularity = 4*pi*A / P^2

circularity = (
    4.0
    * math.pi
    * A
) / (
    P ** 2
)

print(
    "Circularity:",
    circularity
)


# ================================================================
# STEP 8 - EFFECTIVE DIAMETER
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 8 - CALCULATING EFFECTIVE DIAMETER"
)
print("=" * 80)

# Book Eq. 11-20:
#
# de = 2*sqrt(A/pi)

effective_diameter = (
    2.0
    * math.sqrt(
        A / math.pi
    )
)

print(
    "Effective diameter:",
    effective_diameter
)


# ================================================================
# STEP 9 - CENTROID
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 9 - CALCULATING CENTROID"
)
print("=" * 80)

x_bar = float(
    np.mean(global_x)
)

y_bar = float(
    np.mean(global_y)
)

print(
    "Centroid x:",
    x_bar
)

print(
    "Centroid y:",
    y_bar
)


# ================================================================
# STEP 10 - CENTRAL MOMENTS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 10 - CALCULATING CENTRAL MOMENTS"
)
print("=" * 80)

xc = (
    global_x
    - x_bar
)

yc = (
    global_y
    - y_bar
)


def central_moment(p, q):

    return float(
        np.sum(
            (xc ** p)
            *
            (yc ** q)
        )
    )


mu20 = central_moment(
    2,
    0
)

mu02 = central_moment(
    0,
    2
)

mu11 = central_moment(
    1,
    1
)

mu30 = central_moment(
    3,
    0
)

mu03 = central_moment(
    0,
    3
)

mu12 = central_moment(
    1,
    2
)

mu21 = central_moment(
    2,
    1
)

print(
    "mu20:",
    mu20
)

print(
    "mu02:",
    mu02
)

print(
    "mu11:",
    mu11
)

print(
    "mu30:",
    mu30
)

print(
    "mu03:",
    mu03
)

print(
    "mu12:",
    mu12
)

print(
    "mu21:",
    mu21
)


# ================================================================
# STEP 11 - NORMALIZED CENTRAL MOMENTS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 11 - CALCULATING "
    "NORMALIZED CENTRAL MOMENTS"
)
print("=" * 80)

# Book:
#
# eta_pq =
# mu_pq / mu_00^(1+(p+q)/2)
#
# mu_00 = A

mu00 = A


def normalized_moment(
    mu,
    p,
    q
):

    denominator = (
        mu00
        **
        (
            1.0
            +
            (
                (p + q)
                / 2.0
            )
        )
    )

    return (
        mu
        /
        denominator
    )


eta20 = normalized_moment(
    mu20,
    2,
    0
)

eta02 = normalized_moment(
    mu02,
    0,
    2
)

eta11 = normalized_moment(
    mu11,
    1,
    1
)

eta30 = normalized_moment(
    mu30,
    3,
    0
)

eta03 = normalized_moment(
    mu03,
    0,
    3
)

eta12 = normalized_moment(
    mu12,
    1,
    2
)

eta21 = normalized_moment(
    mu21,
    2,
    1
)

print(
    "eta20:",
    eta20
)

print(
    "eta02:",
    eta02
)

print(
    "eta11:",
    eta11
)

print(
    "eta30:",
    eta30
)

print(
    "eta03:",
    eta03
)

print(
    "eta12:",
    eta12
)

print(
    "eta21:",
    eta21
)


# ================================================================
# STEP 12 - HU MOMENT INVARIANTS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 12 - CALCULATING "
    "7 HU MOMENT INVARIANTS"
)
print("=" * 80)

h1 = (
    eta20
    +
    eta02
)

h2 = (
    (eta20 - eta02) ** 2
    +
    4.0 * eta11 ** 2
)

h3 = (
    (eta30 - 3.0 * eta12) ** 2
    +
    (3.0 * eta21 - eta03) ** 2
)

h4 = (
    (eta30 + eta12) ** 2
    +
    (eta21 + eta03) ** 2
)

h5 = (
    (eta30 - 3.0 * eta12)
    *
    (eta30 + eta12)
    *
    (
        (eta30 + eta12) ** 2
        -
        3.0
        *
        (eta21 + eta03) ** 2
    )
    +
    (3.0 * eta21 - eta03)
    *
    (eta21 + eta03)
    *
    (
        3.0
        *
        (eta30 + eta12) ** 2
        -
        (eta21 + eta03) ** 2
    )
)

h6 = (
    (eta20 - eta02)
    *
    (
        (eta30 + eta12) ** 2
        -
        (eta21 + eta03) ** 2
    )
    +
    4.0
    *
    eta11
    *
    (eta30 + eta12)
    *
    (eta21 + eta03)
)

h7 = (
    (3.0 * eta21 - eta03)
    *
    (eta30 + eta12)
    *
    (
        (eta30 + eta12) ** 2
        -
        3.0
        *
        (eta21 + eta03) ** 2
    )
    -
    (eta30 - 3.0 * eta12)
    *
    (eta21 + eta03)
    *
    (
        3.0
        *
        (eta30 + eta12) ** 2
        -
        (eta21 + eta03) ** 2
    )
)

hu = [
    h1,
    h2,
    h3,
    h4,
    h5,
    h6,
    h7
]

for i, value in enumerate(
    hu,
    1
):

    print(
        f"Hu{i}: {value:.15e}"
    )


# ================================================================
# STEP 13 - BOOK MAJOR AXIS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 13 - CALCULATING BOOK MAJOR AXIS"
)
print("=" * 80)

# Major axis:
# maximum Euclidean distance between
# two boundary points.

max_distance = -1.0

major_p1 = None
major_p2 = None

for i in range(
    len(boundary)
):

    for j in range(
        i + 1,
        len(boundary)
    ):

        d = math.sqrt(
            (
                boundary[j, 0]
                -
                boundary[i, 0]
            ) ** 2
            +
            (
                boundary[j, 1]
                -
                boundary[i, 1]
            ) ** 2
        )

        if d > max_distance:

            max_distance = d

            major_p1 = (
                boundary[i].copy()
            )

            major_p2 = (
                boundary[j].copy()
            )

major_axis = max_distance

print(
    "Major axis length:",
    major_axis
)

print(
    "Major axis point 1:",
    major_p1
)

print(
    "Major axis point 2:",
    major_p2
)


# ================================================================
# STEP 14 - BOOK MINOR AXIS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 14 - CALCULATING BOOK MINOR AXIS"
)
print("=" * 80)

# Major-axis unit vector

major_vector = (
    major_p2
    -
    major_p1
)

major_vector = (
    major_vector
    /
    np.linalg.norm(
        major_vector
    )
)

# Perpendicular vector

minor_vector = np.array(
    [
        -major_vector[1],
        major_vector[0]
    ],
    dtype=float
)

# Project boundary points on perpendicular direction

relative_boundary = (
    boundary
    -
    np.array(
        [
            x_bar,
            y_bar
        ]
    )
)

minor_projection = (
    relative_boundary
    @
    minor_vector
)

minor_min = float(
    np.min(
        minor_projection
    )
)

minor_max = float(
    np.max(
        minor_projection
    )
)

minor_axis = (
    minor_max
    -
    minor_min
)

print(
    "Minor axis length:",
    minor_axis
)


# ================================================================
# STEP 15 - REGIONAL ECCENTRICITY
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 15 - CALCULATING "
    "BOOK REGIONAL ECCENTRICITY"
)
print("=" * 80)

# ================================================================
# BOOK DEFINITION:
#
# Eccentricity of a region is the eccentricity
# of an ellipse having the same second central
# moments as the region.
#
# Covariance matrix:
#
# C = [ mu20/A    mu11/A ]
#     [ mu11/A    mu02/A ]
#
# Principal axes are eigenvectors of C.
# Eigenvalues correspond to the principal
# variances.
# ================================================================

covariance_matrix = np.array(
    [
        [
            mu20 / A,
            mu11 / A
        ],
        [
            mu11 / A,
            mu02 / A
        ]
    ],
    dtype=float
)

print(
    "Covariance matrix:"
)

print(
    covariance_matrix
)


# Eigenvalues and eigenvectors

eigenvalues, eigenvectors = np.linalg.eigh(
    covariance_matrix
)

# Sort from largest to smallest

order = np.argsort(
    eigenvalues
)[::-1]

eigenvalues = (
    eigenvalues[order]
)

eigenvectors = (
    eigenvectors[:, order]
)

lambda_major = float(
    eigenvalues[0]
)

lambda_minor = float(
    eigenvalues[1]
)

major_eigenvector = (
    eigenvectors[:, 0]
)

minor_eigenvector = (
    eigenvectors[:, 1]
)

# Book:
#
# e = sqrt(1 - (b/a)^2)
#
# Since a/b follows sqrt(lambda_major/lambda_minor):
#
# e = sqrt(1 - lambda_minor/lambda_major)

if lambda_major > 0:

    eccentricity = math.sqrt(
        max(
            0.0,
            1.0
            -
            (
                lambda_minor
                /
                lambda_major
            )
        )
    )

else:

    eccentricity = 0.0


print(
    "Major eigenvalue:",
    lambda_major
)

print(
    "Minor eigenvalue:",
    lambda_minor
)

print(
    "Major eigenvector:",
    major_eigenvector
)

print(
    "Minor eigenvector:",
    minor_eigenvector
)

print(
    "Book regional eccentricity:",
    eccentricity
)


# ================================================================
# PRINCIPAL AXIS ANGLE
# ================================================================

major_angle = math.degrees(
    math.atan2(
        major_eigenvector[1],
        major_eigenvector[0]
    )
)

print(
    "Principal major-axis angle:",
    major_angle,
    "degrees"
)


# ================================================================
# DISPLAY AXES
# ================================================================

# ------------------------------------------------
# IMPORTANT:
# Numerical Major/Minor axis descriptors remain
# based on the book boundary definitions.
#
# For visualization, both displayed axes are
# centered exactly on the centroid.
# ------------------------------------------------

centroid_global = np.array(
    [
        x_bar,
        y_bar
    ],
    dtype=float
)

# Major axis direction for display

display_major_vector = (
    major_vector
)

# Minor direction perpendicular to major

display_minor_vector = (
    minor_vector
)

# ------------------------------------------------
# Find intersection of a line through centroid
# with the boundary.
# This prevents the DISPLAYED line from extending
# outside the tumor.
# ------------------------------------------------

def line_boundary_intersections(
    center,
    direction,
    boundary_points
):

    direction = (
        direction
        /
        np.linalg.norm(
            direction
        )
    )

    intersections = []

    for i in range(
        len(boundary_points)
    ):

        j = (
            i + 1
        ) % len(
            boundary_points
        )

        p = boundary_points[i]
        q = boundary_points[j]

        edge = q - p

        denominator = (
            direction[0]
            * (-edge[1])
            -
            direction[1]
            * (-edge[0])
        )

        if abs(
            denominator
        ) < 1e-12:

            continue

        diff = p - center

        t = (
            diff[0]
            * (-edge[1])
            -
            diff[1]
            * (-edge[0])
        ) / denominator

        s = (
            direction[0]
            * diff[1]
            -
            direction[1]
            * diff[0]
        ) / denominator

        if (
            0.0
            <= s
            <= 1.0
        ):

            point = (
                center
                +
                t
                * direction
            )

            intersections.append(
                (
                    float(t),
                    point
                )
            )

    return intersections


# Major display intersections

major_intersections = (
    line_boundary_intersections(
        centroid_global,
        display_major_vector,
        boundary
    )
)

if len(
    major_intersections
) >= 2:

    major_intersections.sort(
        key=lambda z: z[0]
    )

    major_display_p1 = (
        major_intersections[0][1]
    )

    major_display_p2 = (
        major_intersections[-1][1]
    )

else:

    major_display_p1 = (
        centroid_global
        -
        display_major_vector
        *
        major_axis
        /
        2.0
    )

    major_display_p2 = (
        centroid_global
        +
        display_major_vector
        *
        major_axis
        /
        2.0
    )


# Minor display intersections

minor_intersections = (
    line_boundary_intersections(
        centroid_global,
        display_minor_vector,
        boundary
    )
)

if len(
    minor_intersections
) >= 2:

    minor_intersections.sort(
        key=lambda z: z[0]
    )

    minor_display_p1 = (
        minor_intersections[0][1]
    )

    minor_display_p2 = (
        minor_intersections[-1][1]
    )

else:

    minor_display_p1 = (
        centroid_global
        -
        display_minor_vector
        *
        minor_axis
        /
        2.0
    )

    minor_display_p2 = (
        centroid_global
        +
        display_minor_vector
        *
        minor_axis
        /
        2.0
    )


print(
    "Major display point 1:",
    major_display_p1
)

print(
    "Major display point 2:",
    major_display_p2
)

print(
    "Minor display point 1:",
    minor_display_p1
)

print(
    "Minor display point 2:",
    minor_display_p2
)


# ================================================================
# STEP 16 - SOLIDITY
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 16 - CALCULATING SOLIDITY"
)
print("=" * 80)

region_points = np.column_stack(
    (
        global_x,
        global_y
    )
)

hull = convex_hull(
    region_points
)

hull_area = polygon_area(
    hull
)

solidity = (
    A
    /
    hull_area
)

print(
    "Convex hull area:",
    hull_area
)

print(
    "Solidity:",
    solidity
)


# ================================================================
# STEP 17 - SAVE NUMERICAL RESULTS
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 17 - SAVING NUMERICAL RESULTS"
)
print("=" * 80)

npz_file = os.path.join(
    OUTPUT_DIR,
    "GTV1_regional_moment_descriptors.npz"
)

np.savez(
    npz_file,

    area=A,
    perimeter=P,
    compactness=compactness,
    circularity=circularity,
    effective_diameter=effective_diameter,

    centroid_x=x_bar,
    centroid_y=y_bar,

    mu20=mu20,
    mu02=mu02,
    mu11=mu11,
    mu30=mu30,
    mu03=mu03,
    mu12=mu12,
    mu21=mu21,

    eta20=eta20,
    eta02=eta02,
    eta11=eta11,
    eta30=eta30,
    eta03=eta03,
    eta12=eta12,
    eta21=eta21,

    Hu1=h1,
    Hu2=h2,
    Hu3=h3,
    Hu4=h4,
    Hu5=h5,
    Hu6=h6,
    Hu7=h7,

    major_axis=major_axis,
    minor_axis=minor_axis,

    covariance_matrix=covariance_matrix,

    major_eigenvalue=lambda_major,
    minor_eigenvalue=lambda_minor,

    eccentricity=eccentricity,

    major_axis_angle=major_angle,

    solidity=solidity,
    convex_hull_area=hull_area
)

print(
    "Saved:",
    npz_file
)


# ================================================================
# STEP 18 - VALUE + FORMULA TABLE
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 18 - SAVING VALUE + FORMULA TABLE"
)
print("=" * 80)

csv_file = os.path.join(
    OUTPUT_DIR,
    "GTV1_regional_moment_descriptors_with_formulas.csv"
)

rows = [

    [
        "Feature",
        "Value",
        "Formula / Definition"
    ],

    [
        "Area",
        A,
        "A = number of pixels in the region"
    ],

    [
        "Perimeter",
        P,
        "P = length of the region boundary"
    ],

    [
        "Compactness",
        compactness,
        "Compactness = P^2 / A"
    ],

    [
        "Circularity",
        circularity,
        "Circularity = 4*pi*A / P^2"
    ],

    [
        "Effective Diameter",
        effective_diameter,
        "de = 2*sqrt(A/pi)"
    ],

    [
        "Centroid X",
        x_bar,
        "x_bar = sum(x*f(x,y)) / m00"
    ],

    [
        "Centroid Y",
        y_bar,
        "y_bar = sum(y*f(x,y)) / m00"
    ],

    [
        "Major Axis",
        major_axis,
        "Maximum Euclidean distance between boundary points"
    ],

    [
        "Minor Axis",
        minor_axis,
        "Perpendicular extent of the boundary relative to the major axis"
    ],

    [
        "Major Eigenvalue",
        lambda_major,
        "Largest eigenvalue of covariance matrix"
    ],

    [
        "Minor Eigenvalue",
        lambda_minor,
        "Smallest eigenvalue of covariance matrix"
    ],

    [
        "Eccentricity",
        eccentricity,
        "sqrt(1 - lambda_minor/lambda_major)"
    ],

    [
        "Solidity",
        solidity,
        "Area / Convex Hull Area"
    ],

    [
        "mu20",
        mu20,
        "sum((x-x_bar)^2)"
    ],

    [
        "mu02",
        mu02,
        "sum((y-y_bar)^2)"
    ],

    [
        "mu11",
        mu11,
        "sum((x-x_bar)(y-y_bar))"
    ],

    [
        "mu30",
        mu30,
        "sum((x-x_bar)^3)"
    ],

    [
        "mu03",
        mu03,
        "sum((y-y_bar)^3)"
    ],

    [
        "mu12",
        mu12,
        "sum((x-x_bar)(y-y_bar)^2)"
    ],

    [
        "mu21",
        mu21,
        "sum((x-x_bar)^2(y-y_bar))"
    ],

    [
        "eta20",
        eta20,
        "eta_pq = mu_pq / mu00^(1+(p+q)/2)"
    ],

    [
        "eta02",
        eta02,
        "eta_pq = mu_pq / mu00^(1+(p+q)/2)"
    ],

    [
        "eta11",
        eta11,
        "eta_pq = mu_pq / mu00^(1+(p+q)/2)"
    ],

    [
        "eta30",
        eta30,
        "eta_pq = mu_pq / mu00^(1+(p+q)/2)"
    ],

    [
        "eta03",
        eta03,
        "eta_pq = mu_pq / mu00^(1+(p+q)/2)"
    ],

    [
        "eta12",
        eta12,
        "eta_pq = mu_pq / mu00^(1+(p+q)/2)"
    ],

    [
        "eta21",
        eta21,
        "eta_pq = mu_pq / mu00^(1+(p+q)/2)"
    ],

    [
        "Hu1",
        h1,
        "eta20 + eta02"
    ],

    [
        "Hu2",
        h2,
        "(eta20-eta02)^2 + 4*eta11^2"
    ],

    [
        "Hu3",
        h3,
        "(eta30-3*eta12)^2 + (3*eta21-eta03)^2"
    ],

    [
        "Hu4",
        h4,
        "(eta30+eta12)^2 + (eta21+eta03)^2"
    ],

    [
        "Hu5",
        h5,
        "Hu invariant moment equation 5"
    ],

    [
        "Hu6",
        h6,
        "Hu invariant moment equation 6"
    ],

    [
        "Hu7",
        h7,
        "Hu invariant moment equation 7"
    ]
]

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerows(
        rows
    )

print(
    "Saved:",
    csv_file
)


# ================================================================
# STEP 19 - REGIONAL DESCRIPTORS IMAGE
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 19 - VISUALIZING REGIONAL DESCRIPTORS"
)
print("=" * 80)

plt.figure(
    figsize=(10, 9)
)

# Region

plt.imshow(
    mask,
    cmap="gray",
    origin="upper"
)

# Boundary

local_boundary = boundary.copy()

local_boundary[:, 0] -= offset_x
local_boundary[:, 1] -= offset_y

plt.plot(
    local_boundary[:, 0],
    local_boundary[:, 1],
    linewidth=1.5,
    label="GTV-1 Boundary"
)

# Centroid

centroid_local = (
    centroid_global
    -
    np.array(
        [
            offset_x,
            offset_y
        ]
    )
)

plt.scatter(
    centroid_local[0],
    centroid_local[1],
    s=70,
    marker="x",
    linewidths=2,
    label="Centroid"
)

# ------------------------------------------------
# Major Axis
# ------------------------------------------------

major_p1_local = (
    major_display_p1
    -
    np.array(
        [
            offset_x,
            offset_y
        ]
    )
)

major_p2_local = (
    major_display_p2
    -
    np.array(
        [
            offset_x,
            offset_y
        ]
    )
)

plt.plot(
    [
        major_p1_local[0],
        major_p2_local[0]
    ],
    [
        major_p1_local[1],
        major_p2_local[1]
    ],
    linewidth=2.5,
    label="Major Axis"
)

# ------------------------------------------------
# Minor Axis
# ------------------------------------------------

minor_p1_local = (
    minor_display_p1
    -
    np.array(
        [
            offset_x,
            offset_y
        ]
    )
)

minor_p2_local = (
    minor_display_p2
    -
    np.array(
        [
            offset_x,
            offset_y
        ]
    )
)

plt.plot(
    [
        minor_p1_local[0],
        minor_p2_local[0]
    ],
    [
        minor_p1_local[1],
        minor_p2_local[1]
    ],
    linewidth=2.5,
    label="Minor Axis"
)

plt.title(
    "GTV-1 Regional Descriptors\n"
    f"Area={A:.0f}, "
    f"Major={major_axis:.2f}, "
    f"Minor={minor_axis:.2f}, "
    f"Eccentricity={eccentricity:.4f}"
)

plt.xlabel(
    "X"
)

plt.ylabel(
    "Y"
)

plt.legend()

plt.axis(
    "equal"
)

plt.tight_layout()

regional_image = os.path.join(
    OUTPUT_DIR,
    "01_Regional_Descriptors.png"
)

plt.savefig(
    regional_image,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "Saved:",
    regional_image
)


# ================================================================
# STEP 20 - HU MOMENTS IMAGE
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 20 - VISUALIZING HU MOMENTS"
)
print("=" * 80)

hu_names = [
    "Hu1",
    "Hu2",
    "Hu3",
    "Hu4",
    "Hu5",
    "Hu6",
    "Hu7"
]

hu_abs = np.abs(
    np.array(
        hu,
        dtype=float
    )
)

hu_abs[
    hu_abs == 0
] = np.finfo(float).tiny

plt.figure(
    figsize=(11, 7)
)

bars = plt.bar(
    hu_names,
    hu_abs,
    width=0.65
)

plt.yscale(
    "log"
)

plt.title(
    "Seven Hu Invariant Moments",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Hu Moment",
    fontsize=12
)

plt.ylabel(
    "|Hu Moment| (log scale)",
    fontsize=12
)

# No horizontal grid lines

plt.grid(
    False
)

# Numerical values

for bar, value in zip(
    bars,
    hu
):

    height = (
        bar.get_height()
    )

    if height > 0:

        plt.text(
            bar.get_x()
            +
            bar.get_width()
            / 2.0,

            height * 1.15,

            f"{value:.3e}",

            ha="center",
            va="bottom",

            fontsize=9,

            rotation=90
        )

plt.xticks(
    fontsize=11
)

plt.yticks(
    fontsize=10
)

plt.tight_layout()

hu_image = os.path.join(
    OUTPUT_DIR,
    "02_Hu_Moments.png"
)

plt.savefig(
    hu_image,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

print(
    "Saved:",
    hu_image
)


# ================================================================
# STEP 21 - REPORT
# ================================================================

print("\n" + "=" * 80)
print(
    "STEP 21 - SAVING REPORT"
)
print("=" * 80)

report_file = os.path.join(
    OUTPUT_DIR,
    "regional_moment_descriptors_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GTV-1 REGIONAL AND MOMENT DESCRIPTORS\n"
    )

    f.write(
        "Based on Digital Image Processing, "
        "Gonzalez & Woods, 4th Edition\n\n"
    )

    f.write(
        "=" * 60
        +
        "\n"
    )

    f.write(
        f"Area = {A}\n"
    )

    f.write(
        f"Perimeter = {P}\n"
    )

    f.write(
        f"Compactness = {compactness}\n"
    )

    f.write(
        f"Circularity = {circularity}\n"
    )

    f.write(
        f"Effective Diameter = "
        f"{effective_diameter}\n"
    )

    f.write(
        f"Centroid X = {x_bar}\n"
    )

    f.write(
        f"Centroid Y = {y_bar}\n"
    )

    f.write(
        f"Major Axis = {major_axis}\n"
    )

    f.write(
        f"Minor Axis = {minor_axis}\n"
    )

    f.write(
        f"Major Eigenvalue = "
        f"{lambda_major}\n"
    )

    f.write(
        f"Minor Eigenvalue = "
        f"{lambda_minor}\n"
    )

    f.write(
        f"Regional Eccentricity = "
        f"{eccentricity}\n"
    )

    f.write(
        f"Major Axis Angle = "
        f"{major_angle}\n"
    )

    f.write(
        f"Solidity = {solidity}\n\n"
    )

    f.write(
        "CENTRAL MOMENTS\n"
    )

    f.write(
        f"mu20 = {mu20}\n"
    )

    f.write(
        f"mu02 = {mu02}\n"
    )

    f.write(
        f"mu11 = {mu11}\n"
    )

    f.write(
        f"mu30 = {mu30}\n"
    )

    f.write(
        f"mu03 = {mu03}\n"
    )

    f.write(
        f"mu12 = {mu12}\n"
    )

    f.write(
        f"mu21 = {mu21}\n\n"
    )

    f.write(
        "NORMALIZED CENTRAL MOMENTS\n"
    )

    f.write(
        f"eta20 = {eta20}\n"
    )

    f.write(
        f"eta02 = {eta02}\n"
    )

    f.write(
        f"eta11 = {eta11}\n"
    )

    f.write(
        f"eta30 = {eta30}\n"
    )

    f.write(
        f"eta03 = {eta03}\n"
    )

    f.write(
        f"eta12 = {eta12}\n"
    )

    f.write(
        f"eta21 = {eta21}\n\n"
    )

    f.write(
        "HU MOMENTS\n"
    )

    for i, value in enumerate(
        hu,
        1
    ):

        f.write(
            f"Hu{i} = {value}\n"
        )

print(
    "Saved:",
    report_file
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 80)
print(
    "REGIONAL AND MOMENT DESCRIPTORS COMPLETE"
)
print("=" * 80)

print()

print(
    "Area:",
    A
)

print(
    "Perimeter:",
    P
)

print(
    "Compactness:",
    compactness
)

print(
    "Circularity:",
    circularity
)

print(
    "Effective Diameter:",
    effective_diameter
)

print(
    "Centroid:",
    x_bar,
    y_bar
)

print(
    "Major Axis:",
    major_axis
)

print(
    "Minor Axis:",
    minor_axis
)

print(
    "Major Eigenvalue:",
    lambda_major
)

print(
    "Minor Eigenvalue:",
    lambda_minor
)

print(
    "Eccentricity:",
    eccentricity
)

print(
    "Major Axis Angle:",
    major_angle
)

print(
    "Solidity:",
    solidity
)

print(
    "\nHu Moments:"
)

for i, value in enumerate(
    hu,
    1
):

    print(
        f"Hu{i}: {value:.12e}"
    )

print()

print(
    "Results saved in:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Saved files:"
)

print(
    "GTV1_regional_moment_descriptors.npz"
)

print(
    "GTV1_regional_moment_descriptors_with_formulas.csv"
)

print(
    "01_Regional_Descriptors.png"
)

print(
    "02_Hu_Moments.png"
)

print(
    "regional_moment_descriptors_report.txt"
)


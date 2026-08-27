import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

INPUT_FILE = os.path.join(
    PATIENT_FOLDER,
    "GTV1_ordered_boundary_slice74.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_7_BOUNDARY_STRAIGHTNESS"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# STEP 1 - LOAD ORDERED BOUNDARY
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING ORDERED BOUNDARY")
print("=" * 70)

boundary = np.load(INPUT_FILE).astype(float)

if boundary.ndim != 2 or boundary.shape[1] != 2:
    raise ValueError(
        "Boundary must have shape (N, 2)."
    )

N = len(boundary)

print("Boundary shape:", boundary.shape)
print("Boundary points:", N)

print("\nFirst 10 boundary points:")

for i in range(min(10, N)):
    print(
        i + 1,
        ":",
        tuple(boundary[i])
    )


# ============================================================
# STEP 2 - CONVERT COORDINATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - CONVERTING BOUNDARY COORDINATES")
print("=" * 70)

# Boundary format:
# column 0 = row = y
# column 1 = column = x

y = boundary[:, 0]
x = boundary[:, 1]


# ============================================================
# STEP 3 - BOUNDARY LENGTH
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CALCULATING BOUNDARY LENGTH")
print("=" * 70)

boundary_length = 0.0

for i in range(N):

    p1 = boundary[i]
    p2 = boundary[(i + 1) % N]

    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]

    segment_length = np.sqrt(
        dx ** 2 + dy ** 2
    )

    boundary_length += segment_length


print(
    "Boundary length:",
    boundary_length
)


# ============================================================
# STEP 4 - DIAMETER / MAJOR AXIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - CALCULATING DIAMETER / MAJOR AXIS")
print("=" * 70)

# Book definition:
#
# Diameter(B) = max D(pi, pj)
#
# where pi and pj are points on the boundary.
#
# The line connecting the two extreme points
# is the MAJOR AXIS.

maximum_distance = -1.0

major_point_1 = None
major_point_2 = None

for i in range(N):

    for j in range(i + 1, N):

        dy = (
            boundary[j, 0]
            - boundary[i, 0]
        )

        dx = (
            boundary[j, 1]
            - boundary[i, 1]
        )

        distance = np.sqrt(
            dx ** 2 + dy ** 2
        )

        if distance > maximum_distance:

            maximum_distance = distance

            major_point_1 = i
            major_point_2 = j


diameter = maximum_distance

point_1 = boundary[major_point_1]
point_2 = boundary[major_point_2]


print(
    "Diameter / Major axis length:",
    diameter
)

print(
    "Major axis point 1:",
    tuple(point_1.astype(int))
)

print(
    "Major axis point 2:",
    tuple(point_2.astype(int))
)


# ============================================================
# STEP 5 - MAJOR AXIS ORIENTATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - CALCULATING MAJOR AXIS ORIENTATION")
print("=" * 70)

dx_major = point_2[1] - point_1[1]
dy_major = point_2[0] - point_1[0]

major_axis_angle = np.degrees(
    np.arctan2(
        dy_major,
        dx_major
    )
)

print(
    "Major axis angle:",
    major_axis_angle,
    "degrees"
)


# ============================================================
# STEP 6 - BOUNDARY STRAIGHTNESS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - CALCULATING BOUNDARY STRAIGHTNESS")
print("=" * 70)

# If straightness is required by the project:
#
# Straightness =
#       Diameter
#       ---------
#       Boundary Length
#
# This is dimensionless.

boundary_straightness = (
    diameter /
    boundary_length
)

print(
    "Boundary straightness:",
    boundary_straightness
)


# ============================================================
# STEP 7 - VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - VISUALIZING BOUNDARY AND MAJOR AXIS")
print("=" * 70)

plt.figure(figsize=(9, 9))

# Boundary
plt.plot(
    x,
    y,
    linewidth=1.5,
    label="Ordered Boundary"
)

# Major axis
plt.plot(
    [point_1[1], point_2[1]],
    [point_1[0], point_2[0]],
    linewidth=2,
    label="Major Axis"
)

# Endpoints
plt.scatter(
    point_1[1],
    point_1[0],
    s=100,
    marker="o",
    label="Major Axis Point 1"
)

plt.scatter(
    point_2[1],
    point_2[0],
    s=100,
    marker="o",
    label="Major Axis Point 2"
)

plt.gca().invert_yaxis()
plt.axis("equal")

plt.xlabel("Column (x)")
plt.ylabel("Row (y)")

plt.title(
    "Boundary Diameter and Major Axis"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_Boundary_Diameter_Major_Axis.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 8 - SAVE NUMERICAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - SAVING NUMERICAL RESULTS")
print("=" * 70)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_boundary_length.npy"
    ),
    np.array(boundary_length)
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_boundary_diameter.npy"
    ),
    np.array(diameter)
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_major_axis_angle.npy"
    ),
    np.array(major_axis_angle)
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_boundary_straightness.npy"
    ),
    np.array(boundary_straightness)
)


# ============================================================
# STEP 9 - SAVE REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - SAVING REPORT")
print("=" * 70)

report_file = os.path.join(
    OUTPUT_FOLDER,
    "boundary_straightness_report.txt"
)

with open(report_file, "w") as f:

    f.write(
        "GTV-1 BOUNDARY STRAIGHTNESS REPORT\n"
    )

    f.write(
        "===================================\n\n"
    )

    f.write(
        "Slice: 74\n"
    )

    f.write(
        "Boundary points: "
        + str(N)
        + "\n\n"
    )

    f.write(
        "Boundary length: "
        + str(boundary_length)
        + " pixels\n"
    )

    f.write(
        "Boundary diameter / major axis: "
        + str(diameter)
        + " pixels\n"
    )

    f.write(
        "Major axis angle: "
        + str(major_axis_angle)
        + " degrees\n"
    )

    f.write(
        "Major axis point 1: "
        + str(tuple(point_1))
        + "\n"
    )

    f.write(
        "Major axis point 2: "
        + str(tuple(point_2))
        + "\n\n"
    )

    f.write(
        "Boundary straightness:\n"
    )

    f.write(
        "Straightness = Diameter / Boundary Length\n"
    )

    f.write(
        str(boundary_straightness)
        + "\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BOUNDARY STRAIGHTNESS COMPLETE")
print("=" * 70)

print(
    "Boundary points:",
    N
)

print(
    "Boundary length:",
    boundary_length
)

print(
    "Diameter / Major axis:",
    diameter
)

print(
    "Major axis angle:",
    major_axis_angle,
    "degrees"
)

print(
    "Boundary straightness:",
    boundary_straightness
)

print("\nResults saved in:")
print(OUTPUT_FOLDER)

print("\nSaved image:")
print(
    "01_Boundary_Diameter_Major_Axis.png"
)

print("\nSaved numerical results:")
print(
    "GTV1_boundary_length.npy"
)
print(
    "GTV1_boundary_diameter.npy"
)
print(
    "GTV1_major_axis_angle.npy"
)
print(
    "GTV1_boundary_straightness.npy"
)

print("\nSaved report:")
print(
    "boundary_straightness_report.txt"
)

input("\nPress Enter to close...")
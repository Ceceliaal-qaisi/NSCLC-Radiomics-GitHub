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
    "STEP_6_BOUNDARY_SIGNATURE"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# STEP 1 - LOAD ORDERED BOUNDARY
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING ORDERED BOUNDARY")
print("=" * 70)

boundary = np.load(INPUT_FILE).astype(float)

print("Boundary shape:", boundary.shape)
print("Boundary points:", len(boundary))

print("\nFirst 10 boundary points:")

for i in range(min(10, len(boundary))):
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
# [row, column]
#
# row    -> y
# column -> x

y = boundary[:, 0]
x = boundary[:, 1]


# ============================================================
# STEP 3 - CALCULATE CENTROID
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CALCULATING CENTROID")
print("=" * 70)

centroid_x = np.mean(x)
centroid_y = np.mean(y)

print("Centroid x:", centroid_x)
print("Centroid y:", centroid_y)

print(
    "Centroid:",
    (centroid_x, centroid_y)
)


# ============================================================
# STEP 4 - CALCULATE BOUNDARY SIGNATURE
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - CALCULATING BOUNDARY SIGNATURE")
print("=" * 70)

# Boundary signature:
#
# Distance from the centroid to each boundary point.
#
# r = sqrt[(x - xc)^2 + (y - yc)^2]

distances = np.sqrt(
    (x - centroid_x) ** 2
    +
    (y - centroid_y) ** 2
)

print(
    "Signature length:",
    len(distances)
)

print("\nFirst 10 signature values:")

for i in range(min(10, len(distances))):

    print(
        i + 1,
        ":",
        distances[i]
    )


# ============================================================
# STEP 5 - CALCULATE ANGLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - CALCULATING ANGLES")
print("=" * 70)

angles = np.arctan2(
    y - centroid_y,
    x - centroid_x
)

# Convert radians to degrees

angles_degrees = np.degrees(
    angles
)

# Convert negative angles to [0, 360)

angles_degrees = (
    angles_degrees + 360
) % 360

print("\nFirst 10 angles:")

for i in range(min(10, len(angles_degrees))):

    print(
        i + 1,
        ":",
        angles_degrees[i]
    )


# ============================================================
# STEP 6 - SORT SIGNATURE BY ANGLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - ORDERING SIGNATURE BY ANGLE")
print("=" * 70)

sort_indices = np.argsort(
    angles_degrees
)

sorted_angles = angles_degrees[
    sort_indices
]

sorted_distances = distances[
    sort_indices
]

print(
    "Ordered signature points:",
    len(sorted_distances)
)

print("\nFirst 10 ordered values:")

for i in range(min(10, len(sorted_distances))):

    print(
        i + 1,
        ": angle =",
        sorted_angles[i],
        ", distance =",
        sorted_distances[i]
    )


# ============================================================
# STEP 7 - SIZE NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - SIZE NORMALIZATION")
print("=" * 70)

# The book states that size normalization can be
# achieved by scaling the range of signature values.

max_distance = np.max(
    sorted_distances
)

if max_distance != 0:

    normalized_signature = (
        sorted_distances
        /
        max_distance
    )

else:

    normalized_signature = (
        sorted_distances.copy()
    )


print(
    "Maximum signature value:",
    max_distance
)

print("\nFirst 10 normalized values:")

for i in range(min(10, len(normalized_signature))):

    print(
        i + 1,
        ":",
        normalized_signature[i]
    )


# ============================================================
# STEP 8 - FIND MAXIMUM SIGNATURE POINT
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - MAXIMUM SIGNATURE VALUE")
print("=" * 70)

max_index = np.argmax(
    sorted_distances
)

print(
    "Maximum distance:",
    sorted_distances[max_index]
)

print(
    "Angle at maximum:",
    sorted_angles[max_index],
    "degrees"
)


# ============================================================
# STEP 9 - VISUALIZE BOUNDARY AND CENTROID
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - VISUALIZING BOUNDARY AND CENTROID")
print("=" * 70)

plt.figure(figsize=(8, 8))

plt.plot(
    x,
    y,
    "-",
    linewidth=1.5,
    label="Tumor Boundary"
)

plt.scatter(
    centroid_x,
    centroid_y,
    s=120,
    marker="x",
    label="Centroid"
)

plt.gca().invert_yaxis()

plt.axis("equal")

plt.xlabel("Column (x)")
plt.ylabel("Row (y)")

plt.title(
    "GTV-1 Boundary and Centroid"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_Boundary_Centroid.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 10 - VISUALIZE BOUNDARY SIGNATURE
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - VISUALIZING BOUNDARY SIGNATURE")
print("=" * 70)

plt.figure(figsize=(10, 5))

plt.plot(
    sorted_angles,
    sorted_distances,
    "-o",
    markersize=3,
    linewidth=1.5
)

plt.xlabel(
    "Angle (degrees)"
)

plt.ylabel(
    "Distance from Centroid (pixels)"
)

plt.title(
    "GTV-1 Boundary Signature"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_Boundary_Signature.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 11 - VISUALIZE NORMALIZED SIGNATURE
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 - VISUALIZING NORMALIZED SIGNATURE")
print("=" * 70)

plt.figure(figsize=(10, 5))

plt.plot(
    sorted_angles,
    normalized_signature,
    "-o",
    markersize=3,
    linewidth=1.5
)

plt.xlabel(
    "Angle (degrees)"
)

plt.ylabel(
    "Normalized Distance"
)

plt.title(
    "Normalized GTV-1 Boundary Signature"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_Normalized_Boundary_Signature.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 12 - VISUALIZE SIGNATURE POLAR REPRESENTATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 - POLAR REPRESENTATION")
print("=" * 70)

theta = np.radians(
    sorted_angles
)

plt.figure(figsize=(8, 8))

ax = plt.subplot(
    111,
    projection="polar"
)

ax.plot(
    theta,
    sorted_distances,
    linewidth=1.5
)

ax.set_title(
    "GTV-1 Boundary Signature - Polar Representation"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_Boundary_Signature_Polar.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 13 - SAVE NUMERICAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 - SAVING RESULTS")
print("=" * 70)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_boundary_signature.npy"
    ),
    sorted_distances
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_signature_angles.npy"
    ),
    sorted_angles
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_normalized_boundary_signature.npy"
    ),
    normalized_signature
)


# ============================================================
# STEP 14 - SAVE REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 - SAVING REPORT")
print("=" * 70)

report_file = os.path.join(
    OUTPUT_FOLDER,
    "boundary_signature_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "GTV-1 BOUNDARY SIGNATURE REPORT\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        "Slice: 74\n"
    )

    f.write(
        "Boundary points: "
        + str(len(boundary))
        + "\n"
    )

    f.write(
        "Centroid x: "
        + str(centroid_x)
        + "\n"
    )

    f.write(
        "Centroid y: "
        + str(centroid_y)
        + "\n\n"
    )

    f.write(
        "BOUNDARY SIGNATURE\n"
    )

    f.write(
        "The signature is represented as the "
        "distance from the centroid to the boundary "
        "as a function of angle.\n\n"
    )

    f.write(
        "Maximum distance: "
        + str(max_distance)
        + "\n"
    )

    f.write(
        "Angle of maximum distance: "
        + str(sorted_angles[max_index])
        + " degrees\n\n"
    )

    f.write(
        "SIZE NORMALIZATION\n"
    )

    f.write(
        "The signature was normalized by dividing "
        "all distances by the maximum distance.\n\n"
    )

    f.write(
        "First 20 signature values:\n"
    )

    for value in sorted_distances[:20]:

        f.write(
            str(value) + "\n"
        )

    f.write(
        "\nFirst 20 normalized values:\n"
    )

    for value in normalized_signature[:20]:

        f.write(
            str(value) + "\n"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BOUNDARY SIGNATURE COMPLETE")
print("=" * 70)

print(
    "Boundary points:",
    len(boundary)
)

print(
    "Signature length:",
    len(sorted_distances)
)

print(
    "Centroid:",
    (centroid_x, centroid_y)
)

print(
    "Maximum distance:",
    max_distance
)

print("\nResults saved in:")

print(
    OUTPUT_FOLDER
)

print("\nSaved images:")

print(
    "01_Boundary_Centroid.png"
)

print(
    "02_Boundary_Signature.png"
)

print(
    "03_Normalized_Boundary_Signature.png"
)

print(
    "04_Boundary_Signature_Polar.png"
)

print("\nSaved numerical results:")

print(
    "GTV1_boundary_signature.npy"
)

print(
    "GTV1_signature_angles.npy"
)

print(
    "GTV1_normalized_boundary_signature.npy"
)

print("\nSaved report:")

print(
    "boundary_signature_report.txt"
)

input("\nPress Enter to close...")
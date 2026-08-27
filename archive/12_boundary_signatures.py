import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

RESAMPLED_BOUNDARY_FILE = os.path.join(
    PATIENT_FOLDER,
    "GTV1_resampled_boundary_slice74.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_6_BOUNDARY_SIGNATURES"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# STEP 1 - LOAD RESAMPLED ORDERED BOUNDARY
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING RESAMPLED ORDERED BOUNDARY")
print("=" * 70)

boundary = np.load(
    RESAMPLED_BOUNDARY_FILE
).astype(float)

print("Boundary shape:", boundary.shape)
print("Boundary points:", len(boundary))

print("\nFirst 10 boundary points:")

for i in range(min(10, len(boundary))):
    print(
        i + 1,
        ":",
        tuple(boundary[i].astype(int))
    )


# ============================================================
# STEP 2 - EXTRACT X AND Y
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - PREPARING BOUNDARY COORDINATES")
print("=" * 70)

# Boundary format:
# row    = y
# column = x

y = boundary[:, 0]
x = boundary[:, 1]

K = len(boundary)

print("Number of boundary points:", K)


# ============================================================
# STEP 3 - CALCULATE TANGENT ANGLE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CALCULATING TANGENT ANGLES")
print("=" * 70)

tangent_angles = np.zeros(K)


for i in range(K):

    # Previous point
    p_prev = boundary[
        (i - 1) % K
    ]

    # Next point
    p_next = boundary[
        (i + 1) % K
    ]

    # Difference vector
    dy = p_next[0] - p_prev[0]
    dx = p_next[1] - p_prev[1]

    # Tangent angle
    angle = np.degrees(
        np.arctan2(
            dy,
            dx
        )
    )

    # Convert to range [0, 360)
    if angle < 0:
        angle += 360

    tangent_angles[i] = angle


print(
    "Tangent angles calculated:",
    len(tangent_angles)
)

print("\nFirst 10 tangent angles:")

for i in range(min(10, K)):

    print(
        i + 1,
        ":",
        round(tangent_angles[i], 3),
        "degrees"
    )


# ============================================================
# STEP 4 - TANGENT ANGLE SIGNATURE
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - BOUNDARY TANGENT-ANGLE SIGNATURE")
print("=" * 70)

signature = tangent_angles.copy()

print(
    "Signature length:",
    len(signature)
)

print("\nMinimum angle:",
      round(signature.min(), 3))

print("Maximum angle:",
      round(signature.max(), 3))

print("Mean angle:",
      round(signature.mean(), 3))


# ============================================================
# STEP 5 - VISUALIZE BOUNDARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - VISUALIZING BOUNDARY")
print("=" * 70)

plt.figure(figsize=(8, 8))

plt.plot(
    x,
    y,
    "-o",
    markersize=4,
    linewidth=1.5
)

# Mark starting point

plt.scatter(
    x[0],
    y[0],
    s=150,
    marker="o",
    label="START"
)

plt.text(
    x[0] + 3,
    y[0] - 3,
    "START",
    fontsize=11,
    fontweight="bold"
)

plt.gca().invert_yaxis()

plt.axis("equal")

plt.xlabel("Column (x)")
plt.ylabel("Row (y)")

plt.title(
    "Resampled Ordered Tumor Boundary"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_Resampled_Boundary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 6 - TANGENT ANGLE SIGNATURE PLOT
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - VISUALIZING TANGENT-ANGLE SIGNATURE")
print("=" * 70)

sample_index = np.arange(K)

plt.figure(figsize=(10, 5))

plt.plot(
    sample_index,
    signature,
    "-o",
    markersize=4
)

plt.xlabel("Boundary Point Index")

plt.ylabel(
    "Tangent Angle (degrees)"
)

plt.title(
    "Boundary Signature - Tangent Angle"
)

plt.ylim(0, 360)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_Tangent_Angle_Signature.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 7 - SLOPE DENSITY FUNCTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - CALCULATING SLOPE DENSITY FUNCTION")
print("=" * 70)

# Histogram of tangent-angle values

NUMBER_OF_BINS = 36

histogram, bin_edges = np.histogram(
    tangent_angles,
    bins=NUMBER_OF_BINS,
    range=(0, 360)
)

# Normalize histogram

slope_density = (
    histogram / np.sum(histogram)
)

bin_centers = (
    bin_edges[:-1]
    +
    bin_edges[1:]
) / 2


print(
    "Number of angle bins:",
    NUMBER_OF_BINS
)

print(
    "Total histogram samples:",
    np.sum(histogram)
)

print("\nSlope-density values:")

for i in range(len(slope_density)):

    print(
        f"{bin_centers[i]:6.1f} deg : "
        f"{slope_density[i]:.4f}"
    )


# ============================================================
# STEP 8 - VISUALIZE SLOPE DENSITY
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - VISUALIZING SLOPE DENSITY FUNCTION")
print("=" * 70)

plt.figure(figsize=(10, 5))

plt.bar(
    bin_centers,
    slope_density,
    width=10
)

plt.xlabel(
    "Tangent Angle (degrees)"
)

plt.ylabel(
    "Normalized Frequency"
)

plt.title(
    "Slope Density Function"
)

plt.xlim(0, 360)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_Slope_Density_Function.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 9 - VISUALIZE TANGENT DIRECTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - VISUALIZING TANGENT DIRECTIONS")
print("=" * 70)

plt.figure(figsize=(8, 8))

plt.plot(
    x,
    y,
    "-o",
    markersize=3,
    linewidth=1
)

# Draw tangent vectors

scale = 5

for i in range(K):

    angle_rad = np.radians(
        tangent_angles[i]
    )

    dx = np.cos(angle_rad) * scale
    dy = np.sin(angle_rad) * scale

    plt.arrow(
        x[i],
        y[i],
        dx,
        dy,
        head_width=1.5,
        head_length=2,
        length_includes_head=True
    )


plt.scatter(
    x[0],
    y[0],
    s=150,
    marker="o",
    label="START"
)

plt.gca().invert_yaxis()

plt.axis("equal")

plt.xlabel("Column (x)")
plt.ylabel("Row (y)")

plt.title(
    "Boundary Tangent Directions"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_Tangent_Directions.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 10 - SAVE NUMERICAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - SAVING NUMERICAL RESULTS")
print("=" * 70)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_tangent_angles.npy"
    ),
    tangent_angles
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_slope_density.npy"
    ),
    slope_density
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_slope_density_bin_centers.npy"
    ),
    bin_centers
)


# ============================================================
# STEP 11 - REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_FOLDER,
    "boundary_signatures_report.txt"
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
        "Boundary samples: "
        + str(K)
        + "\n\n"
    )

    f.write(
        "SIGNATURE METHOD\n"
    )

    f.write(
        "Tangent angle signature\n"
    )

    f.write(
        "The tangent angle at each boundary point "
        "was calculated relative to the x-axis.\n\n"
    )

    f.write(
        "First 10 tangent angles:\n"
    )

    for i in range(min(10, K)):

        f.write(
            f"{i + 1}: "
            f"{tangent_angles[i]:.6f} degrees\n"
        )

    f.write(
        "\nSLOPE DENSITY FUNCTION\n"
    )

    f.write(
        "Number of bins: "
        + str(NUMBER_OF_BINS)
        + "\n\n"
    )

    for i in range(len(slope_density)):

        f.write(
            f"{bin_centers[i]:.1f} degrees: "
            f"{slope_density[i]:.6f}\n"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BOUNDARY SIGNATURES COMPLETE")
print("=" * 70)

print(
    "Boundary samples:",
    K
)

print(
    "Tangent-angle signature:",
    len(tangent_angles)
)

print(
    "Slope-density bins:",
    NUMBER_OF_BINS
)

print("\nResults saved in:")

print(
    OUTPUT_FOLDER
)

print("\nSaved images:")

print(
    "01_Resampled_Boundary.png"
)

print(
    "02_Tangent_Angle_Signature.png"
)

print(
    "03_Slope_Density_Function.png"
)

print(
    "04_Tangent_Directions.png"
)

print("\nSaved numerical results:")

print(
    "GTV1_tangent_angles.npy"
)

print(
    "GTV1_slope_density.npy"
)

print(
    "GTV1_slope_density_bin_centers.npy"
)

print("\nSaved report:")

print(
    "boundary_signatures_report.txt"
)

input("\nPress Enter to close...")
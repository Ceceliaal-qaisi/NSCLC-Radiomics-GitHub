import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

# IMPORTANT:
# This is the file actually created by the Chain Code step.
BOUNDARY_FILE = os.path.join(
    PATIENT_FOLDER,
    "GTV1_resampled_boundary_slice74.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_5_FOURIER_DESCRIPTORS"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# STEP 1 - LOAD RESAMPLED ORDERED BOUNDARY
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING RESAMPLED ORDERED BOUNDARY")
print("=" * 70)

if not os.path.exists(BOUNDARY_FILE):
    raise FileNotFoundError(
        "\nBoundary file not found:\n"
        + BOUNDARY_FILE
        + "\n\nCheck that the Chain Code step was completed."
    )

boundary = np.load(BOUNDARY_FILE).astype(float)

print("Boundary shape:", boundary.shape)
print("Boundary points:", len(boundary))

if boundary.ndim != 2 or boundary.shape[1] != 2:
    raise ValueError(
        "Boundary must have shape (N, 2)."
    )

print("\nFirst 10 boundary points:")

for i in range(min(10, len(boundary))):
    print(
        i + 1,
        ":",
        tuple(boundary[i].astype(int))
    )


# ============================================================
# STEP 2 - COMPLEX REPRESENTATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - CONVERTING BOUNDARY TO COMPLEX SEQUENCE")
print("=" * 70)

# Boundary:
# row    = y
# column = x
#
# Complex representation:
#
# s(k) = x(k) + j*y(k)

x = boundary[:, 1]
y = boundary[:, 0]

s = x + 1j * y

K = len(s)

print("Number of boundary samples K:", K)

print("\nFirst 10 complex values:")

for i in range(min(10, K)):
    print(
        i + 1,
        ":",
        s[i]
    )


# ============================================================
# STEP 3 - TRANSLATION INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - TRANSLATION COMPONENT")
print("=" * 70)

# The DC component a(0) represents the position/centroid
# information of the boundary.
#
# To demonstrate translation normalization, we center
# the boundary by subtracting its centroid.

centroid_x = np.mean(x)
centroid_y = np.mean(y)

centroid = centroid_x + 1j * centroid_y

print("Centroid x:", centroid_x)
print("Centroid y:", centroid_y)
print("Centroid:", centroid)

s_centered = s - centroid


# ============================================================
# STEP 4 - FOURIER TRANSFORM FROM SCRATCH
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - CALCULATING FOURIER DESCRIPTORS")
print("=" * 70)

# Fourier descriptor equation:
#
# a(u) = SUM [s(k) * exp(-j*2*pi*u*k/K)]
#
# No np.fft is used.

fourier_descriptors = np.zeros(
    K,
    dtype=complex
)

for u in range(K):

    total = 0j

    for k in range(K):

        angle = (
            -2.0
            * np.pi
            * u
            * k
            / K
        )

        total += (
            s[k]
            * np.exp(1j * angle)
        )

    fourier_descriptors[u] = total


print("Fourier descriptor count:", K)

print("\nFirst 10 Fourier descriptors:")

for i in range(min(10, K)):
    print(
        i,
        ":",
        fourier_descriptors[i]
    )


# ============================================================
# STEP 5 - CENTERED FOURIER DESCRIPTORS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - TRANSLATION NORMALIZED DESCRIPTORS")
print("=" * 70)

# Translation normalization:
#
# Set the DC component to zero.
#
# This removes the effect of the absolute position
# of the object in the image.

translation_normalized = (
    fourier_descriptors.copy()
)

translation_normalized[0] = 0

print(
    "Original DC component:",
    fourier_descriptors[0]
)

print(
    "Translation-normalized DC component:",
    translation_normalized[0]
)


# ============================================================
# STEP 6 - MAGNITUDE
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - FOURIER DESCRIPTOR MAGNITUDES")
print("=" * 70)

magnitudes = np.abs(
    translation_normalized
)

print("\nFirst 10 magnitudes:")

for i in range(min(10, K)):
    print(
        i,
        ":",
        magnitudes[i]
    )


# ============================================================
# STEP 7 - SCALE NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - SCALE NORMALIZATION")
print("=" * 70)

# Scale normalization:
#
# Divide all descriptor magnitudes by the magnitude
# of the first non-zero harmonic.
#
# This produces a normalized descriptor representation
# that is independent of the overall size.

if magnitudes[1] == 0:

    raise ValueError(
        "Cannot perform scale normalization because "
        "the first harmonic is zero."
    )

scale_reference = magnitudes[1]

normalized_magnitudes = (
    magnitudes / scale_reference
)

print(
    "Scale reference |a(1)|:",
    scale_reference
)

print("\nFirst 10 scale-normalized magnitudes:")

for i in range(min(10, K)):
    print(
        i,
        ":",
        normalized_magnitudes[i]
    )


# ============================================================
# STEP 8 - FULL RECONSTRUCTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - FULL FOURIER RECONSTRUCTION")
print("=" * 70)


def inverse_fourier(
    coefficients,
    number_of_samples
):

    N = len(coefficients)

    reconstructed = np.zeros(
        number_of_samples,
        dtype=complex
    )

    for k in range(number_of_samples):

        total = 0j

        for u in range(N):

            angle = (
                2.0
                * np.pi
                * u
                * k
                / number_of_samples
            )

            total += (
                coefficients[u]
                * np.exp(1j * angle)
            )

        reconstructed[k] = (
            total / number_of_samples
        )

    return reconstructed


reconstructed_full = inverse_fourier(
    fourier_descriptors,
    K
)

reconstruction_error = np.mean(
    np.abs(
        s - reconstructed_full
    )
)

print(
    "Mean reconstruction error:",
    reconstruction_error
)

if reconstruction_error < 1e-6:

    print(
        "Full reconstruction check: PASS"
    )

else:

    print(
        "Full reconstruction check: CHECK"
    )


# ============================================================
# STEP 9 - TRUNCATED FOURIER DESCRIPTORS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - TRUNCATED FOURIER RECONSTRUCTION")
print("=" * 70)

# Low-frequency descriptors describe the global shape.
# Higher-frequency descriptors describe fine details.

descriptor_counts = [
    24,
    12,
    6,
    4,
    2
]

reconstructions = {}


def create_truncated_coefficients(
    descriptors,
    number_to_keep
):

    N = len(descriptors)

    truncated = np.zeros(
        N,
        dtype=complex
    )

    if number_to_keep >= N:

        truncated[:] = descriptors

        return truncated

    # Always retain DC component.
    truncated[0] = descriptors[0]

    remaining = number_to_keep - 1

    positive_count = remaining // 2

    negative_count = (
        remaining - positive_count
    )

    if positive_count > 0:

        truncated[
            1:positive_count + 1
        ] = descriptors[
            1:positive_count + 1
        ]

    if negative_count > 0:

        truncated[
            -negative_count:
        ] = descriptors[
            -negative_count:
        ]

    return truncated


for count in descriptor_counts:

    truncated = create_truncated_coefficients(
        fourier_descriptors,
        count
    )

    reconstructed = inverse_fourier(
        truncated,
        K
    )

    reconstructions[count] = reconstructed

    print(
        "Descriptors retained:",
        count
    )


# ============================================================
# STEP 10 - ORIGINAL BOUNDARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - VISUALIZING ORIGINAL BOUNDARY")
print("=" * 70)

plt.figure(figsize=(7, 7))

plt.plot(
    x,
    y,
    "-o",
    markersize=4
)

plt.gca().invert_yaxis()
plt.axis("equal")

plt.xlabel("Column (x)")
plt.ylabel("Row (y)")

plt.title(
    "Original Resampled Boundary"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_Original_Boundary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 11 - FULL RECONSTRUCTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 - FULL RECONSTRUCTION VISUALIZATION")
print("=" * 70)

plt.figure(figsize=(7, 7))

plt.plot(
    x,
    y,
    "--",
    linewidth=1.5,
    label="Original"
)

plt.plot(
    reconstructed_full.real,
    reconstructed_full.imag,
    "-o",
    markersize=3,
    label="Full Fourier Reconstruction"
)

plt.gca().invert_yaxis()
plt.axis("equal")

plt.xlabel("Column (x)")
plt.ylabel("Row (y)")

plt.title(
    "Full Fourier Reconstruction"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_Full_Reconstruction.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 12 - TRUNCATED RECONSTRUCTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 - VISUALIZING TRUNCATED RECONSTRUCTIONS")
print("=" * 70)

for count in descriptor_counts:

    reconstructed = reconstructions[count]

    plt.figure(figsize=(7, 7))

    plt.plot(
        x,
        y,
        "--",
        linewidth=1,
        label="Original Boundary"
    )

    plt.plot(
        reconstructed.real,
        reconstructed.imag,
        "-o",
        markersize=3,
        label=f"{count} Descriptors"
    )

    plt.gca().invert_yaxis()
    plt.axis("equal")

    plt.xlabel("Column (x)")
    plt.ylabel("Row (y)")

    plt.title(
        f"Fourier Reconstruction - {count} Descriptors"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    filename = (
        f"03_Reconstruction_{count}_Descriptors.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# STEP 13 - ALL RECONSTRUCTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 - COMPARING ALL RECONSTRUCTIONS")
print("=" * 70)

plt.figure(figsize=(9, 8))

plt.plot(
    x,
    y,
    "--",
    linewidth=2,
    label="Original"
)

for count in descriptor_counts:

    reconstructed = reconstructions[count]

    plt.plot(
        reconstructed.real,
        reconstructed.imag,
        linewidth=1.5,
        label=f"{count} descriptors"
    )

plt.gca().invert_yaxis()
plt.axis("equal")

plt.xlabel("Column (x)")
plt.ylabel("Row (y)")

plt.title(
    "Fourier Descriptors and Boundary Reconstruction"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "09_All_Fourier_Reconstructions.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 14 - FOURIER MAGNITUDE SPECTRUM
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 - FOURIER MAGNITUDE SPECTRUM")
print("=" * 70)

plt.figure(figsize=(10, 5))

plt.stem(
    range(K),
    magnitudes
)

plt.xlabel(
    "Fourier Descriptor Index"
)

plt.ylabel(
    "Magnitude"
)

plt.title(
    "Magnitude Spectrum of Fourier Descriptors"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "10_Fourier_Magnitude_Spectrum.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 15 - NORMALIZED MAGNITUDE SPECTRUM
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 - NORMALIZED FOURIER DESCRIPTORS")
print("=" * 70)

plt.figure(figsize=(10, 5))

plt.stem(
    range(K),
    normalized_magnitudes
)

plt.xlabel(
    "Fourier Descriptor Index"
)

plt.ylabel(
    "Normalized Magnitude"
)

plt.title(
    "Scale-Normalized Fourier Descriptor Magnitudes"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "11_Normalized_Fourier_Descriptors.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 16 - SAVE NUMERICAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 16 - SAVING NUMERICAL RESULTS")
print("=" * 70)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_fourier_descriptors.npy"
    ),
    fourier_descriptors
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_translation_normalized_descriptors.npy"
    ),
    translation_normalized
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_fourier_magnitudes.npy"
    ),
    magnitudes
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_normalized_fourier_magnitudes.npy"
    ),
    normalized_magnitudes
)


# ============================================================
# STEP 17 - SAVE REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 17 - SAVING REPORT")
print("=" * 70)

report_file = os.path.join(
    OUTPUT_FOLDER,
    "fourier_descriptors_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "GTV-1 FOURIER DESCRIPTORS REPORT\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        "Patient: LUNG1-001\n"
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
        "COMPLEX REPRESENTATION\n"
    )

    f.write(
        "s(k) = x(k) + j*y(k)\n\n"
    )

    f.write(
        "FOURIER TRANSFORM\n"
    )

    f.write(
        "a(u) = SUM s(k) exp(-j*2*pi*u*k/K)\n"
    )

    f.write(
        "Fourier descriptors calculated from scratch.\n"
    )

    f.write(
        "No np.fft was used.\n\n"
    )

    f.write(
        "TRANSLATION\n"
    )

    f.write(
        "The DC component represents positional information.\n"
    )

    f.write(
        "Translation-normalized descriptors were obtained "
        "by setting the DC component to zero.\n\n"
    )

    f.write(
        "SCALE NORMALIZATION\n"
    )

    f.write(
        "Descriptor magnitudes were normalized using "
        "the magnitude of the first non-zero harmonic |a(1)|.\n\n"
    )

    f.write(
        "LOW AND HIGH FREQUENCIES\n"
    )

    f.write(
        "Low-frequency descriptors represent the global "
        "shape of the boundary.\n"
    )

    f.write(
        "High-frequency descriptors represent finer "
        "boundary details.\n\n"
    )

    f.write(
        "BOUNDARY RECONSTRUCTION\n"
    )

    f.write(
        "Full reconstruction mean error: "
        + str(reconstruction_error)
        + "\n\n"
    )

    f.write(
        "DESCRIPTORS USED FOR TRUNCATED RECONSTRUCTION\n"
    )

    for count in descriptor_counts:

        f.write(
            str(count)
            + " descriptors\n"
        )

    f.write(
        "\nFIRST 10 FOURIER DESCRIPTORS\n"
    )

    for i in range(min(10, K)):

        f.write(
            f"{i}: "
            f"{fourier_descriptors[i]}\n"
        )

    f.write(
        "\nFIRST 10 MAGNITUDES\n"
    )

    for i in range(min(10, K)):

        f.write(
            f"{i}: "
            f"{magnitudes[i]}\n"
        )

    f.write(
        "\nFIRST 10 SCALE-NORMALIZED MAGNITUDES\n"
    )

    for i in range(min(10, K)):

        f.write(
            f"{i}: "
            f"{normalized_magnitudes[i]}\n"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FOURIER DESCRIPTORS COMPLETE")
print("=" * 70)

print(
    "Boundary samples:",
    K
)

print(
    "Fourier descriptors:",
    K
)

print(
    "Full reconstruction error:",
    reconstruction_error
)

print(
    "\nResults saved in:"
)

print(
    OUTPUT_FOLDER
)

print("\nSaved images:")

print(
    "01_Original_Boundary.png"
)

print(
    "02_Full_Reconstruction.png"
)

for count in descriptor_counts:

    print(
        f"03_Reconstruction_{count}_Descriptors.png"
    )

print(
    "09_All_Fourier_Reconstructions.png"
)

print(
    "10_Fourier_Magnitude_Spectrum.png"
)

print(
    "11_Normalized_Fourier_Descriptors.png"
)

print("\nSaved numerical results:")

print(
    "GTV1_fourier_descriptors.npy"
)

print(
    "GTV1_translation_normalized_descriptors.npy"
)

print(
    "GTV1_fourier_magnitudes.npy"
)

print(
    "GTV1_normalized_fourier_magnitudes.npy"
)

print("\nSaved report:")

print(
    "fourier_descriptors_report.txt"
)

input(
    "\nPress Enter to close..."
)
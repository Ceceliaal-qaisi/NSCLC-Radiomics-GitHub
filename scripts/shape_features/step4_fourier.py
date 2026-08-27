import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 4 - FOURIER DESCRIPTORS + SHAPE RECONSTRUCTION
#
# Project 7 - NSCLC Radiomics
#
# Implements:
#   1. Complex boundary representation
#   2. Discrete Fourier Transform
#   3. Fourier descriptors
#   4. Truncated harmonic reconstruction
#   5. Visualization of harmonic contribution
# ============================================================

PATIENT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

STEP3_DIR = os.path.join(
    PATIENT_DIR,
    "STEP3_FINAL"
)

INPUT_FILE = os.path.join(
    STEP3_DIR,
    "step3_final_results.npy"
)

OUTPUT_DIR = os.path.join(
    PATIENT_DIR,
    "FOURIER_DESCRIPTORS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

# Harmonics retained for reconstruction.
#
# Low frequencies -> global shape
# Higher frequencies -> finer boundary details

HARMONIC_LEVELS = [
    1,
    2,
    4,
    8,
    16,
    32
]

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 4 - FOURIER DESCRIPTORS")
print("       + TRUNCATED HARMONIC RECONSTRUCTION")
print("=" * 70)

print("\nInput:")
print(INPUT_FILE)

print("\nOutput:")
print(OUTPUT_DIR)

# ============================================================
# LOAD STEP 3 RESULTS
# ============================================================

print("\n" + "-" * 70)
print("Loading Step 3 results...")
print("-" * 70)

data = np.load(
    INPUT_FILE,
    allow_pickle=True
)

results = data.item()

print("Step 3 results loaded successfully.")
print("Number of slices:", len(results))

# ============================================================
# FOURIER FUNCTIONS
# ============================================================

def boundary_to_complex(points):
    """
    Convert 2-D boundary coordinates into
    a complex-valued signal.

    z = x + j*y
    """

    points = np.asarray(
        points,
        dtype=float
    )

    y = points[:, 0]
    x = points[:, 1]

    z = x + 1j * y

    return z


def compute_fourier_descriptors(points):
    """
    Compute the DFT of the complex boundary.

    The resulting coefficients are the
    Fourier descriptors.
    """

    z = boundary_to_complex(
        points
    )

    descriptors = np.fft.fft(z)

    return descriptors


def reconstruct_boundary(
    descriptors,
    number_of_harmonics
):
    """
    Reconstruct the boundary using a truncated
    set of Fourier harmonics.

    We retain the DC component and the lowest
    positive/negative frequency components.

    This demonstrates the contribution of
    low versus high frequency information.
    """

    N = len(descriptors)

    if N == 0:
        return np.array([])

    number_of_harmonics = min(
        number_of_harmonics,
        (N - 1) // 2
    )

    truncated = np.zeros_like(
        descriptors
    )

    # --------------------------------------------------------
    # DC component
    # --------------------------------------------------------

    truncated[0] = descriptors[0]

    # --------------------------------------------------------
    # Positive and negative frequencies
    # --------------------------------------------------------

    for k in range(
        1,
        number_of_harmonics + 1
    ):

        truncated[k] = descriptors[k]

        truncated[-k] = descriptors[-k]

    reconstructed = np.fft.ifft(
        truncated
    )

    return reconstructed


def normalize_fourier_descriptors(
    descriptors
):
    """
    Scale-normalized Fourier descriptors.

    The first non-zero harmonic magnitude
    is used as the scale reference.

    This is provided as a normalized descriptor
    representation for comparison.
    """

    descriptors = np.asarray(
        descriptors,
        dtype=complex
    )

    normalized = descriptors.copy()

    # Translation invariance:
    # remove DC component.
    normalized[0] = 0

    magnitudes = np.abs(
        normalized
    )

    nonzero = np.where(
        magnitudes > 1e-12
    )[0]

    if len(nonzero) == 0:
        return normalized

    reference = magnitudes[
        nonzero[0]
    ]

    if reference > 0:

        normalized = (
            normalized /
            reference
        )

    return normalized


# ============================================================
# PROCESS EACH SLICE
# ============================================================

fourier_results = {}

print("\n" + "=" * 70)
print("PROCESSING FOURIER DESCRIPTORS")
print("=" * 70)

for slice_number in sorted(
    results.keys()
):

    print("\n" + "-" * 60)
    print("Slice:", slice_number)
    print("-" * 60)

    points = np.asarray(
        results[slice_number][
            "resampled_boundary"
        ],
        dtype=float
    )

    print(
        "Boundary points:",
        len(points)
    )

    # --------------------------------------------------------
    # FOURIER TRANSFORM
    # --------------------------------------------------------

    descriptors = compute_fourier_descriptors(
        points
    )

    print(
        "Fourier descriptors:",
        len(descriptors)
    )

    # --------------------------------------------------------
    # NORMALIZED DESCRIPTORS
    # --------------------------------------------------------

    normalized = (
        normalize_fourier_descriptors(
            descriptors
        )
    )

    print(
        "Normalized Fourier descriptors: DONE"
    )

    # --------------------------------------------------------
    # RECONSTRUCTIONS
    # --------------------------------------------------------

    reconstructions = {}

    for h in HARMONIC_LEVELS:

        reconstructed = reconstruct_boundary(
            descriptors,
            h
        )

        reconstructions[h] = reconstructed

        print(
            f"Reconstruction using "
            f"{h} harmonics: DONE"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    fourier_results[
        int(slice_number)
    ] = {

        "boundary": points,

        "fourier_descriptors":
            descriptors,

        "normalized_fourier_descriptors":
            normalized,

        "reconstructions":
            reconstructions
    }

# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "fourier_results.npy"
)

np.save(
    results_file,
    fourier_results,
    allow_pickle=True
)

print("\n" + "=" * 70)
print("FOURIER RESULTS SAVED")
print("=" * 70)

print(results_file)

# ============================================================
# SAVE FOURIER INFORMATION
# ============================================================

info_file = os.path.join(
    OUTPUT_DIR,
    "fourier_info.txt"
)

with open(
    info_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 4 - FOURIER DESCRIPTORS\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "Representation:\n"
    )

    f.write(
        "z(n) = x(n) + j*y(n)\n\n"
    )

    f.write(
        "Fourier transform:\n"
    )

    f.write(
        "Z(k) = DFT{z(n)}\n\n"
    )

    f.write(
        "Harmonic levels used for reconstruction:\n"
    )

    f.write(
        str(HARMONIC_LEVELS)
    )

    f.write("\n\n")

    f.write(
        "Interpretation:\n"
    )

    f.write(
        "Low-frequency harmonics represent the "
        "global/coarse shape.\n"
    )

    f.write(
        "Higher-frequency harmonics add finer "
        "boundary details and irregularities.\n\n"
    )

    f.write(
        f"Number of processed slices: "
        f"{len(fourier_results)}\n"
    )

print(
    "\nInformation saved:"
)

print(info_file)

# ============================================================
# VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING RECONSTRUCTION FIGURES")
print("=" * 70)

slice_numbers = sorted(
    fourier_results.keys()
)

# Select representative slices:
# first, middle, last

if len(slice_numbers) >= 3:

    visual_slices = [

        slice_numbers[0],

        slice_numbers[
            len(slice_numbers) // 2
        ],

        slice_numbers[-1]
    ]

else:

    visual_slices = slice_numbers


for s in visual_slices:

    data_s = fourier_results[s]

    original = data_s[
        "boundary"
    ]

    reconstructions = data_s[
        "reconstructions"
    ]

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 9)
    )

    axes = axes.flatten()

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    axes[0].plot(
        original[:, 1],
        original[:, 0],
        "k-",
        linewidth=2
    )

    axes[0].set_title(
        "Original Boundary"
    )

    axes[0].set_aspect(
        "equal"
    )

    axes[0].invert_yaxis()

    # --------------------------------------------------------
    # RECONSTRUCTIONS
    # --------------------------------------------------------

    for i, h in enumerate(
        HARMONIC_LEVELS,
        start=1
    ):

        reconstructed = (
            reconstructions[h]
        )

        x = reconstructed.real
        y = reconstructed.imag

        axes[i].plot(
            x,
            y,
            "b-",
            linewidth=2
        )

        axes[i].set_title(
            f"{h} Harmonics"
        )

        axes[i].set_aspect(
            "equal"
        )

        axes[i].invert_yaxis()

    # --------------------------------------------------------
    # EXTRA PLOT: FOURIER MAGNITUDE
    # --------------------------------------------------------

    descriptors = data_s[
        "fourier_descriptors"
    ]

    magnitudes = np.abs(
        descriptors
    )

    axes[7].plot(
        np.arange(
            len(magnitudes)
        ),
        magnitudes
    )

    axes[7].set_title(
        "Fourier Descriptor Magnitudes"
    )

    axes[7].set_xlabel(
        "Harmonic"
    )

    axes[7].set_ylabel(
        "Magnitude"
    )

    fig.suptitle(
        f"Fourier Reconstruction - Slice {s}",
        fontsize=16
    )

    plt.tight_layout()

    image_file = os.path.join(
        OUTPUT_DIR,
        f"slice_{s}_fourier_reconstruction.png"
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

# ============================================================
# CREATE HARMONIC INTERPRETATION FILE
# ============================================================

interpretation_file = os.path.join(
    OUTPUT_DIR,
    "harmonic_interpretation.txt"
)

with open(
    interpretation_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FOURIER HARMONIC INTERPRETATION\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        "1 harmonic:\n"
    )

    f.write(
        "Very coarse representation of the overall "
        "shape.\n\n"
    )

    f.write(
        "2-4 harmonics:\n"
    )

    f.write(
        "Capture increasingly important global "
        "shape characteristics.\n\n"
    )

    f.write(
        "8-16 harmonics:\n"
    )

    f.write(
        "Represent major shape variations and "
        "moderate boundary irregularities.\n\n"
    )

    f.write(
        "32 harmonics:\n"
    )

    f.write(
        "Retain substantially more local boundary "
        "detail.\n\n"
    )

    f.write(
        "Full Fourier representation:\n"
    )

    f.write(
        "Reconstructs the sampled boundary with "
        "minimal truncation loss.\n"
    )

print(
    "\nInterpretation saved:"
)

print(
    interpretation_file
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nFourier descriptors: DONE"
)

print(
    "Normalized Fourier descriptors: DONE"
)

print(
    "Truncated harmonic reconstruction: DONE"
)

print(
    "Harmonic visualization: DONE"
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
    "REGIONAL DESCRIPTORS + 7 HU MOMENTS"
)

print("=" * 70)

input("\nPress ENTER to close...")
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 13 - SPECTRAL (FOURIER-BASED) TEXTURE MEASURES
#
# Based on Chapter 11 - Texture / Spectral Approaches
#
# From scratch / explicit implementation:
#   - Tumor-region extraction
#   - 2-D Fourier spectrum
#   - Polar frequency representation
#   - Spectral-energy function S(r)
#   - Spectral-energy function S(u)
#   - Dominant frequency peak
#   - Fundamental spatial period
#   - Dominant texture direction
#   - Mean and variance of radial spectral energy
#   - Mean and variance of angular spectral energy
#
# No skimage texture functions are used.
# ================================================================

import os
import glob
import math

import numpy as np
import pandas as pd
import pydicom
import nrrd
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_DIR = os.path.join(
    BASE_DIR,
    "82046"
)

MASK_FILE = os.path.join(
    BASE_DIR,
    "GTV1_MASK.nrrd"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_13_SPECTRAL_TEXTURE"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 13 - SPECTRAL (FOURIER-BASED) TEXTURE")
print("=" * 70)


# ================================================================
# STEP 1 - READING CT
# ================================================================

print("\nSTEP 1 - READING CT")
print("=" * 70)

ct_files = glob.glob(
    os.path.join(
        CT_DIR,
        "*.dcm"
    )
)

if len(ct_files) == 0:
    raise FileNotFoundError(
        f"No CT DICOM files found in:\n{CT_DIR}"
    )


def get_instance_number(path):
    ds = pydicom.dcmread(
        path,
        stop_before_pixels=True
    )

    return int(
        getattr(
            ds,
            "InstanceNumber",
            0
        )
    )


ct_files = sorted(
    ct_files,
    key=get_instance_number
)


ct_slices = []


for path in ct_files:

    ds = pydicom.dcmread(path)

    image = ds.pixel_array.astype(
        np.float64
    )

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0.0
        )
    )

    hu = (
        image * slope
        + intercept
    )

    ct_slices.append(
        hu
    )


ct_volume = np.stack(
    ct_slices,
    axis=0
)


print(
    "CT files:",
    len(ct_files)
)

print(
    "CT shape (Z,Y,X):",
    ct_volume.shape
)


# ================================================================
# STEP 2 - READING GTV-1 MASK
# ================================================================

print("\nSTEP 2 - READING GTV-1 MASK")
print("=" * 70)


if not os.path.exists(MASK_FILE):

    raise FileNotFoundError(
        f"GTV1 mask not found:\n{MASK_FILE}"
    )


mask_data, mask_header = nrrd.read(
    MASK_FILE
)


mask = np.asarray(
    mask_data
)


print(
    "Original mask shape:",
    mask.shape
)


# NRRD = (X,Y,Z)
# CT   = (Z,Y,X)

if mask.shape == (
    ct_volume.shape[2],
    ct_volume.shape[1],
    ct_volume.shape[0]
):

    mask = np.transpose(
        mask,
        (2, 1, 0)
    )

elif mask.shape != ct_volume.shape:

    raise ValueError(
        f"CT and mask dimensions do not match.\n"
        f"CT: {ct_volume.shape}\n"
        f"Mask: {mask.shape}"
    )


binary_mask = (
    mask > 0
)


tumor_voxels = int(
    np.sum(binary_mask)
)


print(
    "Converted mask shape:",
    binary_mask.shape
)

print(
    "Tumor voxels:",
    tumor_voxels
)


if tumor_voxels == 0:

    raise ValueError(
        "GTV-1 mask is empty."
    )


# ================================================================
# STEP 3 - FIND TUMOR SLICES
# ================================================================

print("\nSTEP 3 - FINDING TUMOR SLICES")
print("=" * 70)


tumor_slice_indices = np.where(
    np.any(
        binary_mask,
        axis=(1, 2)
    )
)[0]


print(
    "Tumor slices:",
    len(tumor_slice_indices)
)

print(
    "First tumor slice:",
    int(tumor_slice_indices[0])
)

print(
    "Last tumor slice:",
    int(tumor_slice_indices[-1])
)


# ================================================================
# STEP 4 - FOURIER SPECTRUM FUNCTIONS
# ================================================================

print(
    "\nSTEP 4 - DEFINING FOURIER SPECTRUM"
)

print("=" * 70)


def extract_tumor_patch(
    image,
    tumor_mask
):

    rows, cols = np.where(
        tumor_mask
    )

    if len(rows) == 0:

        return None


    r_min = int(
        np.min(rows)
    )

    r_max = int(
        np.max(rows)
    )

    c_min = int(
        np.min(cols)
    )

    c_max = int(
        np.max(cols)
    )


    patch = image[
        r_min:r_max + 1,
        c_min:c_max + 1
    ]


    patch_mask = tumor_mask[
        r_min:r_max + 1,
        c_min:c_max + 1
    ]


    return (
        patch,
        patch_mask
    )


# ================================================================
# RADIAL SPECTRAL ENERGY S(r)
# ================================================================

def calculate_radial_spectral_energy(
    power,
    radial_frequency
):

    max_radius = float(
        np.max(
            radial_frequency
        )
    )


    radial_bins = np.linspace(
        0.0,
        max_radius,
        50
    )


    radial_energy = []

    radial_centers = []


    for i in range(
        len(radial_bins) - 1
    ):

        lower = radial_bins[i]

        upper = radial_bins[i + 1]


        if i == len(radial_bins) - 2:

            region = (
                (radial_frequency >= lower)
                &
                (radial_frequency <= upper)
            )

        else:

            region = (
                (radial_frequency >= lower)
                &
                (radial_frequency < upper)
            )


        energy = float(
            np.sum(
                power[region]
            )
        )


        radial_energy.append(
            energy
        )


        radial_centers.append(
            (lower + upper) / 2.0
        )


    radial_energy = np.asarray(
        radial_energy,
        dtype=np.float64
    )


    radial_centers = np.asarray(
        radial_centers,
        dtype=np.float64
    )


    total = float(
        np.sum(
            radial_energy
        )
    )


    if total > 0:

        radial_probability = (
            radial_energy
            / total
        )

    else:

        radial_probability = np.zeros_like(
            radial_energy
        )


    return (
        radial_centers,
        radial_energy,
        radial_probability
    )


# ================================================================
# ANGULAR SPECTRAL ENERGY S(u)
# ================================================================

def calculate_angular_spectral_energy(
    power,
    FX,
    FY
):

    orientation = (
        np.degrees(
            np.arctan2(
                FY,
                FX
            )
        )
        % 180.0
    )


    angular_bins = np.linspace(
        0.0,
        180.0,
        37
    )


    angular_energy = []

    angular_centers = []


    for i in range(
        len(angular_bins) - 1
    ):

        lower = angular_bins[i]

        upper = angular_bins[i + 1]


        if i == len(angular_bins) - 2:

            region = (
                (orientation >= lower)
                &
                (orientation <= upper)
            )

        else:

            region = (
                (orientation >= lower)
                &
                (orientation < upper)
            )


        energy = float(
            np.sum(
                power[region]
            )
        )


        angular_energy.append(
            energy
        )


        angular_centers.append(
            (lower + upper) / 2.0
        )


    angular_energy = np.asarray(
        angular_energy,
        dtype=np.float64
    )


    angular_centers = np.asarray(
        angular_centers,
        dtype=np.float64
    )


    total = float(
        np.sum(
            angular_energy
        )
    )


    if total > 0:

        angular_probability = (
            angular_energy
            / total
        )

    else:

        angular_probability = np.zeros_like(
            angular_energy
        )


    return (
        angular_centers,
        angular_energy,
        angular_probability
    )


# ================================================================
# MAIN FOURIER TEXTURE FUNCTION
# ================================================================

def calculate_fourier_texture(
    image,
    tumor_mask
):

    extracted = extract_tumor_patch(
        image,
        tumor_mask
    )


    if extracted is None:

        return None


    patch, patch_mask = extracted


    tumor_values = patch[
        patch_mask
    ]


    if len(tumor_values) < 2:

        return None


    tumor_mean = float(
        np.mean(
            tumor_values
        )
    )


    analysis_image = np.array(
        patch,
        dtype=np.float64
    )


    # ------------------------------------------------------------
    # Replace background with tumor mean
    # ------------------------------------------------------------

    analysis_image[
        ~patch_mask
    ] = tumor_mean


    # ------------------------------------------------------------
    # Remove DC / mean component
    # ------------------------------------------------------------

    analysis_image = (
        analysis_image
        - np.mean(
            analysis_image
        )
    )


    # ------------------------------------------------------------
    # 2-D Fourier transform
    # ------------------------------------------------------------

    spectrum_complex = np.fft.fft2(
        analysis_image
    )


    spectrum_shifted = np.fft.fftshift(
        spectrum_complex
    )


    magnitude = np.abs(
        spectrum_shifted
    )


    # ------------------------------------------------------------
    # Power spectrum
    # ------------------------------------------------------------

    power = (
        magnitude ** 2
    )


    center_r = (
        power.shape[0] // 2
    )

    center_c = (
        power.shape[1] // 2
    )


    # Remove DC component
    power[
        center_r,
        center_c
    ] = 0.0


    total_power = float(
        np.sum(
            power
        )
    )


    if total_power <= 0:

        return None


    # ============================================================
    # FREQUENCY COORDINATES
    # ============================================================

    rows, cols = power.shape


    fy = np.fft.fftshift(
        np.fft.fftfreq(
            rows
        )
    )


    fx = np.fft.fftshift(
        np.fft.fftfreq(
            cols
        )
    )


    FX, FY = np.meshgrid(
        fx,
        fy
    )


    radial_frequency = np.sqrt(
        FX ** 2
        +
        FY ** 2
    )


    # ============================================================
    # NORMALIZED SPECTRAL PROBABILITY
    # ============================================================

    spectral_probability = (
        power
        /
        total_power
    )


    # ============================================================
    # SPECTRAL ENERGY CONCENTRATION
    # ============================================================

    spectral_energy = float(
        np.sum(
            spectral_probability ** 2
        )
    )


    # ============================================================
    # SPECTRAL ENTROPY
    # ============================================================

    nonzero_probability = (
        spectral_probability[
            spectral_probability > 0
        ]
    )


    spectral_entropy = float(
        -np.sum(
            nonzero_probability
            *
            np.log2(
                nonzero_probability
            )
        )
    )


    # ============================================================
    # DOMINANT FREQUENCY
    # ============================================================

    max_index = np.unravel_index(
        np.argmax(
            power
        ),
        power.shape
    )


    dominant_frequency = float(
        radial_frequency[
            max_index
        ]
    )


    # ============================================================
    # FUNDAMENTAL SPATIAL PERIOD
    # ============================================================

    if dominant_frequency > 0:

        fundamental_period = float(
            1.0
            /
            dominant_frequency
        )

    else:

        fundamental_period = np.nan


    # ============================================================
    # DOMINANT FOURIER ORIENTATION
    # ============================================================

    dominant_fx = float(
        FX[
            max_index
        ]
    )


    dominant_fy = float(
        FY[
            max_index
        ]
    )


    dominant_orientation = (
        math.degrees(
            math.atan2(
                dominant_fy,
                dominant_fx
            )
        )
        %
        180.0
    )


    # ============================================================
    # S(r)
    # ============================================================

    (
        radial_centers,
        radial_energy,
        radial_probability

    ) = calculate_radial_spectral_energy(
        power,
        radial_frequency
    )


    # ============================================================
    # S(u)
    # ============================================================

    (
        angular_centers,
        angular_energy,
        angular_probability

    ) = calculate_angular_spectral_energy(
        power,
        FX,
        FY
    )


    # ============================================================
    # RADIAL MEAN
    # ============================================================

    radial_mean = float(
        np.sum(
            radial_centers
            *
            radial_probability
        )
    )


    # ============================================================
    # RADIAL VARIANCE
    # ============================================================

    radial_variance = float(
        np.sum(
            (
                radial_centers
                -
                radial_mean
            ) ** 2
            *
            radial_probability
        )
    )


    # ============================================================
    # RADIAL PEAK FREQUENCY
    # ============================================================

    radial_peak_index = int(
        np.argmax(
            radial_energy
        )
    )


    radial_peak_frequency = float(
        radial_centers[
            radial_peak_index
        ]
    )


    # ============================================================
    # ANGULAR MEAN
    # ============================================================

    angular_mean = float(
        np.sum(
            angular_centers
            *
            angular_probability
        )
    )


    # ============================================================
    # ANGULAR VARIANCE
    # ============================================================

    angular_variance = float(
        np.sum(
            (
                angular_centers
                -
                angular_mean
            ) ** 2
            *
            angular_probability
        )
    )


    # ============================================================
    # ANGULAR PEAK ORIENTATION
    # ============================================================

    angular_peak_index = int(
        np.argmax(
            angular_energy
        )
    )


    angular_peak_orientation = float(
        angular_centers[
            angular_peak_index
        ]
    )


    # ============================================================
    # RETURN RESULTS
    # ============================================================

    return {

        "Spectral_Energy":
            spectral_energy,

        "Spectral_Entropy":
            spectral_entropy,

        "Dominant_Frequency":
            dominant_frequency,

        "Fundamental_Period":
            fundamental_period,

        "Dominant_Orientation":
            dominant_orientation,

        "Radial_Mean":
            radial_mean,

        "Radial_Variance":
            radial_variance,

        "Radial_Peak_Frequency":
            radial_peak_frequency,

        "Angular_Mean":
            angular_mean,

        "Angular_Variance":
            angular_variance,

        "Angular_Peak_Orientation":
            angular_peak_orientation,

        "Radial_Frequency":
            radial_centers,

        "Radial_Energy":
            radial_energy,

        "Radial_Probability":
            radial_probability,

        "Angular_Orientation":
            angular_centers,

        "Angular_Energy":
            angular_energy,

        "Angular_Probability":
            angular_probability,

        "Spectrum":
            magnitude,

        "Power":
            power,

        "Patch":
            analysis_image

    }


# ================================================================
# STEP 5 - PROCESS ALL TUMOR SLICES
# ================================================================

print(
    "\nSTEP 5 - PROCESSING TUMOR SLICES"
)

print("=" * 70)


results = []

representative_result = None
representative_slice = None


middle_position = (
    len(tumor_slice_indices)
    // 2
)


for position, slice_index in enumerate(
    tumor_slice_indices
):

    print(
        f"Processing slice {slice_index}"
    )


    image = ct_volume[
        slice_index
    ]


    slice_mask = binary_mask[
        slice_index
    ]


    result = calculate_fourier_texture(
        image,
        slice_mask
    )


    if result is None:

        continue


    results.append({

        "Slice":
            int(slice_index),

        "Spectral_Energy":
            result[
                "Spectral_Energy"
            ],

        "Spectral_Entropy":
            result[
                "Spectral_Entropy"
            ],

        "Dominant_Frequency":
            result[
                "Dominant_Frequency"
            ],

        "Fundamental_Period":
            result[
                "Fundamental_Period"
            ],

        "Dominant_Orientation":
            result[
                "Dominant_Orientation"
            ],

        "Radial_Mean":
            result[
                "Radial_Mean"
            ],

        "Radial_Variance":
            result[
                "Radial_Variance"
            ],

        "Radial_Peak_Frequency":
            result[
                "Radial_Peak_Frequency"
            ],

        "Angular_Mean":
            result[
                "Angular_Mean"
            ],

        "Angular_Variance":
            result[
                "Angular_Variance"
            ],

        "Angular_Peak_Orientation":
            result[
                "Angular_Peak_Orientation"
            ]

    })


    if (
        representative_result is None
        and
        position == middle_position
    ):

        representative_result = result

        representative_slice = int(
            slice_index
        )


# ================================================================
# REPRESENTATIVE SLICE FALLBACK
# ================================================================

if representative_result is None:

    for slice_index in tumor_slice_indices:

        fallback_result = calculate_fourier_texture(
            ct_volume[
                slice_index
            ],
            binary_mask[
                slice_index
            ]
        )

        if fallback_result is not None:

            representative_result = (
                fallback_result
            )

            representative_slice = int(
                slice_index
            )

            break


# ================================================================
# CREATE DATAFRAME
# ================================================================

results_df = pd.DataFrame(
    results
)


if len(results_df) == 0:

    raise ValueError(
        "No Fourier texture results were generated."
    )


print(
    "\nNumber of spectral measurements:",
    len(results_df)
)


# ================================================================
# STEP 6 - OVERALL SPECTRAL FEATURES
# ================================================================

print(
    "\nSTEP 6 - CALCULATING OVERALL SPECTRAL FEATURES"
)

print("=" * 70)


feature_columns = [

    "Spectral_Energy",

    "Spectral_Entropy",

    "Dominant_Frequency",

    "Fundamental_Period",

    "Dominant_Orientation",

    "Radial_Mean",

    "Radial_Variance",

    "Radial_Peak_Frequency",

    "Angular_Mean",

    "Angular_Variance",

    "Angular_Peak_Orientation"

]


overall_features = {}


for feature in feature_columns:

    overall_features[
        feature
    ] = float(
        results_df[
            feature
        ].mean()
    )


    print(
        f"{feature:30s}: "
        f"{overall_features[feature]:.10f}"
    )


# ================================================================
# STEP 7 - FEATURE VARIABILITY
# ================================================================

print(
    "\nSTEP 7 - FEATURE VARIABILITY"
)

print("=" * 70)


for feature in feature_columns:

    mean_value = float(
        results_df[
            feature
        ].mean()
    )


    std_value = float(
        results_df[
            feature
        ].std()
    )


    print(
        f"{feature:30s} "
        f"Mean = {mean_value:.10f} "
        f"SD = {std_value:.10f}"
    )


# ================================================================
# STEP 8 - SAVE DETAILED RESULTS
# ================================================================

print(
    "\nSTEP 8 - SAVING DETAILED RESULTS"
)

print("=" * 70)


detailed_csv = os.path.join(
    OUTPUT_DIR,
    "GTV1_Spectral_Texture.csv"
)


results_df.to_csv(
    detailed_csv,
    index=False
)


print(
    "Saved:",
    detailed_csv
)


# ================================================================
# STEP 9 - SAVE SUMMARY
# ================================================================

print(
    "\nSTEP 9 - SAVING SUMMARY"
)

print("=" * 70)


summary_df = pd.DataFrame({

    "Feature":
        list(
            overall_features.keys()
        ),

    "Mean_Value":
        list(
            overall_features.values()
        )

})


summary_csv = os.path.join(
    OUTPUT_DIR,
    "GTV1_Spectral_summary.csv"
)


summary_df.to_csv(
    summary_csv,
    index=False
)


print(
    "Saved:",
    summary_csv
)


# ================================================================
# STEP 10 - RADIAL SPECTRAL-ENERGY FUNCTION S(r)
# ================================================================

print(
    "\nSTEP 10 - CREATING S(r) RADIAL SPECTRAL-ENERGY PLOT"
)

print("=" * 70)


radial_frequency_plot = (
    representative_result[
        "Radial_Frequency"
    ]
)


radial_probability_plot = (
    representative_result[
        "Radial_Probability"
    ]
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    radial_frequency_plot,
    radial_probability_plot,
    linewidth=2
)


plt.xlabel(
    "Radial frequency (cycles/pixel)"
)


plt.ylabel(
    "Normalized spectral energy"
)


plt.title(
    f"GTV-1 S(r) Spectral Energy | "
    f"Slice {representative_slice}"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


radial_plot = os.path.join(
    OUTPUT_DIR,
    "01_Spectral_Energy_Sr.png"
)


plt.savefig(
    radial_plot,
    dpi=300
)


plt.close()


print(
    "Saved:",
    radial_plot
)


# ================================================================
# STEP 11 - ANGULAR SPECTRAL-ENERGY FUNCTION S(u)
# ================================================================

print(
    "\nSTEP 11 - CREATING S(u) ANGULAR SPECTRAL-ENERGY PLOT"
)

print("=" * 70)


angular_orientation_plot = (
    representative_result[
        "Angular_Orientation"
    ]
)


angular_probability_plot = (
    representative_result[
        "Angular_Probability"
    ]
)


plt.figure(
    figsize=(10, 6)
)


plt.plot(
    angular_orientation_plot,
    angular_probability_plot,
    linewidth=2
)


plt.xlabel(
    "Orientation (degrees)"
)


plt.ylabel(
    "Normalized spectral energy"
)


plt.title(
    f"GTV-1 S(u) Spectral Energy | "
    f"Slice {representative_slice}"
)


plt.xlim(
    0,
    180
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


angular_plot = os.path.join(
    OUTPUT_DIR,
    "02_Spectral_Energy_Su.png"
)


plt.savefig(
    angular_plot,
    dpi=300
)


plt.close()


print(
    "Saved:",
    angular_plot
)


# ================================================================
# STEP 12 - REPRESENTATIVE FOURIER SPECTRUM
# ================================================================

print(
    "\nSTEP 12 - REPRESENTATIVE FOURIER SPECTRUM"
)

print("=" * 70)


spectrum = (
    representative_result[
        "Spectrum"
    ]
)


plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    np.log1p(
        spectrum
    ),
    cmap="gray"
)


plt.colorbar(
    label="log(1 + magnitude)"
)


plt.title(
    f"GTV-1 Fourier Spectrum | "
    f"Slice {representative_slice}"
)


plt.xlabel(
    "Frequency X"
)


plt.ylabel(
    "Frequency Y"
)


plt.tight_layout()


spectrum_plot = os.path.join(
    OUTPUT_DIR,
    "03_Representative_Fourier_Spectrum.png"
)


plt.savefig(
    spectrum_plot,
    dpi=300
)


plt.close()


print(
    "Saved:",
    spectrum_plot
)


# ================================================================
# STEP 13 - REPRESENTATIVE TUMOR PATCH
# ================================================================

print(
    "\nSTEP 13 - REPRESENTATIVE TUMOR PATCH"
)

print("=" * 70)


patch = (
    representative_result[
        "Patch"
    ]
)


plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    patch,
    cmap="gray"
)


plt.colorbar(
    label="Centered intensity"
)


plt.title(
    f"GTV-1 Tumor Patch | "
    f"Slice {representative_slice}"
)


plt.xlabel(
    "X"
)


plt.ylabel(
    "Y"
)


plt.tight_layout()


patch_plot = os.path.join(
    OUTPUT_DIR,
    "04_Representative_Tumor_Patch.png"
)


plt.savefig(
    patch_plot,
    dpi=300
)


plt.close()


print(
    "Saved:",
    patch_plot
)


# ================================================================
# STEP 14 - SAVE REPORT
# ================================================================

print(
    "\nSTEP 14 - SAVING REPORT"
)

print("=" * 70)


report_path = os.path.join(
    OUTPUT_DIR,
    "spectral_texture_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GTV-1 SPECTRAL FOURIER TEXTURE REPORT\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    f.write(
        "Patient: LUNG1-001\n"
    )

    f.write(
        "Series: 69331\n"
    )

    f.write(
        "Segmentation: GTV-1\n\n"
    )


    f.write(
        "METHOD\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    f.write(
        "Tumor-containing CT slices were analyzed "
        "using the 2-D Fourier transform.\n"
    )


    f.write(
        "The tumor region was cropped to its "
        "bounding box.\n"
    )


    f.write(
        "Pixels outside the tumor mask were replaced "
        "with the tumor-region mean.\n"
    )


    f.write(
        "The DC component was removed before Fourier "
        "spectral analysis.\n"
    )


    f.write(
        "The Fourier power spectrum was represented "
        "using radial and angular frequency coordinates.\n\n"
    )


    f.write(
        "SPECTRAL FUNCTIONS\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    f.write(
        "S(r) describes spectral energy as a function "
        "of radial frequency.\n"
    )


    f.write(
        "S(u) describes spectral energy as a function "
        "of orientation.\n\n"
    )


    f.write(
        "DESCRIPTORS\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    f.write(
        "Spectral energy describes concentration of "
        "energy in the normalized Fourier spectrum.\n"
    )


    f.write(
        "Spectral entropy describes the distribution "
        "of spectral energy across frequency components.\n"
    )


    f.write(
        "Dominant frequency is the radial frequency "
        "of the strongest Fourier component.\n"
    )


    f.write(
        "Fundamental spatial period is calculated as "
        "1 / dominant frequency when the frequency "
        "is non-zero.\n"
    )


    f.write(
        "Dominant orientation is the angular location "
        "of the strongest Fourier component.\n"
    )


    f.write(
        "Radial mean and variance describe the "
        "distribution of spectral energy across "
        "radial frequency.\n"
    )


    f.write(
        "Radial peak frequency identifies the radial "
        "frequency band with maximum spectral energy.\n"
    )


    f.write(
        "Angular mean and variance describe the "
        "distribution of spectral energy across "
        "orientation.\n"
    )


    f.write(
        "Angular peak orientation identifies the "
        "orientation band with maximum spectral energy.\n\n"
    )


    f.write(
        "FINAL MEAN FEATURES\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    for feature in feature_columns:

        f.write(
            f"{feature}: "
            f"{overall_features[feature]:.10f}\n"
        )


    f.write(
        "\nREPRESENTATIVE SLICE\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    f.write(
        f"Slice: {representative_slice}\n"
    )


print(
    "Saved:",
    report_path
)


# ================================================================
# STEP 15 - FINAL SUMMARY
# ================================================================

print("\n")

print("=" * 70)

print(
    "STEP 13 - SPECTRAL TEXTURE COMPLETE"
)

print("=" * 70)


print(
    "\nFINAL RESULTS"
)

print(
    "-" * 70
)


for feature in feature_columns:

    print(
        f"{feature:30s}: "
        f"{overall_features[feature]:.6f}"
    )


print(
    "\nFILES"
)


print(
    detailed_csv
)

print(
    summary_csv
)

print(
    radial_plot
)

print(
    angular_plot
)

print(
    spectrum_plot
)

print(
    patch_plot
)

print(
    report_path
)


print("\n")

print("=" * 70)

print(
    "SUCCESS - STEP 13 SPECTRAL TEXTURE"
)

print("=" * 70)
# ================================================================
# STEP 15 - FEATURE STABILITY ANALYSIS
# ================================================================
#
# Boundary perturbation:
#   +1 voxel  -> outward dilation
#   -1 voxel  -> inward erosion
#   +2 voxels -> outward dilation
#   -2 voxels -> inward erosion
#
# The same Fourier-based descriptors are recalculated.
#
# Stability criterion:
#   Mean absolute percentage change <= 10%
#
# Unstable features are excluded from the final stable feature set.
# ================================================================

print("\n")
print("=" * 70)
print("STEP 15 - SPECTRAL FEATURE STABILITY ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------------
# scipy is used only for binary boundary perturbation.
# Fourier and texture descriptors remain explicitly implemented.
# ---------------------------------------------------------------

from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion


# ---------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------

stability_features = [

    "Dominant_Frequency",
    "Fundamental_Period",
    "Dominant_Orientation",
    "Radial_Mean",
    "Radial_Variance",
    "Radial_Peak_Frequency",
    "Angular_Mean",
    "Angular_Variance",
    "Angular_Peak_Orientation"

]


# ---------------------------------------------------------------
# Calculate mean spectral features for a complete 3-D mask
# ---------------------------------------------------------------

def calculate_volume_spectral_features(
    ct_volume_input,
    mask_input
):

    slice_results = []

    tumor_slices = np.where(
        np.any(
            mask_input,
            axis=(1, 2)
        )
    )[0]

    for slice_index in tumor_slices:

        image = ct_volume_input[
            slice_index
        ]

        slice_mask = mask_input[
            slice_index
        ]

        result = calculate_fourier_texture(
            image,
            slice_mask
        )

        if result is None:
            continue

        slice_results.append({

            feature:
                result[feature]

            for feature in stability_features

        })

    if len(slice_results) == 0:
        return None

    df = pd.DataFrame(
        slice_results
    )

    return {

        feature:
            float(
                df[feature].mean()
            )

        for feature in stability_features

    }


# ================================================================
# ORIGINAL FEATURES
# ================================================================

print("\nORIGINAL SEGMENTATION")
print("-" * 70)

original_features = (
    calculate_volume_spectral_features(
        ct_volume,
        binary_mask
    )
)

if original_features is None:

    raise ValueError(
        "Could not calculate original spectral features."
    )


for feature in stability_features:

    print(
        f"{feature:30s}: "
        f"{original_features[feature]:.10f}"
    )


# ================================================================
# CREATE 3-D STRUCTURING ELEMENT
# ================================================================

structure = np.ones(
    (3, 3, 3),
    dtype=bool
)


# ================================================================
# DEFINE PERTURBATIONS
# ================================================================

perturbations = {

    "Original":
        binary_mask,

    "Dilation_1_voxel":
        binary_dilation(
            binary_mask,
            structure=structure,
            iterations=1
        ),

    "Erosion_1_voxel":
        binary_erosion(
            binary_mask,
            structure=structure,
            iterations=1
        ),

    "Dilation_2_voxels":
        binary_dilation(
            binary_mask,
            structure=structure,
            iterations=2
        ),

    "Erosion_2_voxels":
        binary_erosion(
            binary_mask,
            structure=structure,
            iterations=2
        )

}


# ================================================================
# PROCESS PERTURBATIONS
# ================================================================

perturbed_features = {}


for perturbation_name, perturbed_mask in perturbations.items():

    print("\n")
    print(
        f"PROCESSING: {perturbation_name}"
    )
    print("-" * 70)

    voxel_count = int(
        np.sum(perturbed_mask)
    )

    print(
        "Tumor voxels:",
        voxel_count
    )

    if voxel_count == 0:

        print(
            "WARNING: Empty mask. Skipping."
        )

        continue


    feature_result = (
        calculate_volume_spectral_features(
            ct_volume,
            perturbed_mask
        )
    )


    if feature_result is None:

        print(
            "WARNING: No spectral features generated."
        )

        continue


    perturbed_features[
        perturbation_name
    ] = feature_result


    for feature in stability_features:

        print(
            f"{feature:30s}: "
            f"{feature_result[feature]:.10f}"
        )


# ================================================================
# CALCULATE PERCENTAGE CHANGES
# ================================================================

print("\n")
print("=" * 70)
print("CALCULATING FEATURE CHANGES")
print("=" * 70)


stability_rows = []


for feature in stability_features:

    original_value = float(
        original_features[
            feature
        ]
    )


    row = {

        "Feature":
            feature,

        "Original":
            original_value

    }


    absolute_changes = []


    for perturbation_name in [

        "Dilation_1_voxel",
        "Erosion_1_voxel",
        "Dilation_2_voxels",
        "Erosion_2_voxels"

    ]:

        if (
            perturbation_name
            not in perturbed_features
        ):

            row[
                perturbation_name
            ] = np.nan

            row[
                perturbation_name
                + "_ChangePercent"
            ] = np.nan

            continue


        perturbed_value = float(
            perturbed_features[
                perturbation_name
            ][
                feature
            ]
        )


        row[
            perturbation_name
        ] = perturbed_value


        # -------------------------------------------------------
        # Percentage change
        #
        # If original value is zero, absolute difference is used
        # to avoid division by zero.
        # -------------------------------------------------------

        if abs(original_value) > 1e-12:

            percentage_change = (
                abs(
                    perturbed_value
                    - original_value
                )
                /
                abs(original_value)
                * 100.0
            )

        else:

            percentage_change = (
                abs(
                    perturbed_value
                    - original_value
                )
                * 100.0
            )


        row[
            perturbation_name
            + "_ChangePercent"
        ] = percentage_change


        absolute_changes.append(
            percentage_change
        )


    if len(absolute_changes) > 0:

        mean_change = float(
            np.mean(
                absolute_changes
            )
        )

        max_change = float(
            np.max(
                absolute_changes
            )
        )

    else:

        mean_change = np.nan
        max_change = np.nan


    row[
        "Mean_Absolute_Change_Percent"
    ] = mean_change


    row[
        "Maximum_Absolute_Change_Percent"
    ] = max_change


    # ------------------------------------------------------------
    # Stability criterion
    # ------------------------------------------------------------

    if (
        not np.isnan(mean_change)
        and
        mean_change <= 10.0
    ):

        row[
            "Stability"
        ] = "STABLE"

    else:

        row[
            "Stability"
        ] = "UNSTABLE"


    stability_rows.append(
        row
    )


stability_df = pd.DataFrame(
    stability_rows
)


# ================================================================
# PRINT STABILITY RESULTS
# ================================================================

print("\n")
print("=" * 70)
print("FEATURE STABILITY RESULTS")
print("=" * 70)


for _, row in stability_df.iterrows():

    print(
        f"{row['Feature']:30s} "
        f"Mean Change = "
        f"{row['Mean_Absolute_Change_Percent']:.4f}% "
        f"-> {row['Stability']}"
    )


# ================================================================
# SAVE STABILITY CSV
# ================================================================

stability_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_Spectral_Feature_Stability.csv"

)


stability_df.to_csv(

    stability_csv,

    index=False

)


print("\nSaved:")
print(
    stability_csv
)


# ================================================================
# IDENTIFY STABLE FEATURES
# ================================================================

stable_features = (

    stability_df[
        stability_df[
            "Stability"
        ]
        == "STABLE"
    ][
        "Feature"
    ]
    .tolist()

)


unstable_features = (

    stability_df[
        stability_df[
            "Stability"
        ]
        == "UNSTABLE"
    ][
        "Feature"
    ]
    .tolist()

)


# ================================================================
# FINAL STABLE FEATURE SET
# ================================================================

print("\n")
print("=" * 70)
print("FINAL STABLE SPECTRAL FEATURES")
print("=" * 70)


if len(stable_features) == 0:

    print(
        "No spectral features satisfied the stability criterion."
    )

else:

    for feature in stable_features:

        print(
            "KEEP:",
            feature
        )


print("\n")
print("=" * 70)
print("EXCLUDED UNSTABLE FEATURES")
print("=" * 70)


if len(unstable_features) == 0:

    print(
        "None"
    )

else:

    for feature in unstable_features:

        print(
            "EXCLUDE:",
            feature
        )


# ================================================================
# SAVE FINAL STABLE FEATURE SUMMARY
# ================================================================

stable_summary = pd.DataFrame({

    "Stable_Feature":
        stable_features

})


stable_summary_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_Spectral_Stable_Features.csv"

)


stable_summary.to_csv(

    stable_summary_csv,

    index=False

)


print("\nSaved:")
print(
    stable_summary_csv
)


# ================================================================
# SAVE STABILITY REPORT
# ================================================================

stability_report_path = os.path.join(

    OUTPUT_DIR,

    "spectral_feature_stability.txt"

)


with open(

    stability_report_path,

    "w",

    encoding="utf-8"

) as f:


    f.write(
        "GTV-1 SPECTRAL FEATURE STABILITY ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )


    f.write(
        "Patient: LUNG1-001\n"
    )

    f.write(
        "Series: 69331\n"
    )

    f.write(
        "Segmentation: GTV-1\n\n"
    )


    f.write(
        "PURPOSE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Spectral features were tested for sensitivity "
        "to small perturbations of the GTV-1 segmentation boundary.\n\n"
    )


    f.write(
        "PERTURBATIONS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "Dilation by 1 voxel: outward boundary perturbation.\n"
    )

    f.write(
        "Erosion by 1 voxel: inward boundary perturbation.\n"
    )

    f.write(
        "Dilation by 2 voxels: outward boundary perturbation.\n"
    )

    f.write(
        "Erosion by 2 voxels: inward boundary perturbation.\n\n"
    )


    f.write(
        "STABILITY CRITERION\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "A feature is considered STABLE when its "
        "mean absolute percentage change across the "
        "available perturbations is <= 10%.\n\n"
    )


    f.write(
        "FEATURE RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )


    for _, row in stability_df.iterrows():

        f.write(
            f"{row['Feature']}\n"
        )

        f.write(
            f"  Original: "
            f"{row['Original']:.10f}\n"
        )

        f.write(
            f"  Mean absolute change: "
            f"{row['Mean_Absolute_Change_Percent']:.4f}%\n"
        )

        f.write(
            f"  Maximum absolute change: "
            f"{row['Maximum_Absolute_Change_Percent']:.4f}%\n"
        )

        f.write(
            f"  Status: "
            f"{row['Stability']}\n\n"
        )


    f.write(
        "FINAL STABLE FEATURES\n"
    )

    f.write(
        "-" * 70 + "\n"
    )


    if len(stable_features) == 0:

        f.write(
            "None\n"
        )

    else:

        for feature in stable_features:

            f.write(
                f"{feature}\n"
            )


    f.write(
        "\nEXCLUDED FEATURES\n"
    )

    f.write(
        "-" * 70 + "\n"
    )


    if len(unstable_features) == 0:

        f.write(
            "None\n"
        )

    else:

        for feature in unstable_features:

            f.write(
                f"{feature}\n"
            )


print("\nSaved:")
print(
    stability_report_path
)


# ================================================================
# UPDATE FINAL FEATURE SUMMARY
# ================================================================

final_stable_values = {

    feature:
        overall_features[
            feature
        ]

    for feature in stable_features

}


final_stable_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_Spectral_Final_Stable_Features.csv"

)


pd.DataFrame({

    "Feature":
        list(
            final_stable_values.keys()
        ),

    "Original_Mean_Value":
        list(
            final_stable_values.values()
        )

}).to_csv(

    final_stable_csv,

    index=False

)


print("\nSaved:")
print(
    final_stable_csv
)


# ================================================================
# STEP 15 COMPLETE
# ================================================================

print("\n")
print("=" * 70)
print("STEP 15 - SPECTRAL FEATURE STABILITY COMPLETE")
print("=" * 70)

print(
    "\nStable features:",
    len(stable_features)
)

print(
    "Excluded unstable features:",
    len(unstable_features)
)

print(
    "\nStability criterion: mean absolute change <= 10%"
)

print("\n")
print("=" * 70)
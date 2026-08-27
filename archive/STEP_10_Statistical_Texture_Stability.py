# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 10 - STATISTICAL TEXTURE FEATURE STABILITY
#
# Statistical texture descriptors based on the
# statistical texture measures described in:
#
# Gonzalez & Woods - Digital Image Processing
#
# Features:
#   1. Mean
#   2. Variance
#   3. Smoothness
#   4. Third Moment
#   5. Uniformity
#   6. Entropy
#
# Stability analysis:
#   Original segmentation
#   Dilation by 1 voxel
#   Erosion by 1 voxel
#   Dilation by 2 voxels
#   Erosion by 2 voxels
#
# Stability criterion:
#   Mean Absolute Percentage Change <= 10% -> STABLE
#   Mean Absolute Percentage Change > 10%  -> UNSTABLE
#
# ================================================================

import os
import glob

import numpy as np
import pandas as pd
import pydicom
import nrrd

from scipy.ndimage import (
    binary_dilation,
    binary_erosion
)


# ================================================================
# PATHS
# ================================================================

BASE_DIR = (
    r"C:\Users\CeCe\Downloads\nsclc_radiomics"
    r"\LUNG1-001\69331"
)

MASK_PATH = os.path.join(
    BASE_DIR,
    "GTV1_MASK.nrrd"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_10_STATISTICAL_TEXTURE"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# PARAMETERS
# ================================================================

# Stability threshold
STABILITY_THRESHOLD = 10.0

# Number of histogram bins
# A fixed number of bins is used consistently
# for the statistical texture analysis.
NUMBER_OF_BINS = 32


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 10 - STATISTICAL TEXTURE FEATURE STABILITY")
print("=" * 70)


# ================================================================
# STEP 1 - READING CT
# ================================================================

print("\nSTEP 1 - READING CT")
print("=" * 70)

ct_dir = os.path.join(
    BASE_DIR,
    "82046"
)

ct_files = glob.glob(
    os.path.join(
        ct_dir,
        "*.dcm"
    )
)

if len(ct_files) == 0:

    raise FileNotFoundError(
        f"No CT DICOM files found in:\n{ct_dir}"
    )


# ---------------------------------------------------------------
# Sort CT slices using InstanceNumber
# ---------------------------------------------------------------

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


# ---------------------------------------------------------------
# Read CT images and convert to Hounsfield Units
# ---------------------------------------------------------------

ct_slices = []

for path in ct_files:

    ds = pydicom.dcmread(
        path
    )

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
        +
        intercept
    )

    ct_slices.append(
        hu
    )


ct_volume = np.stack(
    ct_slices,
    axis=0
)


print(
    "CT files found:",
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


mask_data, mask_header = nrrd.read(
    MASK_PATH
)


print(
    "Original mask shape:",
    mask_data.shape
)


mask = np.asarray(
    mask_data
)


# ---------------------------------------------------------------
# NRRD orientation:
#
#     (X,Y,Z)
#
# CT volume:
#
#     (Z,Y,X)
#
# Therefore convert the mask to CT orientation.
# ---------------------------------------------------------------

if mask.shape == (
    ct_volume.shape[1],
    ct_volume.shape[2],
    ct_volume.shape[0]
):

    mask = np.transpose(
        mask,
        (2, 1, 0)
    )

elif mask.shape != ct_volume.shape:

    raise ValueError(
        f"Mask shape {mask.shape} does not match "
        f"CT shape {ct_volume.shape}"
    )


# Convert to binary mask
mask = (
    mask > 0
)


original_voxel_count = int(
    np.sum(mask)
)


print(
    "Converted mask shape:",
    mask.shape
)

print(
    "Original tumor voxels:",
    original_voxel_count
)


# ================================================================
# STEP 3 - DEFINING STATISTICAL TEXTURE FEATURES
# ================================================================

print(
    "\nSTEP 3 - DEFINING STATISTICAL TEXTURE FEATURES"
)

print("=" * 70)


def calculate_statistical_features(
    ct_volume,
    binary_mask
):

    # ------------------------------------------------------------
    # Extract CT intensities inside the segmentation
    # ------------------------------------------------------------

    intensities = ct_volume[
        binary_mask
    ].astype(
        np.float64
    )


    if intensities.size == 0:

        raise ValueError(
            "Segmentation contains no voxels."
        )


    # ============================================================
    # 1. MEAN
    #
    # Mean gray level:
    #
    #     m = sum(z * p(z))
    #
    # ============================================================

    mean_value = np.mean(
        intensities
    )


    # ============================================================
    # 2. VARIANCE
    #
    #     sigma^2 = sum((z-m)^2 * p(z))
    #
    # ============================================================

    variance_value = np.var(
        intensities
    )


    # ============================================================
    # 3. SMOOTHNESS
    #
    # Gonzalez & Woods:
    #
    #     R = 1 - 1/(1 + sigma^2)
    #
    # ============================================================

    smoothness_value = (
        1.0
        -
        1.0 /
        (
            1.0
            +
            variance_value
        )
    )


    # ============================================================
    # 4. THIRD MOMENT
    #
    # Central third moment:
    #
    #     mu_3 = sum((z-m)^3 * p(z))
    #
    # The voxel-based calculation below is mathematically
    # equivalent to the probability formulation.
    #
    # ============================================================

    third_moment_value = np.mean(
        (
            intensities
            -
            mean_value
        ) ** 3
    )


    # ============================================================
    # 5. HISTOGRAM
    #
    # Histogram is used to estimate p(z).
    #
    # ============================================================

    min_value = np.min(
        intensities
    )

    max_value = np.max(
        intensities
    )


    # ------------------------------------------------------------
    # Constant-intensity case
    # ------------------------------------------------------------

    if max_value == min_value:

        uniformity_value = 1.0

        entropy_value = 0.0


    else:

        histogram, bin_edges = np.histogram(
            intensities,
            bins=NUMBER_OF_BINS,
            range=(
                min_value,
                max_value
            )
        )


        # --------------------------------------------------------
        # Convert histogram counts to probabilities
        # --------------------------------------------------------

        total_count = np.sum(
            histogram
        )


        probabilities = (
            histogram.astype(
                np.float64
            )
            /
            total_count
        )


        # ========================================================
        # 5. UNIFORMITY
        #
        #     U = sum(p(z)^2)
        #
        # ========================================================

        uniformity_value = np.sum(
            probabilities ** 2
        )


        # ========================================================
        # 6. ENTROPY
        #
        #     H = -sum(p(z) log2 p(z))
        #
        # Zero-probability bins are excluded.
        #
        # ========================================================

        nonzero_probabilities = (
            probabilities[
                probabilities > 0
            ]
        )


        entropy_value = -np.sum(
            nonzero_probabilities
            *
            np.log2(
                nonzero_probabilities
            )
        )


    return {

        "Mean":
            float(
                mean_value
            ),

        "Variance":
            float(
                variance_value
            ),

        "Smoothness":
            float(
                smoothness_value
            ),

        "Third_Moment":
            float(
                third_moment_value
            ),

        "Uniformity":
            float(
                uniformity_value
            ),

        "Entropy":
            float(
                entropy_value
            )
    }


# ================================================================
# STEP 4 - SEGMENTATION PERTURBATIONS
# ================================================================

print(
    "\nSTEP 4 - CREATING SEGMENTATION PERTURBATIONS"
)

print("=" * 70)


# ---------------------------------------------------------------
# Important:
#
# These are 3-D binary morphological perturbations.
#
# The scipy default structuring element uses connectivity = 1,
# corresponding to face-connected neighboring voxels.
#
# ---------------------------------------------------------------

perturbations = {

    "Original":
        mask,

    "Dilation_1_voxel":
        binary_dilation(
            mask,
            iterations=1
        ),

    "Erosion_1_voxel":
        binary_erosion(
            mask,
            iterations=1
        ),

    "Dilation_2_voxels":
        binary_dilation(
            mask,
            iterations=2
        ),

    "Erosion_2_voxels":
        binary_erosion(
            mask,
            iterations=2
        )
}


# ================================================================
# STEP 5 - CALCULATING FEATURES
# ================================================================

print(
    "\nSTEP 5 - CALCULATING STATISTICAL FEATURES"
)

print("=" * 70)


all_results = []


for segmentation_name, current_mask in perturbations.items():

    print(
        f"\nPROCESSING: {segmentation_name}"
    )


    voxel_count = int(
        np.sum(
            current_mask
        )
    )


    print(
        "Tumor voxels:",
        voxel_count
    )


    features = calculate_statistical_features(
        ct_volume,
        current_mask
    )


    row = {

        "Segmentation":
            segmentation_name,

        "Tumor_Voxels":
            voxel_count
    }


    row.update(
        features
    )


    all_results.append(
        row
    )


    for feature_name, value in features.items():

        print(
            f"{feature_name:22s}: "
            f"{value:.10f}"
        )


# ================================================================
# STEP 6 - CREATING RESULTS TABLE
# ================================================================

print(
    "\nSTEP 6 - CREATING RESULTS TABLE"
)

print("=" * 70)


results_df = pd.DataFrame(
    all_results
)


print(
    results_df.to_string(
        index=False
    )
)


# ================================================================
# STEP 7 - FEATURE STABILITY ANALYSIS
# ================================================================

print(
    "\nSTEP 7 - CALCULATING FEATURE STABILITY"
)

print("=" * 70)


feature_columns = [

    "Mean",
    "Variance",
    "Smoothness",
    "Third_Moment",
    "Uniformity",
    "Entropy"

]


# ---------------------------------------------------------------
# Obtain original segmentation values
# ---------------------------------------------------------------

original_row = results_df[
    results_df[
        "Segmentation"
    ]
    ==
    "Original"
].iloc[0]


stability_results = []


perturbed_segmentation_names = [

    "Dilation_1_voxel",
    "Erosion_1_voxel",
    "Dilation_2_voxels",
    "Erosion_2_voxels"

]


for feature_name in feature_columns:


    original_value = float(
        original_row[
            feature_name
        ]
    )


    percentage_changes = []


    # ------------------------------------------------------------
    # Compare every perturbation against ORIGINAL segmentation
    # ------------------------------------------------------------

    for segmentation_name in perturbed_segmentation_names:


        perturbed_row = results_df[
            results_df[
                "Segmentation"
            ]
            ==
            segmentation_name
        ].iloc[0]


        perturbed_value = float(
            perturbed_row[
                feature_name
            ]
        )


        # ========================================================
        # Absolute percentage change
        #
        #     |new - original|
        #     ---------------- × 100
        #        |original|
        #
        # ========================================================

        if abs(
            original_value
        ) < 1e-12:

            if abs(
                perturbed_value
            ) < 1e-12:

                percentage_change = 0.0

            else:

                percentage_change = 100.0

        else:

            percentage_change = (

                abs(
                    perturbed_value
                    -
                    original_value
                )
                /
                abs(
                    original_value
                )
                *
                100.0
            )


        percentage_changes.append(
            percentage_change
        )


    # ============================================================
    # Mean absolute percentage change
    # ============================================================

    mean_absolute_change = np.mean(
        percentage_changes
    )


    maximum_change = np.max(
        percentage_changes
    )


    # ============================================================
    # Stability classification
    # ============================================================

    if (
        mean_absolute_change
        <=
        STABILITY_THRESHOLD
    ):

        status = "STABLE"

    else:

        status = "UNSTABLE"


    stability_results.append({

        "Feature":
            feature_name,

        "Original_Value":
            original_value,

        "Dilation_1_%":
            percentage_changes[0],

        "Erosion_1_%":
            percentage_changes[1],

        "Dilation_2_%":
            percentage_changes[2],

        "Erosion_2_%":
            percentage_changes[3],

        "Mean_Absolute_Change_%":
            mean_absolute_change,

        "Maximum_Change_%":
            maximum_change,

        "Status":
            status

    })


    print(

        f"{feature_name:22s} "
        f"Mean Absolute Change = "
        f"{mean_absolute_change:.4f}% "
        f"-> {status}"

    )


# ================================================================
# STEP 8 - SAVE STABILITY TABLE
# ================================================================

print(
    "\nSTEP 8 - SAVING STABILITY TABLE"
)

print("=" * 70)


stability_df = pd.DataFrame(
    stability_results
)


stability_csv_path = os.path.join(

    OUTPUT_DIR,

    "GTV1_Statistical_Feature_Stability.csv"

)


stability_df.to_csv(

    stability_csv_path,

    index=False

)


print(
    "Saved:",
    stability_csv_path
)


# ================================================================
# STEP 9 - IDENTIFY STABLE / UNSTABLE FEATURES
# ================================================================

stable_features = stability_df[
    stability_df[
        "Status"
    ]
    ==
    "STABLE"
][
    "Feature"
].tolist()


unstable_features = stability_df[
    stability_df[
        "Status"
    ]
    ==
    "UNSTABLE"
][
    "Feature"
].tolist()


print("\n")
print("=" * 70)
print("STATISTICAL TEXTURE FEATURE STABILITY RESULTS")
print("=" * 70)


print(
    "\nStability criterion:"
)

print(
    f"Mean absolute percentage change <= "
    f"{STABILITY_THRESHOLD}% -> STABLE"
)

print(
    f"Mean absolute percentage change > "
    f"{STABILITY_THRESHOLD}% -> UNSTABLE"
)


print(
    "\nStable features:",
    len(stable_features)
)


print(
    "Unstable features:",
    len(unstable_features)
)


# ================================================================
# FINAL STABLE FEATURES
# ================================================================

print(
    "\nFINAL STABLE FEATURES"
)

print("-" * 70)


for feature_name in stable_features:

    print(
        "KEEP:",
        feature_name
    )


# ================================================================
# EXCLUDED UNSTABLE FEATURES
# ================================================================

print(
    "\nEXCLUDED UNSTABLE FEATURES"
)

print("-" * 70)


for feature_name in unstable_features:

    print(
        "EXCLUDE:",
        feature_name
    )


# ================================================================
# STEP 10 - SAVE FINAL STABLE FEATURES
# ================================================================

final_stable_df = stability_df[
    stability_df[
        "Status"
    ]
    ==
    "STABLE"
][
    [
        "Feature",
        "Original_Value",
        "Mean_Absolute_Change_%",
        "Status"
    ]
]


final_stable_path = os.path.join(

    OUTPUT_DIR,

    "GTV1_Statistical_Final_Stable_Features.csv"

)


final_stable_df.to_csv(

    final_stable_path,

    index=False

)


print(
    "\nSaved:",
    final_stable_path
)


# ================================================================
# STEP 11 - SAVE DETAILED REPORT
# ================================================================

report_path = os.path.join(

    OUTPUT_DIR,

    "statistical_texture_stability_report.txt"

)


with open(

    report_path,

    "w",

    encoding="utf-8"

) as f:


    f.write(
        "GTV-1 STATISTICAL TEXTURE FEATURE STABILITY REPORT\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 10 - STATISTICAL TEXTURE FEATURE STABILITY\n\n"
    )


    f.write(
        "Patient: LUNG1-001\n"
    )

    f.write(
        "Series: 69331\n\n"
    )


    # ------------------------------------------------------------
    # CT information
    # ------------------------------------------------------------

    f.write(
        "CT INFORMATION\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        f"Number of CT slices: "
        f"{len(ct_files)}\n"
    )

    f.write(
        f"CT volume shape (Z,Y,X): "
        f"{ct_volume.shape}\n\n"
    )


    # ------------------------------------------------------------
    # Mask information
    # ------------------------------------------------------------

    f.write(
        "SEGMENTATION INFORMATION\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        f"Original tumor voxels: "
        f"{original_voxel_count}\n\n"
    )


    # ------------------------------------------------------------
    # Features
    # ------------------------------------------------------------

    f.write(
        "STATISTICAL TEXTURE FEATURES\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        "1. Mean\n"
    )

    f.write(
        "2. Variance\n"
    )

    f.write(
        "3. Smoothness\n"
    )

    f.write(
        "4. Third Moment\n"
    )

    f.write(
        "5. Uniformity\n"
    )

    f.write(
        "6. Entropy\n\n"
    )


    # ------------------------------------------------------------
    # Perturbations
    # ------------------------------------------------------------

    f.write(
        "SEGMENTATION PERTURBATIONS\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        "1. Original segmentation\n"
    )

    f.write(
        "2. Dilation by 1 voxel\n"
    )

    f.write(
        "3. Erosion by 1 voxel\n"
    )

    f.write(
        "4. Dilation by 2 voxels\n"
    )

    f.write(
        "5. Erosion by 2 voxels\n\n"
    )


    # ------------------------------------------------------------
    # Morphological connectivity
    # ------------------------------------------------------------

    f.write(
        "MORPHOLOGICAL CONNECTIVITY\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        "3-D binary morphology was used with the "
        "default scipy connectivity = 1,\n"
    )

    f.write(
        "corresponding to face-connected neighboring voxels.\n\n"
    )


    # ------------------------------------------------------------
    # Stability criterion
    # ------------------------------------------------------------

    f.write(
        "STABILITY CRITERION\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        "Mean absolute percentage change <= 10%: STABLE\n"
    )

    f.write(
        "Mean absolute percentage change > 10%: UNSTABLE\n\n"
    )


    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    f.write(
        "FEATURE STABILITY RESULTS\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    for _, row in stability_df.iterrows():

        f.write(

            f"{row['Feature']}: "
            f"Mean Absolute Change = "
            f"{row['Mean_Absolute_Change_%']:.6f}% "
            f"-> {row['Status']}\n"

        )


    f.write("\n")


    # ------------------------------------------------------------
    # Stable features
    # ------------------------------------------------------------

    f.write(
        "FINAL STABLE FEATURES\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    for feature_name in stable_features:

        f.write(
            f"KEEP: {feature_name}\n"
        )


    f.write("\n")


    # ------------------------------------------------------------
    # Unstable features
    # ------------------------------------------------------------

    f.write(
        "EXCLUDED UNSTABLE FEATURES\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    for feature_name in unstable_features:

        f.write(
            f"EXCLUDE: {feature_name}\n"
        )


    f.write("\n")


    # ------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------

    f.write(
        "OUTPUT FILES\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        f"{stability_csv_path}\n"
    )

    f.write(
        f"{final_stable_path}\n"
    )

    f.write(
        f"{report_path}\n"
    )


print(
    "Saved:",
    report_path
)


# ================================================================
# STEP 12 - SAVE ALL RESULTS
# ================================================================

all_results_path = os.path.join(

    OUTPUT_DIR,

    "GTV1_Statistical_Texture_Stability_All_Results.csv"

)


results_df.to_csv(

    all_results_path,

    index=False

)


print(
    "Saved:",
    all_results_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("STEP 10 - STATISTICAL TEXTURE STABILITY COMPLETE")
print("=" * 70)


print(
    "\nStability criterion:"
)

print(
    "Mean absolute percentage change <= "
    f"{STABILITY_THRESHOLD}% -> STABLE"
)

print(
    "Mean absolute percentage change > "
    f"{STABILITY_THRESHOLD}% -> UNSTABLE"
)


print(
    "\nStable features:",
    len(stable_features)
)


print(
    "Excluded unstable features:",
    len(unstable_features)
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print("\n")
print("=" * 70)
print("SUCCESS")
print("=" * 70)
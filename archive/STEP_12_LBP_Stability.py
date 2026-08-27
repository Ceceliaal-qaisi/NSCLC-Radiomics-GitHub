
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 12 - LOCAL BINARY PATTERNS (LBP) FEATURE STABILITY
#
# From scratch:
#   - LBP calculation
#   - LBP histogram
#   - Mean LBP
#   - Variance of LBP
#   - Uniformity
#   - Entropy
#   - Segmentation perturbations
#   - Feature stability analysis
#
# Stability criterion:
#   Mean Absolute Change <= 10%  -> STABLE -> KEEP
#   Mean Absolute Change >  10%  -> UNSTABLE -> EXCLUDE
# ================================================================

import os
import glob
import numpy as np
import pandas as pd
import pydicom
import nrrd

from scipy.ndimage import binary_dilation, binary_erosion


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
    "STEP_12_LBP_TEXTURE"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# PARAMETERS
# ================================================================

P = 8
R = 1

NUMBER_OF_BINS = 256

STABILITY_THRESHOLD = 10.0


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("PROJECT 7 - RADIOMICS")
print("STEP 12 - LBP FEATURE STABILITY ANALYSIS")
print("=" * 80)


# ================================================================
# STEP 1 - READING CT
# ================================================================

print("\nSTEP 1 - READING CT")
print("=" * 80)


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
print("=" * 80)


if not os.path.exists(
    MASK_FILE
):

    raise FileNotFoundError(
        f"GTV-1 mask not found:\n{MASK_FILE}"
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

print(
    "Mask dtype:",
    mask.dtype
)


# NRRD = (X,Y,Z)
# CT   = (Z,Y,X)

expected_nrrd_shape = (
    ct_volume.shape[2],
    ct_volume.shape[1],
    ct_volume.shape[0]
)


if mask.shape == expected_nrrd_shape:

    mask = np.transpose(
        mask,
        (2, 1, 0)
    )

elif mask.shape != ct_volume.shape:

    raise ValueError(
        "\nCT and mask dimensions do not match.\n"
        f"CT shape   : {ct_volume.shape}\n"
        f"Mask shape : {mask.shape}"
    )


binary_mask = (
    mask > 0
)


original_tumor_voxels = int(
    np.sum(
        binary_mask
    )
)


print(
    "Converted mask shape:",
    binary_mask.shape
)

print(
    "Original tumor voxels:",
    original_tumor_voxels
)


if original_tumor_voxels == 0:

    raise ValueError(
        "GTV-1 mask is empty."
    )


# ================================================================
# STEP 3 - LBP FROM SCRATCH
# ================================================================

print("\nSTEP 3 - DEFINING LBP FROM SCRATCH")
print("=" * 80)


def calculate_lbp(
    image,
    tumor_mask
):

    rows, cols = image.shape


    lbp_image = np.full(
        (rows, cols),
        -1,
        dtype=np.int32
    )


    # ------------------------------------------------------------
    # Freeman-style 8-neighbour clockwise ordering
    #
    #       0 1 2
    #       7 C 3
    #       6 5 4
    #
    # P = 8
    # R = 1 pixel
    # ------------------------------------------------------------

    for r in range(
        R,
        rows - R
    ):

        for c in range(
            R,
            cols - R
        ):

            if not tumor_mask[r, c]:

                continue


            center = image[r, c]


            neighbors = [

                image[r - 1, c - 1],  # 0

                image[r - 1, c],      # 1

                image[r - 1, c + 1],  # 2

                image[r, c + 1],      # 3

                image[r + 1, c + 1],  # 4

                image[r + 1, c],      # 5

                image[r + 1, c - 1],  # 6

                image[r, c - 1]       # 7

            ]


            code = 0


            for p in range(P):

                if neighbors[p] >= center:

                    code += (
                        2 ** p
                    )


            lbp_image[r, c] = code


    return lbp_image


# ================================================================
# STEP 4 - LBP FEATURE EXTRACTION
# ================================================================

print("\nSTEP 4 - DEFINING LBP FEATURE EXTRACTION")
print("=" * 80)


def calculate_lbp_features(
    ct_volume,
    segmentation
):

    tumor_slice_indices = np.where(
        np.any(
            segmentation,
            axis=(1, 2)
        )
    )[0]


    all_lbp_values = []


    for slice_index in tumor_slice_indices:

        image = ct_volume[
            slice_index
        ]

        slice_mask = segmentation[
            slice_index
        ]


        lbp_image = calculate_lbp(
            image,
            slice_mask
        )


        valid_lbp = lbp_image[
            lbp_image >= 0
        ]


        if len(valid_lbp) == 0:

            continue


        all_lbp_values.extend(
            valid_lbp.tolist()
        )


    if len(all_lbp_values) == 0:

        raise ValueError(
            "No valid LBP pixels found."
        )


    all_lbp_values = np.asarray(
        all_lbp_values,
        dtype=np.int32
    )


    # ------------------------------------------------------------
    # LBP HISTOGRAM
    # ------------------------------------------------------------

    histogram = np.bincount(
        all_lbp_values,
        minlength=NUMBER_OF_BINS
    )


    probability = (
        histogram.astype(
            np.float64
        )
        /
        np.sum(histogram)
    )


    lbp_values = np.arange(
        NUMBER_OF_BINS
    )


    # ------------------------------------------------------------
    # LBP MEAN
    # ------------------------------------------------------------

    lbp_mean = np.sum(
        lbp_values
        *
        probability
    )


    # ------------------------------------------------------------
    # LBP VARIANCE
    # ------------------------------------------------------------

    lbp_variance = np.sum(
        (
            lbp_values
            -
            lbp_mean
        ) ** 2
        *
        probability
    )


    # ------------------------------------------------------------
    # LBP UNIFORMITY
    # ------------------------------------------------------------

    lbp_uniformity = np.sum(
        probability ** 2
    )


    # ------------------------------------------------------------
    # LBP ENTROPY
    # ------------------------------------------------------------

    nonzero_probability = probability[
        probability > 0
    ]


    lbp_entropy = -np.sum(
        nonzero_probability
        *
        np.log2(
            nonzero_probability
        )
    )


    return {

        "Tumor_Voxels":
            int(
                np.sum(
                    segmentation
                )
            ),

        "LBP_Pixels":
            int(
                len(all_lbp_values)
            ),

        "LBP_Mean":
            float(
                lbp_mean
            ),

        "LBP_Variance":
            float(
                lbp_variance
            ),

        "LBP_Uniformity":
            float(
                lbp_uniformity
            ),

        "LBP_Entropy":
            float(
                lbp_entropy
            )

    }


# ================================================================
# STEP 5 - SEGMENTATION PERTURBATIONS
# ================================================================

print("\nSTEP 5 - CREATING SEGMENTATION PERTURBATIONS")
print("=" * 80)


# 3D connectivity:
# every voxel has its 26 neighbouring voxels

structure = np.ones(
    (3, 3, 3),
    dtype=bool
)


segmentations = {

    "Original":
        binary_mask.copy(),

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
# STEP 6 - CALCULATE FEATURES
# ================================================================

print("\nSTEP 6 - CALCULATING LBP FEATURES")
print("=" * 80)


results = []


for name, segmentation in segmentations.items():

    print(
        f"\nPROCESSING: {name}"
    )


    features = calculate_lbp_features(
        ct_volume,
        segmentation
    )


    print(
        "Tumor voxels:",
        features["Tumor_Voxels"]
    )

    print(
        "LBP pixels:",
        features["LBP_Pixels"]
    )

    print(
        f"LBP Mean       : "
        f"{features['LBP_Mean']:.10f}"
    )

    print(
        f"LBP Variance   : "
        f"{features['LBP_Variance']:.10f}"
    )

    print(
        f"LBP Uniformity : "
        f"{features['LBP_Uniformity']:.10f}"
    )

    print(
        f"LBP Entropy    : "
        f"{features['LBP_Entropy']:.10f}"
    )


    results.append({

        "Segmentation":
            name,

        **features

    })


results_df = pd.DataFrame(
    results
)


# ================================================================
# STEP 7 - FEATURE STABILITY
# ================================================================

print("\nSTEP 7 - CALCULATING FEATURE STABILITY")
print("=" * 80)


features_to_test = [

    "LBP_Mean",

    "LBP_Variance",

    "LBP_Uniformity",

    "LBP_Entropy"

]


original_row = results_df[
    results_df[
        "Segmentation"
    ]
    ==
    "Original"
].iloc[0]


stability_results = []


perturbed_names = [

    "Dilation_1_voxel",
    "Erosion_1_voxel",
    "Dilation_2_voxels",
    "Erosion_2_voxels"

]


for feature in features_to_test:

    original_value = float(
        original_row[
            feature
        ]
    )


    percentage_changes = []


    for segmentation_name in perturbed_names:

        perturbed_row = results_df[
            results_df[
                "Segmentation"
            ]
            ==
            segmentation_name
        ].iloc[0]


        perturbed_value = float(
            perturbed_row[
                feature
            ]
        )


        # --------------------------------------------------------
        # Absolute percentage change
        # --------------------------------------------------------

        if abs(original_value) > 1e-12:

            change = (

                abs(
                    perturbed_value
                    -
                    original_value
                )
                /
                abs(original_value)

            ) * 100.0

        else:

            if abs(perturbed_value) <= 1e-12:

                change = 0.0

            else:

                change = 100.0


        percentage_changes.append(
            float(change)
        )


    mean_change = float(
        np.mean(
            percentage_changes
        )
    )


    max_change = float(
        np.max(
            percentage_changes
        )
    )


    if mean_change <= STABILITY_THRESHOLD:

        stability = "STABLE"

    else:

        stability = "UNSTABLE"


    # ------------------------------------------------------------
    # IMPORTANT:
    # Column names are deliberately standardized
    # for STEP 14 integration.
    # ------------------------------------------------------------

    stability_results.append({

        "Feature":
            feature,

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
            mean_change,

        "Maximum_Change_%":
            max_change,

        "Stability":
            stability,

        "Decision":
            (
                "KEEP"
                if stability == "STABLE"
                else "EXCLUDE"
            )

    })


    print(

        f"{feature:20s} "
        f"Mean Absolute Change = "
        f"{mean_change:.4f}% "
        f"-> {stability}"

    )


stability_df = pd.DataFrame(
    stability_results
)


# ================================================================
# STEP 8 - SAVE ALL RESULTS
# ================================================================

print("\nSTEP 8 - SAVING STABILITY RESULTS")
print("=" * 80)


all_results_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_LBP_Stability_All_Results.csv"

)


results_df.to_csv(

    all_results_csv,

    index=False

)


print(
    "Saved:",
    all_results_csv
)


stability_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_LBP_Feature_Stability.csv"

)


stability_df.to_csv(

    stability_csv,

    index=False

)


print(
    "Saved:",
    stability_csv
)


# ================================================================
# STEP 9 - IDENTIFY STABLE / UNSTABLE FEATURES
# ================================================================

print("\nSTEP 9 - IDENTIFYING STABLE FEATURES")
print("=" * 80)


stable_features = stability_df[
    stability_df[
        "Stability"
    ]
    ==
    "STABLE"
][
    "Feature"
].tolist()


unstable_features = stability_df[
    stability_df[
        "Stability"
    ]
    ==
    "UNSTABLE"
][
    "Feature"
].tolist()


# ---------------------------------------------------------------
# STABLE FEATURES FILE
# ---------------------------------------------------------------

stable_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_LBP_Stable_Features.csv"

)


pd.DataFrame({

    "Stable_Feature":
        stable_features

}).to_csv(

    stable_csv,

    index=False

)


# ---------------------------------------------------------------
# FINAL STABLE FEATURES FILE
# ---------------------------------------------------------------

final_stable_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_LBP_Final_Stable_Features.csv"

)


pd.DataFrame({

    "Final_Stable_Feature":
        stable_features

}).to_csv(

    final_stable_csv,

    index=False

)


print(
    "Saved:",
    stable_csv
)

print(
    "Saved:",
    final_stable_csv
)


# ================================================================
# STEP 10 - EVIDENCE-BASED REPORT
# ================================================================

print("\nSTEP 10 - SAVING EVIDENCE-BASED STABILITY REPORT")
print("=" * 80)


report_path = os.path.join(

    OUTPUT_DIR,

    "lbp_feature_stability_report.txt"

)


with open(

    report_path,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 12 - LBP FEATURE STABILITY ANALYSIS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
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
        "LBP PARAMETERS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"P (neighbors): {P}\n"
    )

    f.write(
        f"R (radius): {R}\n"
    )

    f.write(
        f"Number of bins: {NUMBER_OF_BINS}\n\n"
    )


    f.write(
        "SEGMENTATION PERTURBATIONS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Original segmentation\n"
    )

    f.write(
        "Dilation by 1 voxel\n"
    )

    f.write(
        "Erosion by 1 voxel\n"
    )

    f.write(
        "Dilation by 2 voxels\n"
    )

    f.write(
        "Erosion by 2 voxels\n\n"
    )


    f.write(
        "STABILITY CRITERION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Mean absolute percentage change <= 10% "
        "-> STABLE -> KEEP\n"
    )

    f.write(
        "Mean absolute percentage change > 10% "
        "-> UNSTABLE -> EXCLUDE\n\n"
    )


    f.write(
        "FEATURE STABILITY EVIDENCE\n"
    )

    f.write(
        "-" * 80 + "\n"
    )


    for _, row in stability_df.iterrows():

        f.write(
            f"\nFeature: {row['Feature']}\n"
        )

        f.write(
            f"Original value: "
            f"{row['Original_Value']:.10f}\n"
        )

        f.write(
            f"Dilation 1 voxel: "
            f"{row['Dilation_1_%']:.6f}% change\n"
        )

        f.write(
            f"Erosion 1 voxel: "
            f"{row['Erosion_1_%']:.6f}% change\n"
        )

        f.write(
            f"Dilation 2 voxels: "
            f"{row['Dilation_2_%']:.6f}% change\n"
        )

        f.write(
            f"Erosion 2 voxels: "
            f"{row['Erosion_2_%']:.6f}% change\n"
        )

        f.write(
            f"Mean absolute change: "
            f"{row['Mean_Absolute_Change_%']:.6f}%\n"
        )

        f.write(
            f"Maximum absolute change: "
            f"{row['Maximum_Change_%']:.6f}%\n"
        )

        f.write(
            f"Stability: "
            f"{row['Stability']}\n"
        )

        f.write(
            f"Decision: "
            f"{row['Decision']}\n"
        )


    f.write(
        "\n\nFINAL STABLE FEATURES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )


    if len(stable_features) == 0:

        f.write(
            "NONE\n"
        )

    else:

        for feature in stable_features:

            row = stability_df[
                stability_df[
                    "Feature"
                ]
                ==
                feature
            ].iloc[0]


            f.write(

                f"KEEP: {feature} "
                f"(Mean Absolute Change = "
                f"{row['Mean_Absolute_Change_%']:.6f}%)\n"

            )


    f.write(
        "\nEXCLUDED UNSTABLE FEATURES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )


    if len(unstable_features) == 0:

        f.write(
            "NONE\n"
        )

    else:

        for feature in unstable_features:

            row = stability_df[
                stability_df[
                    "Feature"
                ]
                ==
                feature
            ].iloc[0]


            f.write(

                f"EXCLUDE: {feature} "
                f"(Mean Absolute Change = "
                f"{row['Mean_Absolute_Change_%']:.6f}%)\n"

            )


    f.write(
        "\nSUMMARY\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"Stable features: "
        f"{len(stable_features)}\n"
    )

    f.write(
        f"Unstable features: "
        f"{len(unstable_features)}\n"
    )


print(
    "Saved:",
    report_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print(
    "STEP 12 - LBP FEATURE STABILITY COMPLETE"
)
print("=" * 80)


print("\nSTABILITY CRITERION")
print("-" * 80)

print(
    "Mean absolute change <= "
    f"{STABILITY_THRESHOLD}% -> STABLE -> KEEP"
)

print(
    "Mean absolute change > "
    f"{STABILITY_THRESHOLD}% -> UNSTABLE -> EXCLUDE"
)


print("\nFINAL STABLE FEATURES")
print("-" * 80)


if len(stable_features) == 0:

    print("NONE")

else:

    for feature in stable_features:

        row = stability_df[
            stability_df[
                "Feature"
            ]
            ==
            feature
        ].iloc[0]


        print(

            f"KEEP: {feature} "
            f"(Mean Change = "
            f"{row['Mean_Absolute_Change_%']:.4f}%)"

        )


print("\nEXCLUDED UNSTABLE FEATURES")
print("-" * 80)


if len(unstable_features) == 0:

    print("NONE")

else:

    for feature in unstable_features:

        row = stability_df[
            stability_df[
                "Feature"
            ]
            ==
            feature
        ].iloc[0]


        print(

            f"EXCLUDE: {feature} "
            f"(Mean Change = "
            f"{row['Mean_Absolute_Change_%']:.4f}%)"

        )


print("\nFILES")
print("-" * 80)

print(
    all_results_csv
)

print(
    stability_csv
)

print(
    stable_csv
)

print(
    final_stable_csv
)

print(
    report_path
)


print("\n")
print("=" * 80)
print(
    "SUCCESS - STEP 12 LBP STABILITY"
)
print("=" * 80)


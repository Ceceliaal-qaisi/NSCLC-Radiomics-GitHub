
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 11 - GLCM TEXTURE
# FEATURE STABILITY ANALYSIS
#
# From scratch:
#   - GLCM construction
#   - Symmetric GLCM
#   - Contrast
#   - Correlation
#   - Energy
#   - Homogeneity
#   - Entropy
#   - Maximum Probability
#   - Segmentation perturbation
#   - 3D Dilation / Erosion by 1 and 2 voxels
#   - Feature stability analysis
#
# Stability criterion:
#   Mean absolute percentage change <= 10% -> STABLE
#   Mean absolute percentage change > 10%  -> UNSTABLE
#
# Final decision:
#   STABLE   -> KEEP
#   UNSTABLE -> EXCLUDE
# ================================================================

import os
import glob
import math

import numpy as np
import pandas as pd
import pydicom
import nrrd

from scipy.ndimage import binary_dilation, binary_erosion


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

MASK_PATH = os.path.join(
    BASE_DIR,
    "GTV1_MASK.nrrd"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_11_GLCM_TEXTURE"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# PARAMETERS
# ================================================================

N_LEVELS = 32

DISTANCES = [1, 2, 3]

ANGLES = {
    0: (0, 1),
    45: (-1, 1),
    90: (-1, 0),
    135: (-1, -1)
}

STABILITY_THRESHOLD = 10.0


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 11 - GLCM TEXTURE")
print("GLCM FEATURE STABILITY ANALYSIS")
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

print(
    "CT files found:",
    len(ct_files)
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

print(
    "Mask dtype:",
    mask_data.dtype
)

mask = np.asarray(
    mask_data
)


# ------------------------------------------------
# Convert mask to (Z,Y,X)
# ------------------------------------------------

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


mask = mask > 0

print(
    "Converted mask shape:",
    mask.shape
)

original_voxels = int(
    np.sum(mask)
)

print(
    "Original tumor voxels:",
    original_voxels
)


# ================================================================
# STEP 3 - FIND TUMOR INTENSITY RANGE
# ================================================================

print("\nSTEP 3 - FINDING CT INTENSITY RANGE")
print("=" * 70)

tumor_intensities = ct_volume[
    mask
]

if len(tumor_intensities) == 0:

    raise ValueError(
        "Tumor mask is empty."
    )


min_hu = float(
    np.min(
        tumor_intensities
    )
)

max_hu = float(
    np.max(
        tumor_intensities
    )
)

print(
    f"Minimum HU: {min_hu:.6f}"
)

print(
    f"Maximum HU: {max_hu:.6f}"
)

print(
    "Quantization levels:",
    N_LEVELS
)


# ================================================================
# STEP 4 - GRAY-LEVEL QUANTIZATION
# ================================================================

print("\nSTEP 4 - GRAY-LEVEL QUANTIZATION")
print("=" * 70)


def quantize_volume(
    volume,
    levels,
    min_value,
    max_value
):

    if max_value == min_value:

        return np.zeros_like(
            volume,
            dtype=np.int32
        )

    normalized = (
        (volume - min_value)
        /
        (max_value - min_value)
    )

    q = np.floor(
        normalized * levels
    ).astype(
        np.int32
    )

    q[q < 0] = 0

    q[q >= levels] = (
        levels - 1
    )

    return q


quantized_volume = quantize_volume(
    ct_volume,
    N_LEVELS,
    min_hu,
    max_hu
)


occupied_levels = np.unique(
    quantized_volume[mask]
)

print(
    "Quantization completed."
)

print(
    "Occupied gray levels:",
    len(occupied_levels)
)

print(
    "Gray levels:",
    occupied_levels
)

if len(occupied_levels) == N_LEVELS:

    print(
        "PASS - All quantization levels are occupied."
    )

else:

    print(
        "WARNING - Not all quantization levels are occupied."
    )


# ================================================================
# STEP 5 - BUILD SYMMETRIC GLCM FROM SCRATCH
# ================================================================

print("\nSTEP 5 - BUILDING SYMMETRIC GLCM FROM SCRATCH")
print("=" * 70)


def build_glcm_2d(
    image,
    binary_mask,
    distance,
    dr,
    dc,
    levels
):

    glcm = np.zeros(
        (levels, levels),
        dtype=np.float64
    )

    rows, cols = image.shape

    for r in range(rows):

        for c in range(cols):

            if not binary_mask[r, c]:
                continue

            r2 = (
                r
                +
                dr * distance
            )

            c2 = (
                c
                +
                dc * distance
            )

            if (
                r2 < 0
                or r2 >= rows
                or c2 < 0
                or c2 >= cols
            ):
                continue

            if not binary_mask[
                r2,
                c2
            ]:
                continue

            i = int(
                image[r, c]
            )

            j = int(
                image[r2, c2]
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Symmetric GLCM
            # ------------------------------------------------

            glcm[i, j] += 1.0

            glcm[j, i] += 1.0


    total = np.sum(
        glcm
    )

    if total > 0:

        glcm /= total

    return glcm


# ================================================================
# STEP 6 - GLCM FEATURES FROM SCRATCH
# ================================================================


def glcm_contrast(
    glcm
):

    levels = glcm.shape[0]

    result = 0.0

    for i in range(levels):

        for j in range(levels):

            result += (
                (i - j) ** 2
                *
                glcm[i, j]
            )

    return float(
        result
    )


# ------------------------------------------------


def glcm_energy(
    glcm
):

    return float(
        np.sum(
            glcm ** 2
        )
    )


# ------------------------------------------------


def glcm_homogeneity(
    glcm
):

    levels = glcm.shape[0]

    result = 0.0

    for i in range(levels):

        for j in range(levels):

            result += (
                glcm[i, j]
                /
                (
                    1.0
                    +
                    abs(i - j)
                )
            )

    return float(
        result
    )


# ------------------------------------------------


def glcm_entropy(
    glcm
):

    entropy = 0.0

    for value in glcm.flatten():

        if value > 0:

            entropy -= (
                value
                *
                math.log2(
                    value
                )
            )

    return float(
        entropy
    )


# ------------------------------------------------


def glcm_max_probability(
    glcm
):

    return float(
        np.max(
            glcm
        )
    )


# ------------------------------------------------


def glcm_correlation(
    glcm
):

    levels = glcm.shape[0]

    values = np.arange(
        levels,
        dtype=np.float64
    )

    px = np.sum(
        glcm,
        axis=1
    )

    py = np.sum(
        glcm,
        axis=0
    )

    mean_x = np.sum(
        values * px
    )

    mean_y = np.sum(
        values * py
    )

    variance_x = np.sum(
        (
            values
            -
            mean_x
        ) ** 2
        *
        px
    )

    variance_y = np.sum(
        (
            values
            -
            mean_y
        ) ** 2
        *
        py
    )

    sigma_x = math.sqrt(
        max(
            variance_x,
            0.0
        )
    )

    sigma_y = math.sqrt(
        max(
            variance_y,
            0.0
        )
    )

    if (
        sigma_x == 0
        or
        sigma_y == 0
    ):

        return 0.0

    correlation = 0.0

    for i in range(levels):

        for j in range(levels):

            correlation += (
                (i - mean_x)
                *
                (j - mean_y)
                *
                glcm[i, j]
            )

    correlation /= (
        sigma_x
        *
        sigma_y
    )

    return float(
        correlation
    )


# ================================================================
# STEP 7 - CALCULATE OVERALL GLCM FEATURES
# ================================================================


def calculate_glcm_features(
    ct_volume,
    binary_mask,
    quantized_volume
):

    tumor_slice_indices = np.where(
        np.any(
            binary_mask,
            axis=(1, 2)
        )
    )[0]


    measurements = []


    for slice_index in tumor_slice_indices:

        slice_mask = binary_mask[
            slice_index
        ]

        slice_image = quantized_volume[
            slice_index
        ]


        for distance in DISTANCES:

            for angle, direction in ANGLES.items():

                dr, dc = direction


                glcm = build_glcm_2d(
                    slice_image,
                    slice_mask,
                    distance,
                    dr,
                    dc,
                    N_LEVELS
                )


                if np.sum(glcm) == 0:

                    continue


                measurements.append({

                    "Slice":
                        int(
                            slice_index
                        ),

                    "Distance":
                        int(
                            distance
                        ),

                    "Angle":
                        int(
                            angle
                        ),

                    "Contrast":
                        glcm_contrast(
                            glcm
                        ),

                    "Correlation":
                        glcm_correlation(
                            glcm
                        ),

                    "Energy":
                        glcm_energy(
                            glcm
                        ),

                    "Homogeneity":
                        glcm_homogeneity(
                            glcm
                        ),

                    "Entropy":
                        glcm_entropy(
                            glcm
                        ),

                    "Maximum_Probability":
                        glcm_max_probability(
                            glcm
                        )

                })


    df = pd.DataFrame(
        measurements
    )


    feature_columns = [

        "Contrast",
        "Correlation",
        "Energy",
        "Homogeneity",
        "Entropy",
        "Maximum_Probability"

    ]


    overall = {}


    for feature in feature_columns:

        overall[feature] = float(
            df[feature].mean()
        )


    return (
        overall,
        df
    )


# ================================================================
# STEP 8 - ORIGINAL SEGMENTATION
# ================================================================

print("\nSTEP 8 - ORIGINAL SEGMENTATION")
print("=" * 70)

original_features, original_df = (
    calculate_glcm_features(
        ct_volume,
        mask,
        quantized_volume
    )
)

print(
    "Tumor voxels:",
    int(
        np.sum(mask)
    )
)

print(
    "GLCM measurements:",
    len(
        original_df
    )
)

print(
    "\nOriginal GLCM features:"
)

for feature, value in original_features.items():

    print(
        f"{feature:22s}: "
        f"{value:.10f}"
    )


# ================================================================
# STEP 9 - 3D SEGMENTATION PERTURBATIONS
# ================================================================

print("\nSTEP 9 - SEGMENTATION PERTURBATIONS")
print("=" * 70)

print(
    "Perturbation method: "
    "3D binary dilation / erosion"
)

print(
    "Structuring element: 3 x 3 x 3"
)


# ------------------------------------------------
# 3D structuring element
# ------------------------------------------------

structure = np.ones(
    (3, 3, 3),
    dtype=bool
)


perturbed_masks = {

    "Original":
        mask,

    "Dilation_1_voxel":
        binary_dilation(
            mask,
            structure=structure,
            iterations=1
        ),

    "Erosion_1_voxel":
        binary_erosion(
            mask,
            structure=structure,
            iterations=1
        ),

    "Dilation_2_voxels":
        binary_dilation(
            mask,
            structure=structure,
            iterations=2
        ),

    "Erosion_2_voxels":
        binary_erosion(
            mask,
            structure=structure,
            iterations=2
        )

}


perturbed_features = {}


for name, perturbed_mask in perturbed_masks.items():

    print(
        f"\nPROCESSING: {name}"
    )

    voxel_count = int(
        np.sum(
            perturbed_mask
        )
    )

    print(
        "Tumor voxels:",
        voxel_count
    )


    if voxel_count == 0:

        raise ValueError(
            f"{name} produced an empty mask."
        )


    features, _ = (
        calculate_glcm_features(
            ct_volume,
            perturbed_mask,
            quantized_volume
        )
    )


    perturbed_features[
        name
    ] = features


    for feature, value in features.items():

        print(
            f"{feature:22s}: "
            f"{value:.10f}"
        )


# ================================================================
# STEP 10 - FEATURE STABILITY CALCULATION
# ================================================================

print("\nSTEP 10 - CALCULATING FEATURE STABILITY")
print("=" * 70)

feature_columns = [

    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Entropy",
    "Maximum_Probability"

]


perturbation_names = [

    "Dilation_1_voxel",
    "Erosion_1_voxel",
    "Dilation_2_voxels",
    "Erosion_2_voxels"

]


stability_rows = []


for feature in feature_columns:

    original_value = (
        original_features[
            feature
        ]
    )


    changes = {}


    for perturbation in perturbation_names:

        perturbed_value = (
            perturbed_features[
                perturbation
            ][feature]
        )


        # ------------------------------------------------
        # Absolute percentage change
        # ------------------------------------------------

        if abs(
            original_value
        ) < 1e-12:

            percentage_change = np.nan

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

            ) * 100.0


        changes[
            perturbation
        ] = float(
            percentage_change
        )


    valid_changes = [

        value

        for value in changes.values()

        if not np.isnan(
            value
        )

    ]


    if len(
        valid_changes
    ) == 0:

        mean_change = np.nan
        max_change = np.nan

        stability = "UNDEFINED"

        decision = "EXCLUDE"

    else:

        mean_change = float(
            np.mean(
                valid_changes
            )
        )

        max_change = float(
            np.max(
                valid_changes
            )
        )


        if (
            mean_change
            <=
            STABILITY_THRESHOLD
        ):

            stability = "STABLE"

            decision = "KEEP"

        else:

            stability = "UNSTABLE"

            decision = "EXCLUDE"


    stability_rows.append({

        "Feature":
            feature,

        "Original_Value":
            original_value,

        "Dilation_1_%":
            changes[
                "Dilation_1_voxel"
            ],

        "Erosion_1_%":
            changes[
                "Erosion_1_voxel"
            ],

        "Dilation_2_%":
            changes[
                "Dilation_2_voxels"
            ],

        "Erosion_2_%":
            changes[
                "Erosion_2_voxels"
            ],

        "Mean_Absolute_Change_%":
            mean_change,

        "Maximum_Change_%":
            max_change,

        "Stability":
            stability,

        "Decision":
            decision

    })


stability_df = pd.DataFrame(
    stability_rows
)


# ================================================================
# STEP 11 - PRINT STABILITY RESULTS
# ================================================================

print("\n")
print("=" * 70)
print("GLCM FEATURE STABILITY RESULTS")
print("=" * 70)

print(
    f"Stability threshold: "
    f"{STABILITY_THRESHOLD:.1f}%"
)

print(
    "Criterion: Mean absolute percentage change <= 10% = STABLE"
)


for _, row in stability_df.iterrows():

    print(

        f"\n{row['Feature']:22s}"

        f"\n  Dilation 1 voxel : "
        f"{row['Dilation_1_%']:.6f}%"

        f"\n  Erosion 1 voxel  : "
        f"{row['Erosion_1_%']:.6f}%"

        f"\n  Dilation 2 voxels: "
        f"{row['Dilation_2_%']:.6f}%"

        f"\n  Erosion 2 voxels : "
        f"{row['Erosion_2_%']:.6f}%"

        f"\n  Mean change      : "
        f"{row['Mean_Absolute_Change_%']:.6f}%"

        f"\n  Maximum change   : "
        f"{row['Maximum_Change_%']:.6f}%"

        f"\n  Stability        : "
        f"{row['Stability']}"

        f"\n  Decision         : "
        f"{row['Decision']}"

    )


# ================================================================
# STEP 12 - SAVE STABILITY CSV
# ================================================================

print("\nSTEP 12 - SAVING STABILITY RESULTS")
print("=" * 70)

stability_csv_path = os.path.join(

    OUTPUT_DIR,

    "GTV1_GLCM_Feature_Stability.csv"

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
# STEP 13 - SAVE STABLE FEATURES
# ================================================================

print("\nSTEP 13 - SAVING STABLE FEATURES")
print("=" * 70)

stable_df = stability_df[
    stability_df[
        "Stability"
    ]
    ==
    "STABLE"
].copy()


unstable_df = stability_df[
    stability_df[
        "Stability"
    ]
    ==
    "UNSTABLE"
].copy()


stable_features_path = os.path.join(

    OUTPUT_DIR,

    "GTV1_GLCM_Stable_Features.csv"

)


stable_df.to_csv(

    stable_features_path,

    index=False

)


print(
    "Saved:",
    stable_features_path
)


# ================================================================
# STEP 14 - FINAL FEATURE LIST
# ================================================================

print("\nSTEP 14 - CREATING FINAL FEATURE LIST")
print("=" * 70)

final_rows = []


for _, row in stability_df.iterrows():

    final_rows.append({

        "Feature":
            row["Feature"],

        "Original_Value":
            row["Original_Value"],

        "Mean_Absolute_Change_%":
            row[
                "Mean_Absolute_Change_%"
            ],

        "Maximum_Change_%":
            row[
                "Maximum_Change_%"
            ],

        "Stability":
            row[
                "Stability"
            ],

        "Decision":
            row[
                "Decision"
            ]

    })


final_df = pd.DataFrame(
    final_rows
)


final_features_path = os.path.join(

    OUTPUT_DIR,

    "GTV1_GLCM_Final_Stable_Features.csv"

)


final_df.to_csv(

    final_features_path,

    index=False

)


print(
    "Saved:",
    final_features_path
)


# ================================================================
# STEP 15 - SAVE TEXT REPORT
# ================================================================

print("\nSTEP 15 - SAVING REPORT")
print("=" * 70)

report_path = os.path.join(

    OUTPUT_DIR,

    "glcm_feature_stability_report.txt"

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
        "STEP 11 - GLCM TEXTURE\n"
    )

    f.write(
        "GLCM FEATURE STABILITY ANALYSIS\n"
    )

    f.write(
        "=" * 70
        +
        "\n\n"
    )


    f.write(
        "PATIENT INFORMATION\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        "Patient: LUNG1-001\n"
    )

    f.write(
        "Series: 69331\n\n"
    )


    f.write(
        "GLCM PARAMETERS\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        f"Quantization levels: "
        f"{N_LEVELS}\n"
    )

    f.write(
        f"Full GTV-1 HU range: "
        f"{min_hu:.6f} to "
        f"{max_hu:.6f} HU\n"
    )

    f.write(
        f"Distances: "
        f"{DISTANCES}\n"
    )

    f.write(
        "Angles: 0, 45, 90, 135 degrees\n"
    )

    f.write(
        "GLCM type: Symmetric\n"
    )

    f.write(
        "Texture analysis: 2D slice-based\n\n"
    )


    f.write(
        "SEGMENTATION PERTURBATIONS\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        "Original segmentation\n"
    )

    f.write(
        "3D dilation by 1 voxel\n"
    )

    f.write(
        "3D erosion by 1 voxel\n"
    )

    f.write(
        "3D dilation by 2 voxels\n"
    )

    f.write(
        "3D erosion by 2 voxels\n"
    )

    f.write(
        "Structuring element: 3 x 3 x 3\n\n"
    )


    f.write(
        "STABILITY CRITERION\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )

    f.write(
        "Mean absolute percentage change <= 10% "
        "is considered STABLE.\n"
    )

    f.write(
        "Mean absolute percentage change > 10% "
        "is considered UNSTABLE.\n\n"
    )


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
            f"\nFeature: "
            f"{row['Feature']}\n"
        )

        f.write(
            f"Original value: "
            f"{row['Original_Value']:.10f}\n"
        )

        f.write(
            f"Dilation 1 voxel: "
            f"{row['Dilation_1_%']:.6f}%\n"
        )

        f.write(
            f"Erosion 1 voxel: "
            f"{row['Erosion_1_%']:.6f}%\n"
        )

        f.write(
            f"Dilation 2 voxels: "
            f"{row['Dilation_2_%']:.6f}%\n"
        )

        f.write(
            f"Erosion 2 voxels: "
            f"{row['Erosion_2_%']:.6f}%\n"
        )

        f.write(
            f"Mean absolute change: "
            f"{row['Mean_Absolute_Change_%']:.6f}%\n"
        )

        f.write(
            f"Maximum change: "
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
        "\n\nFINAL FEATURES TO KEEP\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    for feature in stable_df[
        "Feature"
    ]:

        f.write(
            f"KEEP: "
            f"{feature}\n"
        )


    f.write(
        "\nFEATURES TO EXCLUDE\n"
    )

    f.write(
        "-" * 70
        +
        "\n"
    )


    for feature in unstable_df[
        "Feature"
    ]:

        f.write(
            f"EXCLUDE: "
            f"{feature}\n"
        )


# ================================================================
# STEP 16 - FINAL VALIDATION
# ================================================================

print("\nSTEP 16 - FINAL VALIDATION")
print("=" * 70)


# ------------------------------------------------
# Validation 1
# ------------------------------------------------

if len(stability_df) == 6:

    print(
        "PASS - Six GLCM features were evaluated."
    )

else:

    print(
        "FAIL - Expected six GLCM features."
    )


# ------------------------------------------------
# Validation 2
# ------------------------------------------------

required_features = set(

    feature_columns

)


actual_features = set(

    stability_df[
        "Feature"
    ]

)


if actual_features == required_features:

    print(
        "PASS - All six required GLCM descriptors exist."
    )

else:

    print(
        "FAIL - Feature list mismatch."
    )


# ------------------------------------------------
# Validation 3
# ------------------------------------------------

if np.all(
    np.isfinite(
        stability_df[
            "Mean_Absolute_Change_%"
        ]
    )
):

    print(
        "PASS - Stability changes are finite."
    )

else:

    print(
        "FAIL - Non-finite stability value detected."
    )


# ------------------------------------------------
# Validation 4
# ------------------------------------------------

if np.all(
    stability_df[
        "Mean_Absolute_Change_%"
    ]
    >=
    0
):

    print(
        "PASS - Percentage changes are non-negative."
    )

else:

    print(
        "FAIL - Negative percentage change detected."
    )


# ------------------------------------------------
# Validation 5
# ------------------------------------------------

if np.all(
    np.isfinite(
        stability_df[
            "Original_Value"
        ]
    )
):

    print(
        "PASS - Original feature values are finite."
    )

else:

    print(
        "FAIL - Non-finite original feature value."
    )


# ================================================================
# STEP 17 - FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("FINAL GLCM STABILITY SUMMARY")
print("=" * 70)


print(
    "\nStable features:",
    len(
        stable_df
    )
)


print(
    "Unstable features:",
    len(
        unstable_df
    )
)


print(
    "\nFINAL FEATURES TO KEEP"
)

print(
    "-" * 70
)


if len(
    stable_df
) == 0:

    print(
        "NONE"
    )

else:

    for feature in stable_df[
        "Feature"
    ]:

        print(
            "KEEP:",
            feature
        )


print(
    "\nFEATURES TO EXCLUDE"
)

print(
    "-" * 70
)


if len(
    unstable_df
) == 0:

    print(
        "NONE"
    )

else:

    for feature in unstable_df[
        "Feature"
    ]:

        print(
            "EXCLUDE:",
            feature
        )


# ================================================================
# FILES
# ================================================================

print(
    "\nFILES SAVED"
)

print(
    "-" * 70
)

print(
    stability_csv_path
)

print(
    stable_features_path
)

print(
    final_features_path
)

print(
    report_path
)


# ================================================================
# COMPLETE
# ================================================================

print("\n")
print("=" * 70)
print(
    "SUCCESS - GLCM FEATURE STABILITY ANALYSIS COMPLETE"
)
print("=" * 70)
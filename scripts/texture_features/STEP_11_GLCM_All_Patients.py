
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 11 - GLCM TEXTURE FEATURES - ALL PATIENTS
#
# GLCM implemented completely from scratch
# 2D slice-based GLCM
# Symmetric GLCM
# Full GTV-1 HU range quantization
#
# Required descriptors:
#   1. Contrast
#   2. Correlation
#   3. Energy / ASM
#   4. Homogeneity
#   5. Entropy
#   6. Maximum Probability
# ================================================================

import os
import glob
import math

import numpy as np
import pandas as pd
import pydicom
import nrrd


# ================================================================
# PATHS
# ================================================================

ROOT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "STEP_11_GLCM_ALL_PATIENTS"
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


FEATURE_COLUMNS = [
    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Entropy",
    "Maximum_Probability"
]


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 11 - GLCM TEXTURE FEATURES - ALL PATIENTS")
print("=" * 70)

print("\nRoot directory:")
print(ROOT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ================================================================
# FIND PATIENTS
# ================================================================

print("\nSTEP 1 - FINDING PATIENTS")
print("=" * 70)

patient_dirs = []

for path in glob.glob(
    os.path.join(ROOT_DIR, "LUNG1-*")
):

    if os.path.isdir(path):

        patient_dirs.append(path)


patient_dirs = sorted(
    patient_dirs
)


print(
    "Patients found:",
    len(patient_dirs)
)


if len(patient_dirs) == 0:

    raise FileNotFoundError(
        "No LUNG1 patient folders found."
    )


# ================================================================
# GLCM CONSTRUCTION
# ================================================================

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

            r2 = r + dr * distance
            c2 = c + dc * distance

            if (
                r2 < 0
                or r2 >= rows
                or c2 < 0
                or c2 >= cols
            ):
                continue

            if not binary_mask[r2, c2]:
                continue

            i = int(image[r, c])
            j = int(image[r2, c2])

            if (
                i < 0
                or i >= levels
                or j < 0
                or j >= levels
            ):
                continue

            glcm[i, j] += 1.0
            glcm[j, i] += 1.0

    total = float(
        np.sum(glcm)
    )

    if total > 0:

        glcm /= total

    return glcm


# ================================================================
# GLCM FEATURES
# ================================================================

def glcm_contrast(glcm):

    levels = glcm.shape[0]

    result = 0.0

    for i in range(levels):

        for j in range(levels):

            result += (
                (i - j) ** 2
                *
                glcm[i, j]
            )

    return float(result)


def glcm_correlation(glcm):

    levels = glcm.shape[0]

    indices = np.arange(
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
        indices * px
    )

    mean_y = np.sum(
        indices * py
    )

    variance_x = np.sum(
        (
            indices - mean_x
        ) ** 2
        *
        px
    )

    variance_y = np.sum(
        (
            indices - mean_y
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
        sigma_x == 0.0
        or sigma_y == 0.0
    ):

        return 0.0

    numerator = 0.0

    for i in range(levels):

        for j in range(levels):

            numerator += (
                (i - mean_x)
                *
                (j - mean_y)
                *
                glcm[i, j]
            )

    return float(
        numerator
        /
        (
            sigma_x
            *
            sigma_y
        )
    )


def glcm_energy(glcm):

    return float(
        np.sum(
            glcm ** 2
        )
    )


def glcm_homogeneity(glcm):

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

    return float(result)


def glcm_entropy(glcm):

    result = 0.0

    for value in glcm.flatten():

        if value > 0:

            result -= (
                value
                *
                math.log2(value)
            )

    return float(result)


def glcm_max_probability(glcm):

    return float(
        np.max(glcm)
    )


# ================================================================
# FIND CT DIRECTORY
# ================================================================

def find_ct_directory(patient_dir):

    candidates = []

    for root, dirs, files in os.walk(patient_dir):

        dicom_count = 0

        for filename in files:

            path = os.path.join(
                root,
                filename
            )

            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True
                )

                if getattr(
                    ds,
                    "Modality",
                    ""
                ) == "CT":

                    dicom_count += 1

            except:

                pass

        if dicom_count > 0:

            candidates.append(
                (
                    root,
                    dicom_count
                )
            )

    if len(candidates) == 0:

        return None

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[0][0]


# ================================================================
# FIND GTV1 MASK
# ================================================================

def find_gtv1_mask(patient_dir):

    candidates = []

    for root, dirs, files in os.walk(patient_dir):

        for filename in files:

            lower = filename.lower()

            if (
                lower.endswith(".nrrd")
                and
                (
                    "gtv1" in lower
                    or "gtv-1" in lower
                    or "gtv_1" in lower
                )
            ):

                candidates.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    if len(candidates) == 0:

        return None

    return candidates[0]


# ================================================================
# READ CT
# ================================================================

def read_ct_volume(ct_dir):

    ct_files = glob.glob(
        os.path.join(
            ct_dir,
            "*.dcm"
        )
    )

    if len(ct_files) == 0:

        raise ValueError(
            "No DICOM files found."
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

    slices = []

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

        slices.append(
            hu
        )

    return np.stack(
        slices,
        axis=0
    )


# ================================================================
# PROCESS ONE PATIENT
# ================================================================

def process_patient(patient_dir):

    patient_id = os.path.basename(
        patient_dir
    )

    print("\n")
    print("-" * 70)

    print(
        "PROCESSING:",
        patient_id
    )

    print("-" * 70)


    # ------------------------------------------------------------
    # Find CT
    # ------------------------------------------------------------

    ct_dir = find_ct_directory(
        patient_dir
    )

    if ct_dir is None:

        raise ValueError(
            "CT directory not found."
        )


    print(
        "CT directory:",
        ct_dir
    )


    # ------------------------------------------------------------
    # Read CT
    # ------------------------------------------------------------

    ct_volume = read_ct_volume(
        ct_dir
    )


    print(
        "CT shape:",
        ct_volume.shape
    )


    # ------------------------------------------------------------
    # Find mask
    # ------------------------------------------------------------

    mask_path = find_gtv1_mask(
        patient_dir
    )


    if mask_path is None:

        raise ValueError(
            "GTV1_MASK.nrrd not found."
        )


    print(
        "GTV-1 mask:",
        mask_path
    )


    # ------------------------------------------------------------
    # Read mask
    # ------------------------------------------------------------

    mask_data, mask_header = nrrd.read(
        mask_path
    )

    mask = np.asarray(
        mask_data
    )


    expected_nrrd_shape = (
        ct_volume.shape[1],
        ct_volume.shape[2],
        ct_volume.shape[0]
    )


    if mask.shape == expected_nrrd_shape:

        mask = np.transpose(
            mask,
            (2, 1, 0)
        )

    elif mask.shape == ct_volume.shape:

        pass

    else:

        raise ValueError(
            f"Mask shape {mask.shape} "
            f"does not match CT shape "
            f"{ct_volume.shape}"
        )


    mask = mask > 0


    tumor_voxels = int(
        np.sum(mask)
    )


    if tumor_voxels == 0:

        raise ValueError(
            "GTV-1 mask is empty."
        )


    print(
        "Tumor voxels:",
        tumor_voxels
    )


    # ------------------------------------------------------------
    # Tumor intensities
    # ------------------------------------------------------------

    tumor_intensities = (
        ct_volume[mask]
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
        f"HU range: "
        f"{min_hu:.3f} to {max_hu:.3f}"
    )


    # ------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------

    if max_hu <= min_hu:

        quantized_tumor = np.zeros(
            len(tumor_intensities),
            dtype=np.int32
        )

    else:

        normalized = (
            tumor_intensities - min_hu
        ) / (
            max_hu - min_hu
        )

        quantized_tumor = np.floor(
            normalized * N_LEVELS
        ).astype(
            np.int32
        )

        quantized_tumor[
            quantized_tumor < 0
        ] = 0

        quantized_tumor[
            quantized_tumor >= N_LEVELS
        ] = N_LEVELS - 1


    quantized_volume = np.full(
        ct_volume.shape,
        -1,
        dtype=np.int32
    )


    quantized_volume[mask] = (
        quantized_tumor
    )


    # ------------------------------------------------------------
    # Tumor slices
    # ------------------------------------------------------------

    tumor_slice_indices = np.where(
        np.any(
            mask,
            axis=(1, 2)
        )
    )[0]


    print(
        "Tumor slices:",
        len(tumor_slice_indices)
    )


    # ------------------------------------------------------------
    # Calculate all GLCM measurements
    # ------------------------------------------------------------

    patient_measurements = []


    for slice_index in tumor_slice_indices:

        slice_mask = (
            mask[
                slice_index
            ]
        )

        slice_image = (
            quantized_volume[
                slice_index
            ]
        )


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


                if np.sum(glcm) <= 0:

                    continue


                patient_measurements.append({

                    "Patient_ID":
                        patient_id,

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


    measurements_df = pd.DataFrame(
        patient_measurements
    )


    if len(measurements_df) == 0:

        raise ValueError(
            "No GLCM measurements generated."
        )


    expected_measurements = (
        len(tumor_slice_indices)
        *
        len(DISTANCES)
        *
        len(ANGLES)
    )


    print(
        "Expected measurements:",
        expected_measurements
    )

    print(
        "Actual measurements:",
        len(measurements_df)
    )


    # ------------------------------------------------------------
    # Patient-level mean
    # ------------------------------------------------------------

    patient_features = {}

    for feature in FEATURE_COLUMNS:

        patient_features[
            feature
        ] = float(
            measurements_df[
                feature
            ].mean()
        )


    return (
        patient_id,
        ct_dir,
        mask_path,
        tumor_voxels,
        len(tumor_slice_indices),
        expected_measurements,
        len(measurements_df),
        measurements_df,
        patient_features
    )


# ================================================================
# PROCESS ALL PATIENTS
# ================================================================

print("\nSTEP 2 - PROCESSING ALL PATIENTS")
print("=" * 70)


all_patient_features = []

all_detailed_measurements = []

processing_status = []


for patient_dir in patient_dirs:

    patient_id = os.path.basename(
        patient_dir
    )

    try:

        (
            patient_id,
            ct_dir,
            mask_path,
            tumor_voxels,
            tumor_slices,
            expected_measurements,
            actual_measurements,
            measurements_df,
            patient_features
        ) = process_patient(
            patient_dir
        )


        # --------------------------------------------------------
        # Patient-level features
        # --------------------------------------------------------

        row = {
            "Patient_ID":
                patient_id
        }


        for feature in FEATURE_COLUMNS:

            row[feature] = (
                patient_features[
                    feature
                ]
            )


        all_patient_features.append(
            row
        )


        # --------------------------------------------------------
        # Detailed measurements
        # --------------------------------------------------------

        all_detailed_measurements.append(
            measurements_df
        )


        # --------------------------------------------------------
        # Status
        # --------------------------------------------------------

        processing_status.append({

            "Patient_ID":
                patient_id,

            "Status":
                "SUCCESS",

            "CT_Directory":
                ct_dir,

            "GTV1_Mask":
                mask_path,

            "Tumor_Slices":
                tumor_slices,

            "Tumor_Voxels":
                tumor_voxels,

            "Expected_Measurements":
                expected_measurements,

            "Actual_Measurements":
                actual_measurements

        })


        print(
            "STATUS:",
            patient_id,
            "SUCCESS"
        )


    except Exception as e:

        print(
            "STATUS:",
            patient_id,
            "FAILED"
        )

        print(
            "Reason:",
            str(e)
        )


        processing_status.append({

            "Patient_ID":
                patient_id,

            "Status":
                "FAILED",

            "CT_Directory":
                "",

            "GTV1_Mask":
                "",

            "Tumor_Slices":
                0,

            "Tumor_Voxels":
                0,

            "Expected_Measurements":
                0,

            "Actual_Measurements":
                0,

            "Error":
                str(e)

        })


# ================================================================
# SAVE PATIENT FEATURES
# ================================================================

print("\nSTEP 3 - SAVING PATIENT-LEVEL FEATURES")
print("=" * 70)


patient_features_df = pd.DataFrame(
    all_patient_features
)


if len(patient_features_df) == 0:

    raise ValueError(
        "No patients were successfully processed."
    )


patient_features_path = os.path.join(
    OUTPUT_DIR,
    "STEP_11_All_Patients_GLCM_Features.csv"
)


patient_features_df.to_csv(
    patient_features_path,
    index=False
)


print(
    "Saved:",
    patient_features_path
)


# ================================================================
# SAVE DETAILED MEASUREMENTS
# ================================================================

print("\nSTEP 4 - SAVING DETAILED MEASUREMENTS")
print("=" * 70)


detailed_df = pd.concat(
    all_detailed_measurements,
    ignore_index=True
)


detailed_path = os.path.join(
    OUTPUT_DIR,
    "STEP_11_All_Patients_GLCM_Detailed.csv"
)


detailed_df.to_csv(
    detailed_path,
    index=False
)


print(
    "Saved:",
    detailed_path
)

print(
    "Total detailed measurements:",
    len(detailed_df)
)


# ================================================================
# SAVE PROCESSING STATUS
# ================================================================

print("\nSTEP 5 - SAVING PROCESSING STATUS")
print("=" * 70)


status_df = pd.DataFrame(
    processing_status
)


status_path = os.path.join(
    OUTPUT_DIR,
    "STEP_11_GLCM_Patient_Processing_Status.csv"
)


status_df.to_csv(
    status_path,
    index=False
)


print(
    "Saved:",
    status_path
)


# ================================================================
# COVERAGE
# ================================================================

print("\nSTEP 6 - FEATURE COVERAGE")
print("=" * 70)


total_patients = len(
    patient_dirs
)


coverage_results = []


for feature in FEATURE_COLUMNS:

    patients_with_feature = int(
        patient_features_df[
            feature
        ].notna().sum()
    )


    coverage_percent = (
        patients_with_feature
        /
        total_patients
        *
        100.0
    )


    coverage_results.append({

        "Feature":
            feature,

        "Patients_With_Feature":
            patients_with_feature,

        "Total_Patients":
            total_patients,

        "Coverage_Percent":
            coverage_percent

    })


coverage_df = pd.DataFrame(
    coverage_results
)


coverage_path = os.path.join(
    OUTPUT_DIR,
    "STEP_11_GLCM_Feature_Coverage.csv"
)


coverage_df.to_csv(
    coverage_path,
    index=False
)


print(
    coverage_df.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    coverage_path
)


# ================================================================
# OVERALL STATISTICS
# ================================================================

print("\nSTEP 7 - OVERALL GLCM STATISTICS")
print("=" * 70)


overall_results = []


for feature in FEATURE_COLUMNS:

    values = patient_features_df[
        feature
    ].dropna()


    overall_results.append({

        "Feature":
            feature,

        "Mean":
            float(
                values.mean()
            ),

        "Standard_Deviation":
            float(
                values.std()
            ),

        "Minimum":
            float(
                values.min()
            ),

        "Maximum":
            float(
                values.max()
            ),

        "Patients":
            int(
                values.count()
            )

    })


overall_df = pd.DataFrame(
    overall_results
)


overall_path = os.path.join(
    OUTPUT_DIR,
    "STEP_11_GLCM_Overall_Statistics.csv"
)


overall_df.to_csv(
    overall_path,
    index=False
)


print(
    overall_df.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    overall_path
)


# ================================================================
# FINAL VALIDATION
# ================================================================

print("\nSTEP 8 - FINAL VALIDATION")
print("=" * 70)


validation_pass = True


# ------------------------------------------------
# Patient count
# ------------------------------------------------

successful_patients = len(
    patient_features_df
)


failed_patients = (
    total_patients
    -
    successful_patients
)


print(
    "Total patients:",
    total_patients
)

print(
    "Successful patients:",
    successful_patients
)

print(
    "Failed patients:",
    failed_patients
)


# ------------------------------------------------
# Feature existence
# ------------------------------------------------

missing_features = [

    feature
    for feature in FEATURE_COLUMNS
    if feature not in patient_features_df.columns

]


if len(missing_features) == 0:

    print(
        "PASS - All six GLCM features exist."
    )

else:

    print(
        "FAIL - Missing features:",
        missing_features
    )

    validation_pass = False


# ------------------------------------------------
# Numerical validation
# ------------------------------------------------

if np.all(
    np.isfinite(
        patient_features_df[
            FEATURE_COLUMNS
        ].values
    )
):

    print(
        "PASS - All patient-level feature values are finite."
    )

else:

    print(
        "FAIL - Invalid feature values detected."
    )

    validation_pass = False


# ------------------------------------------------
# Correlation validation
# ------------------------------------------------

correlation_values = (
    patient_features_df[
        "Correlation"
    ].values
)


if np.all(
    (
        correlation_values >= -1.0
    )
    &
    (
        correlation_values <= 1.0
    )
):

    print(
        "PASS - Correlation values are within [-1, 1]."
    )

else:

    print(
        "FAIL - Correlation outside [-1,1]."
    )

    validation_pass = False


# ------------------------------------------------
# Energy validation
# ------------------------------------------------

energy_values = (
    patient_features_df[
        "Energy"
    ].values
)


if np.all(
    (
        energy_values >= 0.0
    )
    &
    (
        energy_values <= 1.0
    )
):

    print(
        "PASS - Energy values are within [0,1]."
    )

else:

    print(
        "FAIL - Energy outside [0,1]."
    )

    validation_pass = False


# ------------------------------------------------
# Homogeneity validation
# ------------------------------------------------

homogeneity_values = (
    patient_features_df[
        "Homogeneity"
    ].values
)


if np.all(
    (
        homogeneity_values >= 0.0
    )
    &
    (
        homogeneity_values <= 1.0
    )
):

    print(
        "PASS - Homogeneity values are within [0,1]."
    )

else:

    print(
        "FAIL - Homogeneity outside [0,1]."
    )

    validation_pass = False


# ------------------------------------------------
# Maximum probability validation
# ------------------------------------------------

max_probability_values = (
    patient_features_df[
        "Maximum_Probability"
    ].values
)


if np.all(
    (
        max_probability_values >= 0.0
    )
    &
    (
        max_probability_values <= 1.0
    )
):

    print(
        "PASS - Maximum probability values are within [0,1]."
    )

else:

    print(
        "FAIL - Maximum probability outside [0,1]."
    )

    validation_pass = False


# ------------------------------------------------
# Detailed measurement validation
# ------------------------------------------------

required_detailed_columns = [

    "Patient_ID",
    "Slice",
    "Distance",
    "Angle"

] + FEATURE_COLUMNS


missing_detailed = [

    column
    for column in required_detailed_columns
    if column not in detailed_df.columns

]


if len(missing_detailed) == 0:

    print(
        "PASS - Detailed GLCM table contains all required columns."
    )

else:

    print(
        "FAIL - Missing detailed columns:",
        missing_detailed
    )

    validation_pass = False


# ------------------------------------------------
# Distance validation
# ------------------------------------------------

actual_distances = sorted(
    detailed_df[
        "Distance"
    ].unique().tolist()
)


if actual_distances == sorted(
    DISTANCES
):

    print(
        "PASS - All distances [1,2,3] preserved."
    )

else:

    print(
        "FAIL - Distance validation failed."
    )

    validation_pass = False


# ------------------------------------------------
# Orientation validation
# ------------------------------------------------

actual_angles = sorted(
    detailed_df[
        "Angle"
    ].unique().tolist()
)


if actual_angles == sorted(
    ANGLES.keys()
):

    print(
        "PASS - All orientations preserved."
    )

else:

    print(
        "FAIL - Orientation validation failed."
    )

    validation_pass = False


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)

print(
    "STEP 11 - ALL PATIENTS GLCM COMPLETE"
)

print("=" * 70)


print(
    "\nPatients:"
)

print(
    "Total:",
    total_patients
)

print(
    "Success:",
    successful_patients
)

print(
    "Failed:",
    failed_patients
)


print(
    "\nGLCM DESIGN:"
)

print(
    "Gray levels:",
    N_LEVELS
)

print(
    "Distances:",
    DISTANCES
)

print(
    "Angles:",
    list(ANGLES.keys())
)

print(
    "2D slice-based: YES"
)

print(
    "Symmetric GLCM: YES"
)

print(
    "Implemented from scratch: YES"
)


print(
    "\nFEATURES:"
)

for feature in FEATURE_COLUMNS:

    print(
        f"{feature:25s}: "
        f"{patient_features_df[feature].mean():.6f}"
    )


print(
    "\nOUTPUT FILES:"
)

print(
    patient_features_path
)

print(
    detailed_path
)

print(
    status_path
)

print(
    coverage_path
)

print(
    overall_path
)


print("\n")
print("=" * 70)


if validation_pass:

    print(
        "SUCCESS - ALL PATIENTS GLCM VALIDATION PASSED"
    )

else:

    print(
        "WARNING - GLCM COMPLETED WITH VALIDATION ISSUES"
    )


print("=" * 70)

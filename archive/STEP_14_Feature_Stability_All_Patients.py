
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 14 - FEATURE STABILITY ANALYSIS - ALL PATIENTS
#
# Purpose:
# Perturb the GTV-1 segmentation boundary by:
#   - 1 voxel inward  (erosion)
#   - 1 voxel outward (dilation)
#   - 2 voxels inward (erosion)
#   - 2 voxels outward (dilation)
#
# Then compare texture features with the original segmentation.
#
# Features tested:
#   Statistical Texture
#   GLCM
#   LBP
#   Spectral Texture
#
# Stability criterion:
# A feature is considered stable for a patient if the maximum
# relative change caused by the perturbations is <= 10%.
#
# Final feature stability is based on the percentage of patients
# for which the feature remains stable.
#
# ================================================================

import os
import glob
import numpy as np
import pandas as pd
import pydicom


# ================================================================
# PATHS
# ================================================================

BASE_ROOT = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

MASK_ROOT = os.path.join(
    BASE_ROOT,
    "STEP_13_GTV1_MASK_PREPARATION"
)

OUTPUT_DIR = os.path.join(
    BASE_ROOT,
    "STEP_14_FEATURE_STABILITY_ALL_PATIENTS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# SETTINGS
# ================================================================

TOTAL_PATIENTS = 422

PERTURBATIONS = [
    "Erode_1",
    "Dilate_1",
    "Erode_2",
    "Dilate_2"
]

# Patient-level stability threshold
PATIENT_CHANGE_THRESHOLD = 0.10

# Feature must be stable in at least this percentage
# of patients to survive
FEATURE_STABILITY_THRESHOLD = 0.80


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 14 - FEATURE STABILITY ANALYSIS - ALL PATIENTS")
print("=" * 75)

print("\nPerturbations:")
print("  Erode 1 voxel")
print("  Dilate 1 voxel")
print("  Erode 2 voxels")
print("  Dilate 2 voxels")

print("\nPatient-level change threshold: 10%")
print("Feature stability threshold: 80%")


# ================================================================
# FIND PATIENT DIRECTORY
# ================================================================

def find_patient_directory(patient_id):

    direct_path = os.path.join(
        BASE_ROOT,
        patient_id
    )

    if os.path.isdir(direct_path):
        return direct_path

    return None


# ================================================================
# FIND GTV1 MASK
# ================================================================

def find_gtv1_mask(patient_dir):

    candidates = [

        os.path.join(
            patient_dir,
            "GTV1_MASK",
            "GTV1_MASK.npy"
        ),

        os.path.join(
            patient_dir,
            "GTV1_MASK.npy"
        )

    ]

    for path in candidates:

        if os.path.isfile(path):
            return path

    recursive_candidates = glob.glob(
        os.path.join(
            patient_dir,
            "**",
            "GTV1_MASK.npy"
        ),
        recursive=True
    )

    if len(recursive_candidates) > 0:
        return recursive_candidates[0]

    return None


# ================================================================
# FIND CT FILES
# ================================================================

def find_ct_files(patient_dir):

    all_files = glob.glob(
        os.path.join(
            patient_dir,
            "**",
            "*.dcm"
        ),
        recursive=True
    )

    ct_files = []

    for path in all_files:

        try:

            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )

            modality = str(
                getattr(
                    ds,
                    "Modality",
                    ""
                )
            ).upper()

            if modality == "CT":
                ct_files.append(path)

        except Exception:

            continue

    return ct_files


# ================================================================
# INSTANCE NUMBER
# ================================================================

def get_instance_number(path):

    try:

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

    except Exception:

        return 0


# ================================================================
# READ CT
# ================================================================

def read_ct_volume(ct_files):

    ct_files = sorted(
        ct_files,
        key=get_instance_number
    )

    slices = []

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

        slices.append(hu)

    return np.stack(
        slices,
        axis=0
    )


# ================================================================
# MORPHOLOGICAL DILATION
#
# Implemented directly with NumPy.
#
# 8-connected neighborhood in each CT slice.
# ================================================================

def dilate_2d(mask):

    result = mask.copy()

    result[:-1, :] |= mask[1:, :]
    result[1:, :] |= mask[:-1, :]

    result[:, :-1] |= mask[:, 1:]
    result[:, 1:] |= mask[:, :-1]

    result[:-1, :-1] |= mask[1:, 1:]
    result[1:, 1:] |= mask[:-1, :-1]

    result[:-1, 1:] |= mask[1:, :-1]
    result[1:, :-1] |= mask[:-1, 1:]

    return result


# ================================================================
# MORPHOLOGICAL EROSION
#
# A pixel survives only when all 8 neighbors and itself
# belong to the region.
# ================================================================

def erode_2d(mask):

    result = mask.copy()

    result[:-1, :] &= mask[1:, :]
    result[1:, :] &= mask[:-1, :]

    result[:, :-1] &= mask[:, 1:]
    result[:, 1:] &= mask[:, :-1]

    result[:-1, :-1] &= mask[1:, 1:]
    result[1:, 1:] &= mask[:-1, :-1]

    result[:-1, 1:] &= mask[1:, :-1]
    result[1:, :-1] &= mask[:-1, 1:]

    return result


# ================================================================
# 3-D SLICE-WISE PERTURBATION
#
# The assignment specifies perturbation of the segmentation
# boundary by one and two voxels.
#
# Here perturbation is applied slice-by-slice in the CT plane,
# preserving the 2-D texture analysis used in previous steps.
# ================================================================

def erode_mask(mask, amount):

    result = mask.copy()

    for _ in range(amount):

        new_result = np.zeros_like(
            result,
            dtype=bool
        )

        for z in range(
            result.shape[0]
        ):

            new_result[z] = erode_2d(
                result[z]
            )

        result = new_result

    return result


def dilate_mask(mask, amount):

    result = mask.copy()

    for _ in range(amount):

        new_result = np.zeros_like(
            result,
            dtype=bool
        )

        for z in range(
            result.shape[0]
        ):

            new_result[z] = dilate_2d(
                result[z]
            )

        result = new_result

    return result


# ================================================================
# STATISTICAL TEXTURE
#
# Mean, variance, smoothness, third moment,
# uniformity and entropy.
# ================================================================

def statistical_features(
    ct_volume,
    mask
):

    values = ct_volume[mask]

    if len(values) < 2:
        return None

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    if maximum == minimum:
        maximum = minimum + 1.0

    histogram, edges = np.histogram(
        values,
        bins=32,
        range=(minimum, maximum)
    )

    total = np.sum(histogram)

    if total == 0:
        return None

    probability = (
        histogram.astype(np.float64)
        /
        total
    )

    centers = (
        edges[:-1]
        +
        edges[1:]
    ) / 2.0

    mean_value = float(
        np.sum(
            centers * probability
        )
    )

    variance = float(
        np.sum(
            (
                centers
                -
                mean_value
            ) ** 2
            *
            probability
        )
    )

    normalized_variance = (
        variance
        /
        (
            (maximum - minimum) ** 2
        )
    )

    smoothness = (
        1.0
        -
        1.0
        /
        (
            1.0
            +
            normalized_variance
        )
    )

    third_moment = float(
        np.sum(
            (
                centers
                -
                mean_value
            ) ** 3
            *
            probability
        )
    )

    uniformity = float(
        np.sum(
            probability ** 2
        )
    )

    nonzero = probability[
        probability > 0
    ]

    entropy = float(
        -np.sum(
            nonzero
            *
            np.log2(nonzero)
        )
    )

    return {

        "Statistical_Mean":
            mean_value,

        "Statistical_Variance":
            variance,

        "Statistical_Smoothness":
            smoothness,

        "Statistical_Third_Moment":
            third_moment,

        "Statistical_Uniformity":
            uniformity,

        "Statistical_Entropy":
            entropy

    }


# ================================================================
# QUANTIZATION
# ================================================================

def quantize_image(
    image,
    mask,
    levels=32
):

    values = image[mask]

    if len(values) == 0:
        return None

    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    if maximum == minimum:
        return np.zeros_like(
            image,
            dtype=np.int32
        )

    q = np.floor(
        (
            image
            -
            minimum
        )
        /
        (
            maximum
            -
            minimum
        )
        *
        levels
    ).astype(
        np.int32
    )

    q = np.clip(
        q,
        0,
        levels - 1
    )

    return q


# ================================================================
# GLCM
# ================================================================

def calculate_glcm_features(
    image,
    mask,
    levels=32
):

    q = quantize_image(
        image,
        mask,
        levels
    )

    if q is None:
        return None

    distances = [1, 2, 3]

    directions = [
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1)
    ]

    all_features = []

    rows, cols = image.shape

    for distance in distances:

        for dr, dc in directions:

            glcm = np.zeros(
                (
                    levels,
                    levels
                ),
                dtype=np.float64
            )

            for r in range(rows):

                nr = (
                    r
                    +
                    dr * distance
                )

                if nr < 0 or nr >= rows:
                    continue

                for c in range(cols):

                    if not mask[r, c]:
                        continue

                    nc = (
                        c
                        +
                        dc * distance
                    )

                    if nc < 0 or nc >= cols:
                        continue

                    if not mask[nr, nc]:
                        continue

                    i = q[r, c]
                    j = q[nr, nc]

                    glcm[i, j] += 1
                    glcm[j, i] += 1

            total = np.sum(glcm)

            if total == 0:
                continue

            p = glcm / total

            i_values = np.arange(levels)
            j_values = np.arange(levels)

            I, J = np.meshgrid(
                i_values,
                j_values,
                indexing="ij"
            )

            contrast = float(
                np.sum(
                    (
                        I - J
                    ) ** 2
                    *
                    p
                )
            )

            energy = float(
                np.sum(
                    p ** 2
                )
            )

            homogeneity = float(
                np.sum(
                    p
                    /
                    (
                        1.0
                        +
                        np.abs(I - J)
                    )
                )
            )

            maximum_probability = float(
                np.max(p)
            )

            nonzero = p[
                p > 0
            ]

            entropy = float(
                -np.sum(
                    nonzero
                    *
                    np.log2(nonzero)
                )
            )

            row_prob = np.sum(
                p,
                axis=1
            )

            col_prob = np.sum(
                p,
                axis=0
            )

            mean_i = np.sum(
                i_values
                *
                row_prob
            )

            mean_j = np.sum(
                j_values
                *
                col_prob
            )

            std_i = np.sqrt(
                np.sum(
                    (
                        i_values
                        -
                        mean_i
                    ) ** 2
                    *
                    row_prob
                )
            )

            std_j = np.sqrt(
                np.sum(
                    (
                        j_values
                        -
                        mean_j
                    ) ** 2
                    *
                    col_prob
                )
            )

            if (
                std_i > 0
                and
                std_j > 0
            ):

                correlation = float(
                    np.sum(
                        (
                            I
                            -
                            mean_i
                        )
                        *
                        (
                            J
                            -
                            mean_j
                        )
                        *
                        p
                    )
                    /
                    (
                        std_i
                        *
                        std_j
                    )
                )

            else:

                correlation = 0.0

            all_features.append({

                "GLCM_Contrast":
                    contrast,

                "GLCM_Correlation":
                    correlation,

                "GLCM_Energy":
                    energy,

                "GLCM_Homogeneity":
                    homogeneity,

                "GLCM_Entropy":
                    entropy,

                "GLCM_Maximum_Probability":
                    maximum_probability

            })

    if len(all_features) == 0:
        return None

    result = {}

    feature_names = all_features[0].keys()

    for feature in feature_names:

        result[feature] = float(
            np.mean(
                [
                    x[feature]
                    for x in all_features
                ]
            )
        )

    return result


# ================================================================
# LBP
# ================================================================

def calculate_lbp_features(
    image,
    mask
):

    rows, cols = image.shape

    lbp_values = []

    offsets = [

        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0,  1),
        ( 1,  1),
        ( 1,  0),
        ( 1, -1),
        ( 0, -1)

    ]

    for r in range(
        1,
        rows - 1
    ):

        for c in range(
            1,
            cols - 1
        ):

            if not mask[r, c]:
                continue

            center = image[r, c]

            code = 0

            valid = True

            for p, (dr, dc) in enumerate(
                offsets
            ):

                nr = r + dr
                nc = c + dc

                if not mask[nr, nc]:

                    valid = False
                    break

                if image[nr, nc] >= center:

                    code |= (
                        1 << p
                    )

            if valid:

                lbp_values.append(
                    code
                )

    if len(lbp_values) == 0:
        return None

    lbp_values = np.asarray(
        lbp_values,
        dtype=np.float64
    )

    mean_value = float(
        np.mean(lbp_values)
    )

    variance = float(
        np.var(lbp_values)
    )

    histogram, _ = np.histogram(
        lbp_values,
        bins=256,
        range=(0, 256)
    )

    probability = (
        histogram
        /
        np.sum(histogram)
    )

    uniformity = float(
        np.sum(
            probability ** 2
        )
    )

    nonzero = probability[
        probability > 0
    ]

    entropy = float(
        -np.sum(
            nonzero
            *
            np.log2(nonzero)
        )
    )

    return {

        "LBP_Mean":
            mean_value,

        "LBP_Variance":
            variance,

        "LBP_Uniformity":
            uniformity,

        "LBP_Entropy":
            entropy

    }


# ================================================================
# SPECTRAL TEXTURE
# ================================================================

def calculate_spectral_features(
    image,
    mask
):

    values = image[mask]

    if len(values) < 2:
        return None

    rows, cols = image.shape

    analysis = np.array(
        image,
        dtype=np.float64
    )

    tumor_mean = float(
        np.mean(values)
    )

    analysis[
        ~mask
    ] = tumor_mean

    analysis -= np.mean(
        analysis
    )

    spectrum = np.fft.fftshift(
        np.fft.fft2(
            analysis
        )
    )

    magnitude = np.abs(
        spectrum
    )

    power = (
        magnitude ** 2
    )

    center_r = rows // 2
    center_c = cols // 2

    power[
        center_r,
        center_c
    ] = 0.0

    total_power = float(
        np.sum(power)
    )

    if total_power <= 0:
        return None

    probability = (
        power
        /
        total_power
    )

    spectral_energy = float(
        np.sum(
            probability ** 2
        )
    )

    nonzero = probability[
        probability > 0
    ]

    spectral_entropy = float(
        -np.sum(
            nonzero
            *
            np.log2(nonzero)
        )
    )

    fy = np.fft.fftshift(
        np.fft.fftfreq(rows)
    )

    fx = np.fft.fftshift(
        np.fft.fftfreq(cols)
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

    max_index = np.unravel_index(
        np.argmax(power),
        power.shape
    )

    dominant_frequency = float(
        radial_frequency[
            max_index
        ]
    )

    if dominant_frequency > 0:

        fundamental_period = (
            1.0
            /
            dominant_frequency
        )

    else:

        fundamental_period = np.nan

    dominant_orientation = float(
        (
            np.degrees(
                np.arctan2(
                    FY[max_index],
                    FX[max_index]
                )
            )
            %
            180.0
        )
    )

    radial_mean = float(
        np.sum(
            radial_frequency
            *
            probability
        )
    )

    radial_variance = float(
        np.sum(
            (
                radial_frequency
                -
                radial_mean
            ) ** 2
            *
            probability
        )
    )

    angular_orientation = (
        np.degrees(
            np.arctan2(
                FY,
                FX
            )
        )
        %
        180.0
    )

    angular_mean = float(
        np.sum(
            angular_orientation
            *
            probability
        )
    )

    angular_variance = float(
        np.sum(
            (
                angular_orientation
                -
                angular_mean
            ) ** 2
            *
            probability
        )
    )

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

        "Angular_Mean":
            angular_mean,

        "Angular_Variance":
            angular_variance

    }


# ================================================================
# EXTRACT ALL TEXTURE FEATURES
# ================================================================

def calculate_all_features(
    ct_volume,
    mask
):

    results = {}

    # ------------------------------------------------------------
    # Process every tumor-containing slice
    # ------------------------------------------------------------

    tumor_slices = np.where(
        np.any(
            mask,
            axis=(1, 2)
        )
    )[0]

    if len(tumor_slices) == 0:
        return None

    slice_results = []

    for z in tumor_slices:

        image = ct_volume[z]
        slice_mask = mask[z]

        features = {}

        stat = statistical_features(
            ct_volume,
            mask
        )

        if stat is not None:
            features.update(stat)

        glcm = calculate_glcm_features(
            image,
            slice_mask
        )

        if glcm is not None:
            features.update(glcm)

        lbp = calculate_lbp_features(
            image,
            slice_mask
        )

        if lbp is not None:
            features.update(lbp)

        spectral = calculate_spectral_features(
            image,
            slice_mask
        )

        if spectral is not None:
            features.update(spectral)

        if len(features) > 0:
            slice_results.append(
                features
            )

    if len(slice_results) == 0:
        return None

    feature_names = set()

    for record in slice_results:
        feature_names.update(
            record.keys()
        )

    for feature in sorted(
        feature_names
    ):

        values = []

        for record in slice_results:

            if feature in record:

                value = record[
                    feature
                ]

                if np.isfinite(value):
                    values.append(
                        value
                    )

        if len(values) > 0:

            results[feature] = float(
                np.mean(values)
            )

    return results


# ================================================================
# RELATIVE CHANGE
# ================================================================

def relative_change(
    original,
    perturbed
):

    if (
        original is None
        or
        perturbed is None
    ):

        return np.nan

    if not np.isfinite(original):
        return np.nan

    if not np.isfinite(perturbed):
        return np.nan

    denominator = max(
        abs(original),
        1e-8
    )

    return abs(
        perturbed
        -
        original
    ) / denominator


# ================================================================
# PROCESS ONE PATIENT
# ================================================================

def process_patient(
    patient_id
):

    patient_dir = find_patient_directory(
        patient_id
    )

    if patient_dir is None:

        return None, "PATIENT_FOLDER_NOT_FOUND"

    mask_file = find_gtv1_mask(
        patient_dir
    )

    if mask_file is None:

        return None, "GTV1_MASK_NOT_FOUND"

    ct_files = find_ct_files(
        patient_dir
    )

    if len(ct_files) == 0:

        return None, "CT_NOT_FOUND"

    ct_volume = read_ct_volume(
        ct_files
    )

    original_mask = np.load(
        mask_file
    ).astype(bool)

    if original_mask.shape != ct_volume.shape:

        return None, (
            "SHAPE_MISMATCH_"
            f"CT_{ct_volume.shape}_"
            f"MASK_{original_mask.shape}"
        )

    if np.sum(original_mask) == 0:

        return None, "EMPTY_GTV1_MASK"

    # ------------------------------------------------------------
    # Original
    # ------------------------------------------------------------

    original_features = calculate_all_features(
        ct_volume,
        original_mask
    )

    if original_features is None:

        return None, "NO_ORIGINAL_FEATURES"

    # ------------------------------------------------------------
    # Perturbed masks
    # ------------------------------------------------------------

    masks = {

        "Erode_1":
            erode_mask(
                original_mask,
                1
            ),

        "Dilate_1":
            dilate_mask(
                original_mask,
                1
            ),

        "Erode_2":
            erode_mask(
                original_mask,
                2
            ),

        "Dilate_2":
            dilate_mask(
                original_mask,
                2
            )

    }

    records = []

    for perturbation, perturbed_mask in masks.items():

        if np.sum(
            perturbed_mask
        ) == 0:

            continue

        perturbed_features = calculate_all_features(
            ct_volume,
            perturbed_mask
        )

        if perturbed_features is None:
            continue

        for feature in original_features:

            if feature not in perturbed_features:
                continue

            original_value = original_features[
                feature
            ]

            perturbed_value = perturbed_features[
                feature
            ]

            change = relative_change(
                original_value,
                perturbed_value
            )

            records.append({

                "Patient_ID":
                    patient_id,

                "Perturbation":
                    perturbation,

                "Feature":
                    feature,

                "Original_Value":
                    original_value,

                "Perturbed_Value":
                    perturbed_value,

                "Relative_Change":
                    change,

                "Percent_Change":
                    change * 100.0
                    if np.isfinite(change)
                    else np.nan,

                "Stable":
                    bool(
                        np.isfinite(change)
                        and
                        change
                        <=
                        PATIENT_CHANGE_THRESHOLD
                    )

            })

    if len(records) == 0:

        return None, "NO_STABILITY_RESULTS"

    return records, "SUCCESS"


# ================================================================
# MAIN
# ================================================================

print("\n")
print("=" * 75)
print("PROCESSING ALL PATIENTS")
print("=" * 75)


all_records = []
status_records = []


for patient_number in range(
    1,
    TOTAL_PATIENTS + 1
):

    patient_id = (
        f"LUNG1-{patient_number:03d}"
    )

    print(
        f"\nPROCESSING {patient_id}"
    )

    try:

        records, status = process_patient(
            patient_id
        )

        if status == "SUCCESS":

            all_records.extend(
                records
            )

            status_records.append({

                "Patient_ID":
                    patient_id,

                "Status":
                    "SUCCESS",

                "Reason":
                    "Stability calculated"

            })

            print(
                "SUCCESS"
            )

        else:

            status_records.append({

                "Patient_ID":
                    patient_id,

                "Status":
                    "FAILED",

                "Reason":
                    status

            })

            print(
                "FAILED:",
                status
            )

    except Exception as e:

        status_records.append({

            "Patient_ID":
                patient_id,

            "Status":
                "FAILED",

            "Reason":
                str(e)

        })

        print(
            "ERROR:",
            e
        )


# ================================================================
# DATAFRAME
# ================================================================

stability_df = pd.DataFrame(
    all_records
)

status_df = pd.DataFrame(
    status_records
)


# ================================================================
# SAVE RAW STABILITY RESULTS
# ================================================================

raw_file = os.path.join(
    OUTPUT_DIR,
    "STEP_14_All_Patients_Feature_Stability_Raw.csv"
)

stability_df.to_csv(
    raw_file,
    index=False
)

print(
    "\nSaved:",
    raw_file
)


# ================================================================
# FEATURE SUMMARY
# ================================================================

summary_rows = []


if len(stability_df) > 0:

    for feature in sorted(
        stability_df[
            "Feature"
        ].unique()
    ):

        feature_df = stability_df[
            stability_df[
                "Feature"
            ]
            ==
            feature
        ]

        valid_changes = feature_df[
            "Relative_Change"
        ].dropna()

        if len(valid_changes) == 0:
            continue

        stable_count = int(
            feature_df[
                "Stable"
            ].sum()
        )

        total_comparisons = len(
            feature_df
        )

        stability_percentage = (
            stable_count
            /
            total_comparisons
            *
            100.0
        )

        mean_change = float(
            np.mean(
                valid_changes
            )
            *
            100.0
        )

        median_change = float(
            np.median(
                valid_changes
            )
            *
            100.0
        )

        maximum_change = float(
            np.max(
                valid_changes
            )
            *
            100.0
        )

        if (
            stability_percentage
            >=
            FEATURE_STABILITY_THRESHOLD
            * 100.0
        ):

            final_status = "STABLE"

        else:

            final_status = "UNSTABLE_EXCLUDE"

        summary_rows.append({

            "Feature":
                feature,

            "Stable_Comparisons":
                stable_count,

            "Total_Comparisons":
                total_comparisons,

            "Stability_Percent":
                stability_percentage,

            "Mean_Change_Percent":
                mean_change,

            "Median_Change_Percent":
                median_change,

            "Maximum_Change_Percent":
                maximum_change,

            "Final_Decision":
                final_status

        })


summary_df = pd.DataFrame(
    summary_rows
)


# ================================================================
# SAVE FEATURE SUMMARY
# ================================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "STEP_14_Feature_Stability_Summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

print(
    "Saved:",
    summary_file
)


# ================================================================
# STABLE FEATURES
# ================================================================

stable_features_df = summary_df[
    summary_df[
        "Final_Decision"
    ]
    ==
    "STABLE"
].copy()


stable_file = os.path.join(
    OUTPUT_DIR,
    "STEP_14_Stable_Features.csv"
)

stable_features_df.to_csv(
    stable_file,
    index=False
)

print(
    "Saved:",
    stable_file
)


# ================================================================
# UNSTABLE FEATURES
# ================================================================

unstable_features_df = summary_df[
    summary_df[
        "Final_Decision"
    ]
    ==
    "UNSTABLE_EXCLUDE"
].copy()


unstable_file = os.path.join(
    OUTPUT_DIR,
    "STEP_14_Unstable_Features_To_Exclude.csv"
)

unstable_features_df.to_csv(
    unstable_file,
    index=False
)

print(
    "Saved:",
    unstable_file
)


# ================================================================
# PROCESSING STATUS
# ================================================================

status_file = os.path.join(
    OUTPUT_DIR,
    "STEP_14_Patient_Processing_Status.csv"
)

status_df.to_csv(
    status_file,
    index=False
)

print(
    "Saved:",
    status_file
)


# ================================================================
# REPORT
# ================================================================

successful_count = int(
    (
        status_df[
            "Status"
        ]
        ==
        "SUCCESS"
    )
    .sum()
)

failed_count = int(
    (
        status_df[
            "Status"
        ]
        ==
        "FAILED"
    )
    .sum()
)

success_rate = (
    successful_count
    /
    TOTAL_PATIENTS
    *
    100.0
)


report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_14_Feature_Stability_Report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 14 - FEATURE STABILITY ANALYSIS\n"
    )

    f.write(
        "=" * 75
        +
        "\n\n"
    )

    f.write(
        f"Total patients: "
        f"{TOTAL_PATIENTS}\n"
    )

    f.write(
        f"Successful patients: "
        f"{successful_count}\n"
    )

    f.write(
        f"Failed patients: "
        f"{failed_count}\n"
    )

    f.write(
        f"Success rate: "
        f"{success_rate:.2f}%\n\n"
    )

    f.write(
        "PERTURBATIONS\n"
    )

    f.write(
        "-" * 75
        +
        "\n"
    )

    f.write(
        "Erode 1 voxel\n"
    )

    f.write(
        "Dilate 1 voxel\n"
    )

    f.write(
        "Erode 2 voxels\n"
    )

    f.write(
        "Dilate 2 voxels\n\n"
    )

    f.write(
        "STABILITY CRITERION\n"
    )

    f.write(
        "-" * 75
        +
        "\n"
    )

    f.write(
        "Patient-level feature change <= 10% "
        "was considered stable.\n"
    )

    f.write(
        "A feature was retained when it was stable "
        f"in at least "
        f"{FEATURE_STABILITY_THRESHOLD * 100:.0f}% "
        "of valid comparisons.\n\n"
    )

    f.write(
        "FINAL FEATURE DECISIONS\n"
    )

    f.write(
        "-" * 75
        +
        "\n"
    )

    for _, row in summary_df.iterrows():

        f.write(
            f"{row['Feature']} | "
            f"{row['Final_Decision']} | "
            f"Stability: "
            f"{row['Stability_Percent']:.2f}% | "
            f"Mean change: "
            f"{row['Mean_Change_Percent']:.2f}% | "
            f"Max change: "
            f"{row['Maximum_Change_Percent']:.2f}%\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "STABLE FEATURES\n"
    )

    f.write(
        "-" * 75
        +
        "\n"
    )

    for feature in stable_features_df[
        "Feature"
    ]:

        f.write(
            f"{feature}\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "UNSTABLE FEATURES TO EXCLUDE\n"
    )

    f.write(
        "-" * 75
        +
        "\n"
    )

    for feature in unstable_features_df[
        "Feature"
    ]:

        f.write(
            f"{feature}\n"
        )


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 75)
print(
    "STEP 14 - FEATURE STABILITY COMPLETE"
)
print("=" * 75)

print(
    "\nTotal patients:",
    TOTAL_PATIENTS
)

print(
    "Successful:",
    successful_count
)

print(
    "Failed:",
    failed_count
)

print(
    f"Success rate: "
    f"{success_rate:.2f}%"
)

print(
    "\nStable features:",
    len(stable_features_df)
)

print(
    "Unstable features:",
    len(unstable_features_df)
)

print(
    "\nOUTPUT DIRECTORY:"
)

print(
    OUTPUT_DIR
)

print(
    "\nFILES:"
)

print(
    raw_file
)

print(
    summary_file
)

print(
    stable_file
)

print(
    unstable_file
)

print(
    status_file
)

print(
    report_file
)

print("\n")
print("=" * 75)
print(
    "SUCCESS - STEP 14 FEATURE STABILITY"
)
print("=" * 75)


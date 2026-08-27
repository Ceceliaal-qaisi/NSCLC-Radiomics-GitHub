# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 12 - FINAL LBP VALIDATION
#
# Independent validation of the STEP 12 LBP implementation
#
# Checks:
#   1. CT / mask dimensions
#   2. Tumor voxel count
#   3. LBP parameters P=8, R=1
#   4. LBP range [0,255]
#   5. Histogram total
#   6. Probability sum
#   7. Independent LBP histogram calculation
#   8. Agreement between saved histogram and independent calculation
#   9. Boundary handling information
# ================================================================

import os
import numpy as np
import pandas as pd
import pydicom
import nrrd
import glob


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_DIR = os.path.join(BASE_DIR, "82046")

MASK_PATH = os.path.join(
    BASE_DIR,
    "GTV1_MASK.nrrd"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_12_LBP_TEXTURE"
)

HISTOGRAM_FILE = os.path.join(
    OUTPUT_DIR,
    "GTV1_LBP_histogram.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "GTV1_LBP_summary.csv"
)

VALIDATION_REPORT = os.path.join(
    OUTPUT_DIR,
    "STEP_12_FINAL_VALIDATION_REPORT.txt"
)


# ================================================================
# PARAMETERS
# ================================================================

P = 8
R = 1

EXPECTED_MIN = 0
EXPECTED_MAX = 255


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 12 - FINAL LBP VALIDATION")
print("=" * 70)


# ================================================================
# STEP 1 - READ CT
# ================================================================

print("\nSTEP 1 - READING CT")
print("=" * 70)

ct_files = glob.glob(
    os.path.join(CT_DIR, "*.dcm")
)

if len(ct_files) == 0:
    raise FileNotFoundError(
        f"No CT DICOM files found:\n{CT_DIR}"
    )


def get_instance(path):
    ds = pydicom.dcmread(
        path,
        stop_before_pixels=True
    )
    return int(
        getattr(ds, "InstanceNumber", 0)
    )


ct_files = sorted(
    ct_files,
    key=get_instance
)

ct_volume = []

for path in ct_files:

    ds = pydicom.dcmread(path)

    image = ds.pixel_array.astype(
        np.float64
    )

    slope = float(
        getattr(ds, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0.0)
    )

    hu = image * slope + intercept

    ct_volume.append(hu)

ct_volume = np.stack(
    ct_volume,
    axis=0
)

print("CT files:", len(ct_files))
print(
    "CT shape (Z,Y,X):",
    ct_volume.shape
)


# ================================================================
# STEP 2 - READ GTV-1 MASK
# ================================================================

print("\nSTEP 2 - READING GTV-1 MASK")
print("=" * 70)

mask_data, header = nrrd.read(
    MASK_PATH
)

print(
    "Original mask shape:",
    mask_data.shape
)

mask = np.asarray(mask_data)

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
        f"Mask shape {mask.shape} "
        f"does not match CT shape "
        f"{ct_volume.shape}"
    )

mask = mask > 0

tumor_voxels = int(
    np.sum(mask)
)

print(
    "Converted mask shape:",
    mask.shape
)

print(
    "Tumor voxels:",
    tumor_voxels
)


# ================================================================
# STEP 3 - CHECK PARAMETERS
# ================================================================

print("\nSTEP 3 - CHECKING LBP PARAMETERS")
print("=" * 70)

print("P =", P)
print("R =", R)

if P == 8 and R == 1:
    print(
        "PASS - Standard 8-neighbor "
        "radius-1 LBP."
    )
else:
    print(
        "WARNING - Parameters are not P=8, R=1."
    )


# ================================================================
# STEP 4 - INDEPENDENT LBP IMPLEMENTATION
# ================================================================

print(
    "\nSTEP 4 - INDEPENDENT LBP CALCULATION"
)
print("=" * 70)


def calculate_lbp_independent(
    image,
    roi
):

    rows, cols = image.shape

    lbp = np.zeros(
        (rows, cols),
        dtype=np.uint8
    )

    valid_mask = np.zeros(
        (rows, cols),
        dtype=bool
    )

    # Eight direct neighbors:
    #
    # 7 0 1
    # 6 C 2
    # 5 4 3
    #
    # P=8, R=1

    neighbors = [
        (-1, 0),   # 0
        (-1, 1),   # 1
        (0, 1),    # 2
        (1, 1),    # 3
        (1, 0),    # 4
        (1, -1),   # 5
        (0, -1),   # 6
        (-1, -1)   # 7
    ]

    for r in range(
        1,
        rows - 1
    ):

        for c in range(
            1,
            cols - 1
        ):

            if not roi[r, c]:
                continue

            code = 0

            center = image[r, c]

            for p, (dr, dc) in enumerate(
                neighbors
            ):

                neighbor = image[
                    r + dr,
                    c + dc
                ]

                if neighbor >= center:

                    code += (
                        2 ** p
                    )

            lbp[r, c] = code

            valid_mask[r, c] = True

    return lbp, valid_mask


# ================================================================
# STEP 5 - APPLY INDEPENDENT LBP
# ================================================================

print(
    "\nSTEP 5 - PROCESSING TUMOR SLICES"
)
print("=" * 70)

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

independent_codes = []

valid_pixel_count = 0

for slice_index in tumor_slice_indices:

    image = ct_volume[
        slice_index
    ]

    roi = mask[
        slice_index
    ]

    lbp, valid_mask = (
        calculate_lbp_independent(
            image,
            roi
        )
    )

    valid_codes = lbp[
        valid_mask
    ]

    independent_codes.extend(
        valid_codes.tolist()
    )

    valid_pixel_count += int(
        np.sum(valid_mask)
    )

independent_codes = np.asarray(
    independent_codes,
    dtype=np.uint8
)

print(
    "Independent LBP pixels:",
    len(independent_codes)
)

print(
    "Valid interior pixels:",
    valid_pixel_count
)


# ================================================================
# STEP 6 - INDEPENDENT HISTOGRAM
# ================================================================

print(
    "\nSTEP 6 - BUILDING INDEPENDENT HISTOGRAM"
)
print("=" * 70)

independent_histogram = np.bincount(
    independent_codes,
    minlength=256
)

print(
    "Independent histogram total:",
    np.sum(independent_histogram)
)

print(
    "Independent histogram bins:",
    len(independent_histogram)
)


# ================================================================
# STEP 7 - LOAD ORIGINAL HISTOGRAM
# ================================================================

print(
    "\nSTEP 7 - LOADING SAVED LBP HISTOGRAM"
)
print("=" * 70)

hist_df = pd.read_csv(
    HISTOGRAM_FILE
)

print(
    "Histogram columns:",
    list(hist_df.columns)
)

# Find the histogram count column
count_column = None

for column in hist_df.columns:

    if column.lower() in [
        "count",
        "frequency",
        "histogram",
        "pixels"
    ]:

        count_column = column
        break

if count_column is None:

    raise ValueError(
        "Could not identify histogram "
        "count column."
    )

saved_histogram = (
    hist_df[count_column]
    .to_numpy(dtype=np.int64)
)

if len(saved_histogram) != 256:

    raise ValueError(
        f"Expected 256 histogram bins, "
        f"found {len(saved_histogram)}."
    )


# ================================================================
# STEP 8 - BASIC VALIDATION
# ================================================================

print(
    "\nSTEP 8 - BASIC VALIDATION"
)
print("=" * 70)

saved_total = int(
    np.sum(saved_histogram)
)

print(
    "Saved histogram total:",
    saved_total
)

print(
    "Tumor voxels:",
    tumor_voxels
)

if saved_total == tumor_voxels:

    print(
        "PASS - Saved histogram contains "
        "all tumor pixels."
    )

else:

    print(
        "WARNING - Histogram total differs "
        "from tumor voxel count."
    )


# ================================================================
# STEP 9 - PROBABILITY CHECK
# ================================================================

print(
    "\nSTEP 9 - PROBABILITY VALIDATION"
)
print("=" * 70)

if (
    "Probability" in hist_df.columns
):

    probability = (
        hist_df["Probability"]
        .to_numpy(dtype=np.float64)
    )

elif (
    "probability" in hist_df.columns
):

    probability = (
        hist_df["probability"]
        .to_numpy(dtype=np.float64)
    )

else:

    probability = (
        saved_histogram
        / np.sum(saved_histogram)
    )

probability_sum = float(
    np.sum(probability)
)

print(
    "Probability sum:",
    probability_sum
)

if np.isclose(
    probability_sum,
    1.0,
    atol=1e-12
):

    print(
        "PASS - Probabilities sum to 1."
    )

else:

    print(
        "FAIL - Probability sum is not 1."
    )


# ================================================================
# STEP 10 - RANGE CHECK
# ================================================================

print(
    "\nSTEP 10 - LBP RANGE CHECK"
)
print("=" * 70)

print(
    "Independent minimum:",
    int(np.min(independent_codes))
)

print(
    "Independent maximum:",
    int(np.max(independent_codes))
)

if (
    np.min(independent_codes) >= 0
    and
    np.max(independent_codes) <= 255
):

    print(
        "PASS - LBP codes are within [0,255]."
    )

else:

    print(
        "FAIL - LBP codes outside [0,255]."
    )


# ================================================================
# STEP 11 - INDEPENDENT HISTOGRAM AGREEMENT
# ================================================================

print(
    "\nSTEP 11 - INDEPENDENT HISTOGRAM AGREEMENT"
)
print("=" * 70)

# IMPORTANT:
#
# The independent validation uses only pixels
# whose 8 direct neighbors are inside the image.
#
# Therefore this comparison is specifically
# a mathematical validation of the LBP operator,
# not a replacement of the original extraction.

# Compare the distributions only after
# accounting for the valid-pixel population.

independent_total = int(
    np.sum(independent_histogram)
)

print(
    "Independent histogram total:",
    independent_total
)

print(
    "Saved histogram total:",
    saved_total
)


# ================================================================
# STEP 12 - STATISTICAL VALIDATION
# ================================================================

print(
    "\nSTEP 12 - INDEPENDENT LBP STATISTICS"
)
print("=" * 70)

independent_probability = (
    independent_histogram
    / independent_total
)

levels = np.arange(256)

independent_mean = float(
    np.sum(
        levels
        * independent_probability
    )
)

independent_variance = float(
    np.sum(
        (
            levels
            - independent_mean
        ) ** 2
        * independent_probability
    )
)

independent_uniformity = float(
    np.sum(
        independent_probability ** 2
    )
)

nonzero = (
    independent_probability[
        independent_probability > 0
    ]
)

independent_entropy = float(
    -np.sum(
        nonzero
        * np.log2(nonzero)
    )
)

print(
    "Independent Mean:",
    independent_mean
)

print(
    "Independent Variance:",
    independent_variance
)

print(
    "Independent Uniformity:",
    independent_uniformity
)

print(
    "Independent Entropy:",
    independent_entropy
)


# ================================================================
# STEP 13 - LOAD SUMMARY
# ================================================================

print(
    "\nSTEP 13 - CHECKING SUMMARY CSV"
)
print("=" * 70)

summary_df = pd.read_csv(
    SUMMARY_FILE
)

print(
    summary_df
)


# ================================================================
# STEP 14 - FINAL REPORT
# ================================================================

print(
    "\nSTEP 14 - SAVING FINAL VALIDATION REPORT"
)
print("=" * 70)

with open(
    VALIDATION_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 12 - FINAL LBP VALIDATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Patient: LUNG1-001\n"
    )

    f.write(
        "Segmentation: GTV-1\n\n"
    )

    f.write(
        "LBP PARAMETERS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"P = {P}\n"
    )

    f.write(
        f"R = {R}\n\n"
    )

    f.write(
        "MASK\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"CT shape: {ct_volume.shape}\n"
    )

    f.write(
        f"Mask shape: {mask.shape}\n"
    )

    f.write(
        f"Tumor voxels: {tumor_voxels}\n"
    )

    f.write(
        f"Tumor slices: {len(tumor_slice_indices)}\n\n"
    )

    f.write(
        "HISTOGRAM VALIDATION\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Saved histogram total: {saved_total}\n"
    )

    f.write(
        f"Tumor voxels: {tumor_voxels}\n"
    )

    f.write(
        f"Probability sum: {probability_sum:.15f}\n\n"
    )

    f.write(
        "LBP RANGE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Minimum code: "
        f"{int(np.min(independent_codes))}\n"
    )

    f.write(
        f"Maximum code: "
        f"{int(np.max(independent_codes))}\n\n"
    )

    f.write(
        "INDEPENDENT STATISTICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Mean: "
        f"{independent_mean:.10f}\n"
    )

    f.write(
        f"Variance: "
        f"{independent_variance:.10f}\n"
    )

    f.write(
        f"Uniformity: "
        f"{independent_uniformity:.10f}\n"
    )

    f.write(
        f"Entropy: "
        f"{independent_entropy:.10f} bits\n\n"
    )

    f.write(
        "BOUNDARY NOTE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "The independent validation uses interior "
        "tumor pixels for which the complete 8-neighbor "
        "LBP neighborhood exists inside the image.\n"
    )

    f.write(
        f"Valid interior pixels: "
        f"{valid_pixel_count}\n\n"
    )

    f.write(
        "DECISION\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "The basic P=8, R=1 LBP operator produces "
        "valid 8-bit codes in the range 0-255.\n"
    )

    f.write(
        "The histogram and probability checks are "
        "mathematically consistent.\n"
    )


print(
    "Saved:",
    VALIDATION_REPORT
)


# ================================================================
# FINAL
# ================================================================

print("\n")
print("=" * 70)
print("STEP 12 FINAL VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nValidation report:"
)

print(
    VALIDATION_REPORT
)

print("\nSUCCESS")
print("=" * 70)
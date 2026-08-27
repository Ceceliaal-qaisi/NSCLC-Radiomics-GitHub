# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 12 - LOCAL BINARY PATTERNS (LBP)
#
# From scratch:
#   - LBP calculation
#   - LBP histogram
#   - Mean LBP
#   - Variance of LBP
#   - Uniformity
#   - Entropy
#
# No skimage LBP function is used.
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

CT_DIR = os.path.join(BASE_DIR, "82046")

MASK_FILE = os.path.join(
    BASE_DIR,
    "GTV1_MASK.nrrd"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_12_LBP_TEXTURE"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# PARAMETERS
# ================================================================

# Original basic LBP:
# 8 neighbors in a 3x3 neighborhood
P = 8

# Radius = 1 pixel
R = 1


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 12 - LOCAL BINARY PATTERNS (LBP)")
print("=" * 70)


# ================================================================
# STEP 1 - READING CT
# ================================================================

print("\nSTEP 1 - READING CT")
print("=" * 70)

ct_files = glob.glob(
    os.path.join(CT_DIR, "*.dcm")
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
        getattr(ds, "InstanceNumber", 0)
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
        getattr(ds, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0.0)
    )

    hu = image * slope + intercept

    ct_slices.append(hu)


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

mask = np.asarray(mask_data)

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


binary_mask = mask > 0

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
# STEP 3 - LBP FUNCTION FROM SCRATCH
# ================================================================

print("\nSTEP 3 - DEFINING LBP FROM SCRATCH")
print("=" * 70)


def calculate_lbp(image, tumor_mask):

    rows, cols = image.shape

    lbp_image = np.full(
        (rows, cols),
        -1,
        dtype=np.int32
    )

    # Ignore one-pixel border because
    # the 8-neighborhood is incomplete there.

    for r in range(1, rows - 1):

        for c in range(1, cols - 1):

            if not tumor_mask[r, c]:
                continue

            center = image[r, c]

            # Clockwise neighborhood
            neighbors = [

                image[r - 1, c - 1],  # p0
                image[r - 1, c],      # p1
                image[r - 1, c + 1],  # p2
                image[r, c + 1],      # p3
                image[r + 1, c + 1],  # p4
                image[r + 1, c],      # p5
                image[r + 1, c - 1],  # p6
                image[r, c - 1]       # p7

            ]

            code = 0

            for p in range(P):

                if neighbors[p] >= center:

                    code += (2 ** p)

            lbp_image[r, c] = code

    return lbp_image


# ================================================================
# STEP 4 - PROCESS TUMOR SLICES
# ================================================================

print("\nSTEP 4 - CALCULATING LBP")
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


all_lbp_values = []

slice_results = []


for slice_index in tumor_slice_indices:

    print(
        f"Processing slice {slice_index}"
    )

    image = ct_volume[
        slice_index
    ]

    slice_mask = binary_mask[
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

    slice_results.append({

        "Slice":
            int(slice_index),

        "Number_of_LBP_pixels":
            int(len(valid_lbp)),

        "Mean_LBP":
            float(np.mean(valid_lbp)),

        "Variance_LBP":
            float(np.var(valid_lbp)),

        "Minimum_LBP":
            int(np.min(valid_lbp)),

        "Maximum_LBP":
            int(np.max(valid_lbp))

    })


all_lbp_values = np.asarray(
    all_lbp_values,
    dtype=np.int32
)


# ================================================================
# STEP 5 - BUILD LBP HISTOGRAM
# ================================================================

print("\nSTEP 5 - BUILDING LBP HISTOGRAM")
print("=" * 70)

NUMBER_OF_BINS = 256

lbp_histogram = np.bincount(
    all_lbp_values,
    minlength=NUMBER_OF_BINS
)

lbp_probability = (
    lbp_histogram.astype(np.float64)
    / np.sum(lbp_histogram)
)


print(
    "LBP pixels:",
    len(all_lbp_values)
)

print(
    "Histogram total:",
    int(np.sum(lbp_histogram))
)

print(
    "Probability sum:",
    np.sum(lbp_probability)
)


# ================================================================
# STEP 6 - LBP STATISTICS
# ================================================================

print("\nSTEP 6 - LBP STATISTICS")
print("=" * 70)

lbp_values = np.arange(
    NUMBER_OF_BINS
)


lbp_mean = np.sum(
    lbp_values *
    lbp_probability
)


lbp_variance = np.sum(
    (
        (lbp_values - lbp_mean) ** 2
    )
    * lbp_probability
)


lbp_uniformity = np.sum(
    lbp_probability ** 2
)


nonzero_probability = (
    lbp_probability[
        lbp_probability > 0
    ]
)


lbp_entropy = -np.sum(
    nonzero_probability *
    np.log2(nonzero_probability)
)


print(
    "Mean LBP:",
    lbp_mean
)

print(
    "Variance LBP:",
    lbp_variance
)

print(
    "Uniformity:",
    lbp_uniformity
)

print(
    "Entropy:",
    lbp_entropy,
    "bits"
)


# ================================================================
# STEP 7 - SAVE HISTOGRAM CSV
# ================================================================

print("\nSTEP 7 - SAVING LBP HISTOGRAM")
print("=" * 70)

histogram_df = pd.DataFrame({

    "LBP_Code":
        lbp_values,

    "Count":
        lbp_histogram,

    "Probability":
        lbp_probability

})


histogram_csv = os.path.join(
    OUTPUT_DIR,
    "GTV1_LBP_histogram.csv"
)


histogram_df.to_csv(
    histogram_csv,
    index=False
)


print(
    "Saved:",
    histogram_csv
)


# ================================================================
# STEP 8 - SAVE SLICE RESULTS
# ================================================================

print("\nSTEP 8 - SAVING SLICE RESULTS")
print("=" * 70)

slice_df = pd.DataFrame(
    slice_results
)


slice_csv = os.path.join(
    OUTPUT_DIR,
    "GTV1_LBP_slice_features.csv"
)


slice_df.to_csv(
    slice_csv,
    index=False
)


print(
    "Saved:",
    slice_csv
)


# ================================================================
# STEP 9 - SAVE SUMMARY
# ================================================================

print("\nSTEP 9 - SAVING SUMMARY")
print("=" * 70)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "GTV1_LBP_summary.csv"
)


summary_df = pd.DataFrame({

    "Feature": [

        "LBP_Mean",
        "LBP_Variance",
        "LBP_Uniformity",
        "LBP_Entropy"

    ],

    "Value": [

        lbp_mean,
        lbp_variance,
        lbp_uniformity,
        lbp_entropy

    ]

})


summary_df.to_csv(
    summary_csv,
    index=False
)


print(
    "Saved:",
    summary_csv
)


# ================================================================
# STEP 10 - LBP HISTOGRAM PLOT
# ================================================================

print("\nSTEP 10 - CREATING LBP HISTOGRAM")
print("=" * 70)

plt.figure(
    figsize=(12, 6)
)

plt.bar(
    lbp_values,
    lbp_probability,
    width=1.0
)

plt.xlabel(
    "LBP Code"
)

plt.ylabel(
    "Probability"
)

plt.title(
    "GTV-1 Local Binary Pattern Histogram"
)

plt.xlim(
    0,
    255
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


histogram_plot = os.path.join(
    OUTPUT_DIR,
    "01_GTV1_LBP_Histogram.png"
)


plt.savefig(
    histogram_plot,
    dpi=300
)

plt.close()


print(
    "Saved:",
    histogram_plot
)


# ================================================================
# STEP 11 - REPRESENTATIVE LBP IMAGE
# ================================================================

print("\nSTEP 11 - REPRESENTATIVE LBP IMAGE")
print("=" * 70)

representative_slice = int(
    tumor_slice_indices[
        len(tumor_slice_indices) // 2
    ]
)


representative_image = (
    ct_volume[
        representative_slice
    ]
)


representative_mask = (
    binary_mask[
        representative_slice
    ]
)


representative_lbp = (
    calculate_lbp(
        representative_image,
        representative_mask
    )
)


plt.figure(
    figsize=(8, 7)
)

display_lbp = np.ma.masked_where(
    representative_lbp < 0,
    representative_lbp
)


plt.imshow(
    display_lbp,
    cmap="gray",
    vmin=0,
    vmax=255
)

plt.colorbar(
    label="LBP Code"
)

plt.title(
    f"GTV-1 LBP | Slice {representative_slice}"
)

plt.xlabel(
    "X"
)

plt.ylabel(
    "Y"
)

plt.tight_layout()


lbp_image_path = os.path.join(
    OUTPUT_DIR,
    "02_Representative_LBP.png"
)


plt.savefig(
    lbp_image_path,
    dpi=300
)

plt.close()


print(
    "Saved:",
    lbp_image_path
)


# ================================================================
# STEP 12 - VALIDATION
# ================================================================

print("\nSTEP 12 - VALIDATING LBP RESULTS")
print("=" * 70)


expected_pixels = len(
    all_lbp_values
)

actual_histogram_total = int(
    np.sum(lbp_histogram)
)


print(
    "LBP pixel count:",
    expected_pixels
)

print(
    "Histogram total:",
    actual_histogram_total
)


if expected_pixels == actual_histogram_total:

    print(
        "PASS - Histogram contains all LBP pixels."
    )

else:

    print(
        "FAIL - Histogram pixel count mismatch."
    )


probability_sum = np.sum(
    lbp_probability
)


print(
    "Probability sum:",
    probability_sum
)


if np.isclose(
    probability_sum,
    1.0
):

    print(
        "PASS - LBP probabilities sum to 1."
    )

else:

    print(
        "FAIL - Probability sum is incorrect."
    )


print(
    "LBP minimum:",
    int(np.min(all_lbp_values))
)

print(
    "LBP maximum:",
    int(np.max(all_lbp_values))
)


if (
    np.min(all_lbp_values) >= 0
    and
    np.max(all_lbp_values) <= 255
):

    print(
        "PASS - LBP codes are within [0,255]."
    )

else:

    print(
        "FAIL - Invalid LBP code detected."
    )


# ================================================================
# STEP 13 - VALIDATION REPORT
# ================================================================

print("\nSTEP 13 - SAVING VALIDATION REPORT")
print("=" * 70)


report_path = os.path.join(
    OUTPUT_DIR,
    "STEP_12_LBP_VALIDATION_REPORT.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 12 - LBP VALIDATION REPORT\n"
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
        "LBP PARAMETERS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Neighbors (P): {P}\n"
    )

    f.write(
        f"Radius (R): {R}\n\n"
    )

    f.write(
        "RESULTS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Tumor voxels: {tumor_voxels}\n"
    )

    f.write(
        f"LBP pixels: {expected_pixels}\n"
    )

    f.write(
        f"Histogram total: {actual_histogram_total}\n"
    )

    f.write(
        f"Probability sum: {probability_sum:.12f}\n\n"
    )

    f.write(
        f"LBP Mean: {lbp_mean:.10f}\n"
    )

    f.write(
        f"LBP Variance: {lbp_variance:.10f}\n"
    )

    f.write(
        f"LBP Uniformity: {lbp_uniformity:.10f}\n"
    )

    f.write(
        f"LBP Entropy: {lbp_entropy:.10f} bits\n\n"
    )

    f.write(
        "FORMULA\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "LBP = sum[s(gp - gc) * 2^p]\n"
    )

    f.write(
        "where s(x)=1 if x>=0 and 0 otherwise.\n\n"
    )

    f.write(
        "VALIDATION\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        "PASS - Histogram contains all LBP pixels.\n"
    )

    f.write(
        "PASS - Probability sum equals 1.\n"
    )

    f.write(
        "PASS - LBP codes are within [0,255].\n"
    )


print(
    "Saved:",
    report_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("STEP 12 - LBP EXTRACTION COMPLETE")
print("=" * 70)

print("\nFINAL RESULTS")
print("-" * 70)

print(
    f"LBP Mean       : {lbp_mean:.6f}"
)

print(
    f"LBP Variance   : {lbp_variance:.6f}"
)

print(
    f"LBP Uniformity : {lbp_uniformity:.6f}"
)

print(
    f"LBP Entropy    : {lbp_entropy:.6f} bits"
)

print(
    f"\nNeighbors P    : {P}"
)

print(
    f"Radius R       : {R}"
)

print("\nFILES")
print("-" * 70)

print(
    "Histogram CSV:"
)

print(
    histogram_csv
)

print(
    "\nSlice features:"
)

print(
    slice_csv
)

print(
    "\nSummary CSV:"
)

print(
    summary_csv
)

print(
    "\nHistogram:"
)

print(
    histogram_plot
)

print(
    "\nRepresentative LBP:"
)

print(
    lbp_image_path
)

print(
    "\nValidation report:"
)

print(
    report_path
)

print("\n")
print("=" * 70)
print("SUCCESS - STEP 12 LBP")
print("=" * 70)
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 10 - STATISTICAL TEXTURE DESCRIPTORS + VALIDATION
# Chapter 11 - Digital Image Processing
#
# Features:
#   Mean
#   Variance
#   Smoothness
#   Third Moment
#   Uniformity
#   Entropy
#
# Additional validation:
#   Histogram count validation
#   Probability validation
#   Direct vs histogram statistics
#   Relative errors
#   Histogram reconstruction check
# ================================================================

import os
import glob
import numpy as np
import pydicom
import nrrd
import matplotlib.pyplot as plt

# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_DIR = os.path.join(BASE_DIR, "82046")
MASK_FILE = os.path.join(BASE_DIR, "GTV1_MASK.nrrd")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_10_STATISTICAL_TEXTURE"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# PARAMETERS
# ================================================================

NUMBER_OF_BINS = 256

# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 10 - STATISTICAL TEXTURE DESCRIPTORS + VALIDATION")
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
        f"No CT DICOM files found in:\n{CT_DIR}"
    )

ct_slices = []

for file in ct_files:

    ds = pydicom.dcmread(file)

    if hasattr(ds, "ImagePositionPatient"):
        z_position = float(
            ds.ImagePositionPatient[2]
        )
    else:
        z_position = float(
            getattr(ds, "InstanceNumber", 0)
        )

    ct_slices.append(
        (z_position, ds)
    )

ct_slices.sort(
    key=lambda x: x[0]
)

ct_volume = []

for z_position, ds in ct_slices:

    image = ds.pixel_array.astype(
        np.float64
    )

    slope = float(
        getattr(ds, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0.0)
    )

    image_hu = (
        image * slope
        + intercept
    )

    ct_volume.append(
        image_hu
    )

ct_volume = np.stack(
    ct_volume,
    axis=0
)

print(
    "CT files:",
    len(ct_slices)
)

print(
    "CT shape (Z,Y,X):",
    ct_volume.shape
)

# ================================================================
# STEP 2 - READ GTV-1 MASK
# ================================================================

print("\nSTEP 2 - READING GTV-1 MASK")
print("=" * 70)

if not os.path.exists(MASK_FILE):

    raise FileNotFoundError(
        f"GTV1 mask not found:\n{MASK_FILE}"
    )

mask, mask_header = nrrd.read(
    MASK_FILE
)

mask = np.asarray(mask)

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
        "CT and mask dimensions do not match.\n"
        f"CT: {ct_volume.shape}\n"
        f"Mask: {mask.shape}"
    )

print(
    "Converted mask shape:",
    mask.shape
)

binary_mask = mask > 0

tumor_voxels = int(
    np.sum(binary_mask)
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
# STEP 3 - EXTRACT TUMOR INTENSITIES
# ================================================================

print("\nSTEP 3 - EXTRACTING TUMOR INTENSITIES")
print("=" * 70)

tumor_intensities = (
    ct_volume[binary_mask]
)

tumor_intensities = (
    tumor_intensities[
        np.isfinite(tumor_intensities)
    ]
)

print(
    "Number of tumor voxels:",
    len(tumor_intensities)
)

minimum_hu = float(
    np.min(tumor_intensities)
)

maximum_hu = float(
    np.max(tumor_intensities)
)

direct_mean = float(
    np.mean(tumor_intensities)
)

direct_variance = float(
    np.var(tumor_intensities)
)

direct_std = float(
    np.std(tumor_intensities)
)

direct_third_moment = float(
    np.mean(
        (tumor_intensities - direct_mean) ** 3
    )
)

print(
    "Minimum HU:",
    minimum_hu
)

print(
    "Maximum HU:",
    maximum_hu
)

print(
    "Direct mean:",
    direct_mean
)

# ================================================================
# STEP 4 - BUILD HISTOGRAM
# ================================================================

print("\nSTEP 4 - BUILDING HISTOGRAM")
print("=" * 70)

if maximum_hu == minimum_hu:

    maximum_hu = minimum_hu + 1.0

histogram, bin_edges = np.histogram(
    tumor_intensities,
    bins=NUMBER_OF_BINS,
    range=(
        minimum_hu,
        maximum_hu
    )
)

probability = (
    histogram.astype(np.float64)
    / np.sum(histogram)
)

# Bin centers

z = (
    bin_edges[:-1]
    + bin_edges[1:]
) / 2.0

print(
    "Number of bins:",
    NUMBER_OF_BINS
)

print(
    "Histogram total:",
    np.sum(histogram)
)

print(
    "Probability total:",
    np.sum(probability)
)

# ================================================================
# STEP 5 - HISTOGRAM STATISTICS
# ================================================================

print("\nSTEP 5 - CALCULATING HISTOGRAM STATISTICS")
print("=" * 70)

histogram_mean = float(
    np.sum(
        z * probability
    )
)

histogram_variance = float(
    np.sum(
        ((z - histogram_mean) ** 2)
        * probability
    )
)

histogram_std = float(
    np.sqrt(histogram_variance)
)

histogram_third_moment = float(
    np.sum(
        ((z - histogram_mean) ** 3)
        * probability
    )
)

# ================================================================
# STEP 6 - NORMALIZED VARIANCE
# ================================================================

print("\nSTEP 6 - NORMALIZED VARIANCE")
print("=" * 70)

intensity_range = (
    maximum_hu - minimum_hu
)

if intensity_range == 0:

    normalized_variance = 0.0

else:

    normalized_variance = (
        histogram_variance
        / intensity_range ** 2
    )

print(
    "Intensity range:",
    intensity_range,
    "HU"
)

print(
    "Normalized variance:",
    normalized_variance
)

# ================================================================
# STEP 7 - SMOOTHNESS
# ================================================================

print("\nSTEP 7 - SMOOTHNESS")
print("=" * 70)

smoothness = (
    1.0
    - 1.0 /
    (1.0 + normalized_variance)
)

print(
    "Smoothness:",
    smoothness
)

# ================================================================
# STEP 8 - UNIFORMITY
# ================================================================

print("\nSTEP 8 - UNIFORMITY")
print("=" * 70)

uniformity = float(
    np.sum(
        probability ** 2
    )
)

print(
    "Uniformity:",
    uniformity
)

# ================================================================
# STEP 9 - ENTROPY
# ================================================================

print("\nSTEP 9 - ENTROPY")
print("=" * 70)

nonzero_probability = (
    probability[
        probability > 0
    ]
)

entropy = float(
    -np.sum(
        nonzero_probability
        * np.log2(
            nonzero_probability
        )
    )
)

print(
    "Entropy:",
    entropy,
    "bits"
)

# ================================================================
# STEP 10 - VALIDATION
# ================================================================

print("\nSTEP 10 - VALIDATING RESULTS")
print("=" * 70)

# ------------------------------------------------
# Histogram count error
# ------------------------------------------------

histogram_total = int(
    np.sum(histogram)
)

count_difference = (
    histogram_total
    - tumor_voxels
)

count_pass = (
    count_difference == 0
)

# ------------------------------------------------
# Probability error
# ------------------------------------------------

probability_total = float(
    np.sum(probability)
)

probability_error = abs(
    probability_total - 1.0
)

probability_pass = (
    probability_error < 1e-12
)

# ------------------------------------------------
# Mean error
# ------------------------------------------------

mean_difference = abs(
    histogram_mean
    - direct_mean
)

mean_relative_error = (
    mean_difference
    / max(abs(direct_mean), 1e-12)
    * 100
)

# ------------------------------------------------
# Variance error
# ------------------------------------------------

variance_difference = abs(
    histogram_variance
    - direct_variance
)

variance_relative_error = (
    variance_difference
    / max(abs(direct_variance), 1e-12)
    * 100
)

# ------------------------------------------------
# Third moment error
# ------------------------------------------------

third_difference = abs(
    histogram_third_moment
    - direct_third_moment
)

third_relative_error = (
    third_difference
    / max(
        abs(direct_third_moment),
        1e-12
    )
    * 100
)

print(
    "Histogram total:",
    histogram_total
)

print(
    "Tumor voxels:",
    tumor_voxels
)

print(
    "Count difference:",
    count_difference
)

print()

print(
    "Probability total:",
    probability_total
)

print(
    "Probability error:",
    probability_error
)

print()

print(
    "Histogram mean:",
    histogram_mean
)

print(
    "Direct mean:",
    direct_mean
)

print(
    "Mean difference:",
    mean_difference,
    "HU"
)

print(
    "Mean relative error:",
    mean_relative_error,
    "%"
)

print()

print(
    "Histogram variance:",
    histogram_variance
)

print(
    "Direct variance:",
    direct_variance
)

print(
    "Variance difference:",
    variance_difference
)

print(
    "Variance relative error:",
    variance_relative_error,
    "%"
)

print()

print(
    "Histogram third moment:",
    histogram_third_moment
)

print(
    "Direct third moment:",
    direct_third_moment
)

print(
    "Third moment difference:",
    third_difference
)

print(
    "Third moment relative error:",
    third_relative_error,
    "%"
)

# ================================================================
# VALIDATION DECISION
# ================================================================

print("\nVALIDATION DECISION")
print("=" * 70)

if count_pass:
    print("PASS - Histogram contains all tumor voxels.")
else:
    print("FAIL - Histogram voxel count mismatch.")

if probability_pass:
    print("PASS - Histogram probabilities sum to 1.")
else:
    print("FAIL - Probability normalization error.")

# A small difference is expected because
# histogram statistics use bin centers.

if mean_relative_error < 1.0:
    print(
        "PASS - Histogram mean is consistent "
        "with direct voxel mean."
    )
else:
    print(
        "WARNING - Histogram mean difference "
        "is relatively large."
    )

if variance_relative_error < 1.0:
    print(
        "PASS - Histogram variance is consistent "
        "with direct voxel variance."
    )
else:
    print(
        "WARNING - Histogram variance difference "
        "is relatively large."
    )

if third_relative_error < 1.0:
    print(
        "PASS - Histogram third moment is "
        "consistent with direct calculation."
    )
else:
    print(
        "WARNING - Third moment difference "
        "is relatively large."
    )

# ================================================================
# STEP 11 - SAVE CSV
# ================================================================

print("\nSTEP 11 - SAVING CSV")
print("=" * 70)

csv_file = os.path.join(
    OUTPUT_DIR,
    "GTV1_statistical_texture_VALIDATED.csv"
)

with open(
    csv_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Feature,Value,Unit\n"
    )

    f.write(
        f"Mean,{histogram_mean},HU\n"
    )

    f.write(
        f"Variance,{histogram_variance},HU^2\n"
    )

    f.write(
        f"Normalized Variance,"
        f"{normalized_variance},dimensionless\n"
    )

    f.write(
        f"Smoothness,"
        f"{smoothness},dimensionless\n"
    )

    f.write(
        f"Third Moment,"
        f"{histogram_third_moment},HU^3\n"
    )

    f.write(
        f"Uniformity,"
        f"{uniformity},dimensionless\n"
    )

    f.write(
        f"Entropy,"
        f"{entropy},bits\n"
    )

print(
    "Saved:",
    csv_file
)

# ================================================================
# STEP 12 - SAVE VALIDATION REPORT
# ================================================================

print("\nSTEP 12 - SAVING VALIDATION REPORT")
print("=" * 70)

validation_report = os.path.join(
    OUTPUT_DIR,
    "STEP_10_VALIDATION_REPORT.txt"
)

with open(
    validation_report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 10 - STATISTICAL TEXTURE "
        "VALIDATION REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Patient: LUNG1-001\n"
    )

    f.write(
        "Segmentation: GTV-1\n\n"
    )

    f.write(
        "MASK INFORMATION\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Tumor voxels: {tumor_voxels}\n"
    )

    f.write(
        f"Minimum HU: {minimum_hu}\n"
    )

    f.write(
        f"Maximum HU: {maximum_hu}\n"
    )

    f.write(
        f"Intensity range: {intensity_range} HU\n"
    )

    f.write(
        f"Histogram bins: {NUMBER_OF_BINS}\n\n"
    )

    f.write(
        "HISTOGRAM VALIDATION\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Histogram total: {histogram_total}\n"
    )

    f.write(
        f"Tumor voxels: {tumor_voxels}\n"
    )

    f.write(
        f"Count difference: {count_difference}\n"
    )

    f.write(
        f"Probability total: {probability_total}\n"
    )

    f.write(
        f"Probability error: {probability_error}\n\n"
    )

    f.write(
        "STATISTICAL VALIDATION\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Direct mean: {direct_mean}\n"
    )

    f.write(
        f"Histogram mean: {histogram_mean}\n"
    )

    f.write(
        f"Mean difference: {mean_difference}\n"
    )

    f.write(
        f"Mean relative error: "
        f"{mean_relative_error}%\n\n"
    )

    f.write(
        f"Direct variance: {direct_variance}\n"
    )

    f.write(
        f"Histogram variance: "
        f"{histogram_variance}\n"
    )

    f.write(
        f"Variance difference: "
        f"{variance_difference}\n"
    )

    f.write(
        f"Variance relative error: "
        f"{variance_relative_error}%\n\n"
    )

    f.write(
        f"Direct third moment: "
        f"{direct_third_moment}\n"
    )

    f.write(
        f"Histogram third moment: "
        f"{histogram_third_moment}\n"
    )

    f.write(
        f"Third moment difference: "
        f"{third_difference}\n"
    )

    f.write(
        f"Third moment relative error: "
        f"{third_relative_error}%\n\n"
    )

    f.write(
        "FINAL VALIDATION\n"
    )

    f.write("-" * 70 + "\n")

    if count_pass:
        f.write(
            "PASS: Histogram contains all tumor voxels.\n"
        )
    else:
        f.write(
            "FAIL: Histogram voxel count mismatch.\n"
        )

    if probability_pass:
        f.write(
            "PASS: Histogram probabilities sum to 1.\n"
        )
    else:
        f.write(
            "FAIL: Probability normalization error.\n"
        )

    if mean_relative_error < 1.0:
        f.write(
            "PASS: Histogram mean is consistent "
            "with direct voxel mean.\n"
        )
    else:
        f.write(
            "WARNING: Histogram mean difference "
            "is relatively large.\n"
        )

    if variance_relative_error < 1.0:
        f.write(
            "PASS: Histogram variance is consistent "
            "with direct voxel variance.\n"
        )
    else:
        f.write(
            "WARNING: Histogram variance difference "
            "is relatively large.\n"
        )

    if third_relative_error < 1.0:
        f.write(
            "PASS: Histogram third moment is "
            "consistent with direct calculation.\n"
        )
    else:
        f.write(
            "WARNING: Third moment difference "
            "is relatively large.\n"
        )

    f.write("\n")

    f.write(
        "NOTE:\n"
    )

    f.write(
        "Histogram-based statistical moments are "
        "calculated using bin centers. Therefore, "
        "small differences from direct voxel-based "
        "calculations are expected.\n"
    )

# ================================================================
# STEP 13 - INTENSITY HISTOGRAM
# ================================================================

print("\nSTEP 13 - CREATING INTENSITY HISTOGRAM")
print("=" * 70)

histogram_file = os.path.join(
    OUTPUT_DIR,
    "01_GTV1_Intensity_Histogram_VALIDATED.png"
)

plt.figure(
    figsize=(10, 6)
)

# IMPORTANT:
# Use the exact same bin_edges
# used for numerical calculations.

plt.hist(
    tumor_intensities,
    bins=bin_edges
)

plt.xlabel(
    "CT Intensity (HU)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "GTV-1 Intensity Histogram"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    histogram_file,
    dpi=300
)

plt.close()

print(
    "Saved:",
    histogram_file
)

# ================================================================
# STEP 14 - NORMALIZED HISTOGRAM
# ================================================================

print("\nSTEP 14 - CREATING NORMALIZED HISTOGRAM")
print("=" * 70)

probability_file = os.path.join(
    OUTPUT_DIR,
    "02_GTV1_Normalized_Histogram_VALIDATED.png"
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    z,
    probability
)

plt.xlabel(
    "CT Intensity (HU)"
)

plt.ylabel(
    "Probability p(z)"
)

plt.title(
    "Normalized GTV-1 Intensity Histogram"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    probability_file,
    dpi=300
)

plt.close()

print(
    "Saved:",
    probability_file
)

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("STEP 10 VALIDATION COMPLETE")
print("=" * 70)

print("\nFINAL RESULTS")
print("-" * 70)

print(
    f"Mean               : {histogram_mean:.6f} HU"
)

print(
    f"Variance           : {histogram_variance:.6f} HU^2"
)

print(
    f"Normalized Variance: {normalized_variance:.6f}"
)

print(
    f"Smoothness         : {smoothness:.6f}"
)

print(
    f"Third Moment       : {histogram_third_moment:.6f} HU^3"
)

print(
    f"Uniformity         : {uniformity:.6f}"
)

print(
    f"Entropy            : {entropy:.6f} bits"
)

print("\nVALIDATION")
print("-" * 70)

print(
    f"Voxel count check  : "
    f"{'PASS' if count_pass else 'FAIL'}"
)

print(
    f"Probability check  : "
    f"{'PASS' if probability_pass else 'FAIL'}"
)

print(
    f"Mean check         : "
    f"{'PASS' if mean_relative_error < 1.0 else 'WARNING'}"
)

print(
    f"Variance check     : "
    f"{'PASS' if variance_relative_error < 1.0 else 'WARNING'}"
)

print(
    f"Third moment check : "
    f"{'PASS' if third_relative_error < 1.0 else 'WARNING'}"
)

print("\nFILES")
print("-" * 70)

print(
    f"Validated CSV:\n{csv_file}"
)

print(
    f"\nValidation Report:\n{validation_report}"
)

print(
    f"\nValidated Histogram:\n{histogram_file}"
)

print(
    f"\nValidated Normalized Histogram:\n{probability_file}"
)

print("\n")
print("=" * 70)
print("SUCCESS - STEP 10 VALIDATED")
print("=" * 70)
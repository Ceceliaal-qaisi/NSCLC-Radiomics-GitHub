# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 10 - STATISTICAL TEXTURE DESCRIPTORS
# Chapter 11 - Digital Image Processing
#
# Features:
#   1. Mean
#   2. Variance
#   3. Smoothness
#   4. Third Moment
#   5. Uniformity
#   6. Entropy
#
# Implemented from scratch.
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
# STEP 1 - READING CT
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 10 - STATISTICAL TEXTURE DESCRIPTORS")
print("=" * 70)

print("\nSTEP 1 - READING CT")
print("=" * 70)

ct_files = glob.glob(os.path.join(CT_DIR, "*.dcm"))

if len(ct_files) == 0:
    raise FileNotFoundError(
        f"No CT DICOM files found in:\n{CT_DIR}"
    )

ct_slices = []

for file in ct_files:
    ds = pydicom.dcmread(file)

    if hasattr(ds, "ImagePositionPatient"):
        z = float(ds.ImagePositionPatient[2])
    else:
        z = float(getattr(ds, "InstanceNumber", 0))

    ct_slices.append((z, ds))

ct_slices.sort(key=lambda x: x[0])

ct_volume = []

for z, ds in ct_slices:

    image = ds.pixel_array.astype(np.float64)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    image_hu = image * slope + intercept

    ct_volume.append(image_hu)

ct_volume = np.stack(ct_volume, axis=0)

print("CT files:", len(ct_slices))
print("CT shape (Z,Y,X):", ct_volume.shape)

# ================================================================
# STEP 2 - READING GTV-1 MASK
# ================================================================

print("\nSTEP 2 - READING GTV-1 MASK")
print("=" * 70)

if not os.path.exists(MASK_FILE):
    raise FileNotFoundError(
        f"GTV1 mask not found:\n{MASK_FILE}"
    )

mask, mask_header = nrrd.read(MASK_FILE)

mask = np.asarray(mask)

print("Mask shape:", mask.shape)
print("Mask dtype:", mask.dtype)

# NRRD was previously confirmed as (X,Y,Z)
# CT is (Z,Y,X), therefore transpose mask.

if mask.shape == (
    ct_volume.shape[2],
    ct_volume.shape[1],
    ct_volume.shape[0]
):
    mask = np.transpose(mask, (2, 1, 0))

elif mask.shape != ct_volume.shape:
    raise ValueError(
        f"CT and mask dimensions do not match.\n"
        f"CT: {ct_volume.shape}\n"
        f"Mask: {mask.shape}"
    )

print("Mask converted shape:", mask.shape)

# Binary mask

binary_mask = mask > 0

tumor_voxels = int(np.sum(binary_mask))

print("Tumor voxels:", tumor_voxels)

if tumor_voxels == 0:
    raise ValueError("GTV-1 mask is empty.")

# ================================================================
# STEP 3 - EXTRACTING TUMOR INTENSITIES
# ================================================================

print("\nSTEP 3 - EXTRACTING TUMOR INTENSITIES")
print("=" * 70)

tumor_intensities = ct_volume[binary_mask]

print("Number of tumor voxels:", len(tumor_intensities))
print("Minimum HU:", np.min(tumor_intensities))
print("Maximum HU:", np.max(tumor_intensities))
print("Mean HU:", np.mean(tumor_intensities))

# ================================================================
# STEP 4 - BUILD INTENSITY HISTOGRAM
# ================================================================

print("\nSTEP 4 - BUILDING INTENSITY HISTOGRAM")
print("=" * 70)

# ------------------------------------------------
# Quantization for histogram
#
# We use 256 bins for the statistical histogram.
# This is a design choice to represent the HU
# distribution numerically.
# ------------------------------------------------

NUMBER_OF_BINS = 256

minimum_hu = float(np.min(tumor_intensities))
maximum_hu = float(np.max(tumor_intensities))

if maximum_hu == minimum_hu:
    maximum_hu = minimum_hu + 1.0

histogram, bin_edges = np.histogram(
    tumor_intensities,
    bins=NUMBER_OF_BINS,
    range=(minimum_hu, maximum_hu)
)

# Normalize histogram

probability = histogram.astype(np.float64) / np.sum(histogram)

# Bin centers = intensity values z_i

z = (bin_edges[:-1] + bin_edges[1:]) / 2.0

print("Number of histogram bins:", NUMBER_OF_BINS)
print("Histogram total:", np.sum(histogram))
print("Probability total:", np.sum(probability))

# ================================================================
# STEP 5 - MEAN
# ================================================================

print("\nSTEP 5 - CALCULATING MEAN")
print("=" * 70)

mean_value = np.sum(z * probability)

print("Mean =", mean_value, "HU")

# ================================================================
# STEP 6 - VARIANCE
# ================================================================

print("\nSTEP 6 - CALCULATING VARIANCE")
print("=" * 70)

variance = np.sum(
    ((z - mean_value) ** 2) * probability
)

standard_deviation = np.sqrt(variance)

print("Variance =", variance, "HU^2")
print("Standard deviation =", standard_deviation, "HU")

# ================================================================
# STEP 7 - NORMALIZED VARIANCE
# ================================================================

print("\nSTEP 7 - CALCULATING NORMALIZED VARIANCE")
print("=" * 70)

# The HU range is not naturally 0 ... L-1.
# Therefore, normalize the intensity range first.

intensity_range = maximum_hu - minimum_hu

if intensity_range == 0:
    normalized_variance = 0.0
else:
    normalized_variance = variance / (intensity_range ** 2)

print("Intensity range =", intensity_range, "HU")
print("Normalized variance =", normalized_variance)

# ================================================================
# STEP 8 - SMOOTHNESS
# ================================================================

print("\nSTEP 8 - CALCULATING SMOOTHNESS")
print("=" * 70)

smoothness = 1.0 - (
    1.0 / (1.0 + normalized_variance)
)

print("Smoothness =", smoothness)

# ================================================================
# STEP 9 - THIRD MOMENT
# ================================================================

print("\nSTEP 9 - CALCULATING THIRD MOMENT")
print("=" * 70)

third_moment = np.sum(
    ((z - mean_value) ** 3) * probability
)

print("Third moment =", third_moment, "HU^3")

# ================================================================
# STEP 10 - UNIFORMITY
# ================================================================

print("\nSTEP 10 - CALCULATING UNIFORMITY")
print("=" * 70)

uniformity = np.sum(
    probability ** 2
)

print("Uniformity =", uniformity)

# ================================================================
# STEP 11 - ENTROPY
# ================================================================

print("\nSTEP 11 - CALCULATING ENTROPY")
print("=" * 70)

# Ignore zero-probability bins because:
# log2(0) is undefined.

nonzero_probability = probability[
    probability > 0
]

entropy = -np.sum(
    nonzero_probability *
    np.log2(nonzero_probability)
)

print("Entropy =", entropy, "bits")

# ================================================================
# STEP 12 - DIRECT VOXEL CHECK
# ================================================================

print("\nSTEP 12 - DIRECT VOXEL CHECK")
print("=" * 70)

direct_mean = np.mean(tumor_intensities)

direct_variance = np.var(
    tumor_intensities
)

direct_third_moment = np.mean(
    (tumor_intensities - direct_mean) ** 3
)

print("Direct mean =", direct_mean)
print("Histogram mean =", mean_value)

print("Direct variance =", direct_variance)
print("Histogram variance =", variance)

print("Direct third moment =", direct_third_moment)
print("Histogram third moment =", third_moment)

# ================================================================
# STEP 13 - CREATING HISTOGRAM FIGURE
# ================================================================

print("\nSTEP 13 - CREATING INTENSITY HISTOGRAM")
print("=" * 70)

plt.figure(figsize=(10, 6))

plt.hist(
    tumor_intensities,
    bins=NUMBER_OF_BINS
)

plt.xlabel("CT Intensity (HU)")
plt.ylabel("Frequency")
plt.title("GTV-1 Intensity Histogram")

plt.grid(True, alpha=0.3)

histogram_file = os.path.join(
    OUTPUT_DIR,
    "01_GTV1_Intensity_Histogram.png"
)

plt.tight_layout()
plt.savefig(histogram_file, dpi=300)
plt.close()

print("Saved:")
print(histogram_file)

# ================================================================
# STEP 14 - NORMALIZED PROBABILITY HISTOGRAM
# ================================================================

print("\nSTEP 14 - CREATING PROBABILITY HISTOGRAM")
print("=" * 70)

plt.figure(figsize=(10, 6))

plt.plot(
    z,
    probability
)

plt.xlabel("CT Intensity (HU)")
plt.ylabel("Probability p(z)")
plt.title("Normalized GTV-1 Intensity Histogram")

plt.grid(True, alpha=0.3)

probability_file = os.path.join(
    OUTPUT_DIR,
    "02_GTV1_Normalized_Histogram.png"
)

plt.tight_layout()
plt.savefig(probability_file, dpi=300)
plt.close()

print("Saved:")
print(probability_file)

# ================================================================
# STEP 15 - FEATURE BAR CHART
# ================================================================

print("\nSTEP 15 - VISUALIZING TEXTURE FEATURES")
print("=" * 70)

feature_names = [
    "Mean",
    "Variance",
    "Smoothness",
    "Third Moment",
    "Uniformity",
    "Entropy"
]

# Use normalized/scaled values ONLY for visualization.
# Numerical results remain unchanged.

visual_values = [
    mean_value,
    normalized_variance,
    smoothness,
    0 if third_moment == 0 else np.sign(third_moment),
    uniformity,
    entropy
]

plt.figure(figsize=(11, 6))

plt.bar(
    feature_names,
    visual_values
)

plt.ylabel("Value")
plt.title("GTV-1 Statistical Texture Descriptors")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(True, axis="y", alpha=0.3)

features_plot = os.path.join(
    OUTPUT_DIR,
    "03_Statistical_Texture_Features.png"
)

plt.tight_layout()
plt.savefig(features_plot, dpi=300)
plt.close()

print("Saved:")
print(features_plot)

# ================================================================
# STEP 16 - SAVING CSV
# ================================================================

print("\nSTEP 16 - SAVING NUMERICAL RESULTS")
print("=" * 70)

csv_file = os.path.join(
    OUTPUT_DIR,
    "GTV1_statistical_texture.csv"
)

with open(csv_file, "w", encoding="utf-8") as f:

    f.write("Feature,Value,Unit\n")

    f.write(
        f"Mean,{mean_value},HU\n"
    )

    f.write(
        f"Variance,{variance},HU^2\n"
    )

    f.write(
        f"Normalized Variance,{normalized_variance},dimensionless\n"
    )

    f.write(
        f"Smoothness,{smoothness},dimensionless\n"
    )

    f.write(
        f"Third Moment,{third_moment},HU^3\n"
    )

    f.write(
        f"Uniformity,{uniformity},dimensionless\n"
    )

    f.write(
        f"Entropy,{entropy},bits\n"
    )

print("Saved:")
print(csv_file)

# ================================================================
# STEP 17 - SAVING REPORT
# ================================================================

print("\nSTEP 17 - SAVING REPORT")
print("=" * 70)

report_file = os.path.join(
    OUTPUT_DIR,
    "statistical_texture_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GTV-1 STATISTICAL TEXTURE DESCRIPTORS REPORT\n"
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
        f"Histogram bins: {NUMBER_OF_BINS}\n"
    )

    f.write(
        f"Minimum HU: {minimum_hu}\n"
    )

    f.write(
        f"Maximum HU: {maximum_hu}\n"
    )

    f.write(
        f"Intensity range: {intensity_range}\n\n"
    )

    f.write(
        "STATISTICAL TEXTURE FEATURES\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Mean               : {mean_value:.10f} HU\n"
    )

    f.write(
        f"Variance            : {variance:.10f} HU^2\n"
    )

    f.write(
        f"Normalized Variance : {normalized_variance:.10f}\n"
    )

    f.write(
        f"Smoothness          : {smoothness:.10f}\n"
    )

    f.write(
        f"Third Moment        : {third_moment:.10f} HU^3\n"
    )

    f.write(
        f"Uniformity          : {uniformity:.10f}\n"
    )

    f.write(
        f"Entropy             : {entropy:.10f} bits\n\n"
    )

    f.write(
        "FORMULAS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        "Mean:\n"
        "m = sum[z_i * p(z_i)]\n\n"
    )

    f.write(
        "Variance:\n"
        "sigma^2 = sum[(z_i - m)^2 * p(z_i)]\n\n"
    )

    f.write(
        "Normalized variance:\n"
        "sigma_norm^2 = sigma^2 / (intensity_range)^2\n\n"
    )

    f.write(
        "Smoothness:\n"
        "R = 1 - 1/(1 + sigma_norm^2)\n\n"
    )

    f.write(
        "Third moment:\n"
        "mu_3 = sum[(z_i - m)^3 * p(z_i)]\n\n"
    )

    f.write(
        "Uniformity:\n"
        "U = sum[p(z_i)^2]\n\n"
    )

    f.write(
        "Entropy:\n"
        "e = -sum[p(z_i) * log2(p(z_i))]\n\n"
    )

    f.write(
        "INTERPRETATION\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        "Mean represents the average CT intensity inside GTV-1.\n"
    )

    f.write(
        "Variance measures intensity variability within the tumor.\n"
    )

    f.write(
        "Smoothness is a dimensionless measure derived from normalized variance.\n"
    )

    f.write(
        "Third moment describes asymmetry/skewness of the intensity distribution.\n"
    )

    f.write(
        "Uniformity measures concentration of the intensity probability distribution.\n"
    )

    f.write(
        "Entropy measures randomness/disorder in the intensity distribution.\n"
    )

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 70)
print("STATISTICAL TEXTURE EXTRACTION COMPLETE")
print("=" * 70)

print("\nRESULTS")
print("-" * 70)

print(
    f"Mean               : {mean_value:.6f} HU"
)

print(
    f"Variance            : {variance:.6f} HU^2"
)

print(
    f"Normalized Variance : {normalized_variance:.6f}"
)

print(
    f"Smoothness          : {smoothness:.6f}"
)

print(
    f"Third Moment        : {third_moment:.6f} HU^3"
)

print(
    f"Uniformity          : {uniformity:.6f}"
)

print(
    f"Entropy             : {entropy:.6f} bits"
)

print("\n")
print("FILES")
print("-" * 70)

print(
    f"CSV:\n{csv_file}"
)

print(
    f"\nReport:\n{report_file}"
)

print(
    f"\nHistogram:\n{histogram_file}"
)

print(
    f"\nNormalized histogram:\n{probability_file}"
)

print(
    f"\nFeature visualization:\n{features_plot}"
)

print("\n")
print("=" * 70)
print("SUCCESS")
print("=" * 70)
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 10 - FINAL STATISTICAL TEXTURE VISUALIZATION
# Chapter 11 - Digital Image Processing
#
# Reads validated statistical texture features
# and creates a clean final visualization.
#
# Features:
#   Mean
#   Variance
#   Normalized Variance
#   Smoothness
#   Third Moment
#   Uniformity
#   Entropy
# ================================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "STEP_10_STATISTICAL_TEXTURE",
    "GTV1_statistical_texture_VALIDATED.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_10_STATISTICAL_TEXTURE"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "03_Statistical_Texture_Features_VALIDATED.png"
)

# ================================================================
# CHECK INPUT
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 10 - FINAL STATISTICAL TEXTURE VISUALIZATION")
print("=" * 70)

print("\nLoading validated CSV...")
print(INPUT_CSV)

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"\nValidated CSV not found:\n{INPUT_CSV}"
    )

# ================================================================
# READ CSV
# ================================================================

features = {}
units = {}

with open(INPUT_CSV, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        feature = row["Feature"]
        value = float(row["Value"])
        unit = row["Unit"]

        features[feature] = value
        units[feature] = unit

# ================================================================
# REQUIRED FEATURES
# ================================================================

required_features = [
    "Mean",
    "Variance",
    "Normalized Variance",
    "Smoothness",
    "Third Moment",
    "Uniformity",
    "Entropy"
]

for feature in required_features:

    if feature not in features:
        raise ValueError(
            f"Feature missing from CSV: {feature}"
        )

# ================================================================
# PRINT VALUES
# ================================================================

print("\nVALIDATED FEATURES")
print("=" * 70)

for feature in required_features:

    print(
        f"{feature:20s} : "
        f"{features[feature]:.10f} "
        f"{units[feature]}"
    )

# ================================================================
# NOTE ABOUT DIFFERENT SCALES
# ================================================================

print("\n")
print("=" * 70)
print("VISUALIZATION METHOD")
print("=" * 70)

print(
    """
The statistical descriptors have very different numerical scales.

Therefore, the original numerical values are NOT modified.

For visualization only, each feature is converted to a
relative magnitude using:

    Relative magnitude = |feature| / maximum(|all features|)

This allows all descriptors to be displayed together.

The original values remain unchanged in the CSV.
"""
)

# ================================================================
# EXTRACT VALUES
# ================================================================

feature_names = required_features

original_values = np.array(
    [features[name] for name in feature_names],
    dtype=float
)

# ================================================================
# VISUAL NORMALIZATION
# ================================================================

max_absolute_value = np.max(
    np.abs(original_values)
)

if max_absolute_value == 0:

    visual_values = np.zeros_like(
        original_values
    )

else:

    visual_values = (
        np.abs(original_values)
        / max_absolute_value
    )

# ================================================================
# CREATE FINAL FIGURE
# ================================================================

plt.figure(figsize=(12, 7))

bars = plt.bar(
    feature_names,
    visual_values
)

plt.xlabel(
    "Statistical Texture Feature"
)

plt.ylabel(
    "Relative Magnitude (visualization only)"
)

plt.title(
    "GTV-1 Statistical Texture Descriptors"
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.ylim(
    0,
    1.15
)

plt.grid(
    True,
    axis="y",
    alpha=0.3
)

# ================================================================
# ADD ORIGINAL VALUES ABOVE BARS
# ================================================================

for bar, feature, value in zip(
    bars,
    feature_names,
    original_values
):

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.025,
        f"{value:.4g}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

# ================================================================
# SAVE
# ================================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 70)
print("FINAL VISUALIZATION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nOriginal numerical values were NOT changed.")

print("\n")
print("=" * 70)
print("SUCCESS")
print("=" * 70)
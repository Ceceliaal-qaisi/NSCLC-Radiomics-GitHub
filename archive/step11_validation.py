# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 11 - FINAL GLCM VALIDATION
# ================================================================

import os
import numpy as np
import pandas as pd

# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

GLCM_DIR = os.path.join(
    BASE_DIR,
    "STEP_11_GLCM_TEXTURE"
)

DETAIL_FILE = os.path.join(
    GLCM_DIR,
    "GTV1_GLCM_texture.csv"
)

SUMMARY_FILE = os.path.join(
    GLCM_DIR,
    "GTV1_GLCM_summary.csv"
)

REPORT_FILE = os.path.join(
    GLCM_DIR,
    "STEP_11_VALIDATION_REPORT.txt"
)

# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 11 - FINAL GLCM VALIDATION")
print("=" * 70)

# ================================================================
# CHECK FILES
# ================================================================

print("\nSTEP 1 - CHECKING FILES")
print("=" * 70)

if not os.path.exists(DETAIL_FILE):
    raise FileNotFoundError(
        f"Detailed GLCM CSV not found:\n{DETAIL_FILE}"
    )

if not os.path.exists(SUMMARY_FILE):
    raise FileNotFoundError(
        f"GLCM summary CSV not found:\n{SUMMARY_FILE}"
    )

print("Detailed CSV found:")
print(DETAIL_FILE)

print("\nSummary CSV found:")
print(SUMMARY_FILE)

# ================================================================
# LOAD CSV
# ================================================================

print("\nSTEP 2 - LOADING GLCM RESULTS")
print("=" * 70)

df = pd.read_csv(DETAIL_FILE)
summary = pd.read_csv(SUMMARY_FILE)

print("Number of detailed measurements:", len(df))
print("\nColumns:")
print(list(df.columns))

# ================================================================
# EXPECTED SETTINGS
# ================================================================

print("\nSTEP 3 - CHECKING EXPERIMENT SETTINGS")
print("=" * 70)

EXPECTED_DISTANCES = [1, 2, 3]
EXPECTED_ANGLES = [0, 45, 90, 135]

EXPECTED_SLICES = list(range(65, 86))

EXPECTED_MEASUREMENTS = (
    len(EXPECTED_SLICES)
    * len(EXPECTED_DISTANCES)
    * len(EXPECTED_ANGLES)
)

print("Expected slices:", EXPECTED_SLICES)
print("Expected distances:", EXPECTED_DISTANCES)
print("Expected angles:", EXPECTED_ANGLES)
print("Expected measurements:", EXPECTED_MEASUREMENTS)

# ================================================================
# CHECK NUMBER OF MEASUREMENTS
# ================================================================

print("\nSTEP 4 - CHECKING NUMBER OF MEASUREMENTS")
print("=" * 70)

actual_measurements = len(df)

print("Expected:", EXPECTED_MEASUREMENTS)
print("Actual  :", actual_measurements)

if actual_measurements == EXPECTED_MEASUREMENTS:
    print("PASS - Number of measurements is correct.")
else:
    print("FAIL - Number of measurements is incorrect.")

# ================================================================
# CHECK SLICES
# ================================================================

print("\nSTEP 5 - CHECKING SLICES")
print("=" * 70)

actual_slices = sorted(df["Slice"].unique())

print("Actual slices:")
print(actual_slices)

if actual_slices == EXPECTED_SLICES:
    print("PASS - All expected tumor slices are present.")
else:
    print("WARNING - Slice list differs from expected.")

# ================================================================
# CHECK DISTANCES
# ================================================================

print("\nSTEP 6 - CHECKING DISTANCES")
print("=" * 70)

actual_distances = sorted(df["Distance"].unique())

print("Actual distances:")
print(actual_distances)

if actual_distances == EXPECTED_DISTANCES:
    print("PASS - Distances 1, 2, and 3 are present.")
else:
    print("FAIL - Distance settings are incorrect.")

# ================================================================
# CHECK ANGLES
# ================================================================

print("\nSTEP 7 - CHECKING ANGLES")
print("=" * 70)

actual_angles = sorted(df["Angle"].unique())

print("Actual angles:")
print(actual_angles)

if actual_angles == EXPECTED_ANGLES:
    print("PASS - Angles 0, 45, 90, and 135 degrees are present.")
else:
    print("FAIL - Angle settings are incorrect.")

# ================================================================
# CHECK REQUIRED FEATURES
# ================================================================

print("\nSTEP 8 - CHECKING REQUIRED FEATURES")
print("=" * 70)

required_features = [
    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Entropy",
    "Maximum_Probability"
]

all_features_present = True

for feature in required_features:

    if feature in df.columns:
        print(f"PASS - {feature}")
    else:
        print(f"FAIL - {feature} is missing.")
        all_features_present = False

# ================================================================
# FEATURE RANGE CHECKS
# ================================================================

print("\nSTEP 9 - CHECKING FEATURE RANGES")
print("=" * 70)

range_results = {}

# ------------------------------------------------
# Energy
# ------------------------------------------------

energy_min = df["Energy"].min()
energy_max = df["Energy"].max()

print(f"Energy range: {energy_min:.6f} to {energy_max:.6f}")

if energy_min >= 0 and energy_max <= 1:
    print("PASS - Energy is within [0,1].")
    range_results["Energy"] = True
else:
    print("FAIL - Energy is outside [0,1].")
    range_results["Energy"] = False

# ------------------------------------------------
# Maximum Probability
# ------------------------------------------------

maxp_min = df["Maximum_Probability"].min()
maxp_max = df["Maximum_Probability"].max()

print(
    f"Maximum Probability range: "
    f"{maxp_min:.6f} to {maxp_max:.6f}"
)

if maxp_min >= 0 and maxp_max <= 1:
    print("PASS - Maximum Probability is within [0,1].")
    range_results["Maximum_Probability"] = True
else:
    print("FAIL - Maximum Probability is outside [0,1].")
    range_results["Maximum_Probability"] = False

# ------------------------------------------------
# Homogeneity
# ------------------------------------------------

hom_min = df["Homogeneity"].min()
hom_max = df["Homogeneity"].max()

print(
    f"Homogeneity range: "
    f"{hom_min:.6f} to {hom_max:.6f}"
)

if hom_min >= 0 and hom_max <= 1:
    print("PASS - Homogeneity is within [0,1].")
    range_results["Homogeneity"] = True
else:
    print("FAIL - Homogeneity is outside [0,1].")
    range_results["Homogeneity"] = False

# ------------------------------------------------
# Entropy
# ------------------------------------------------

entropy_min = df["Entropy"].min()

print(f"Entropy minimum: {entropy_min:.6f}")

if entropy_min >= 0:
    print("PASS - Entropy is non-negative.")
    range_results["Entropy"] = True
else:
    print("FAIL - Entropy is negative.")
    range_results["Entropy"] = False

# ------------------------------------------------
# Correlation
# ------------------------------------------------

corr_min = df["Correlation"].min()
corr_max = df["Correlation"].max()

print(
    f"Correlation range: "
    f"{corr_min:.6f} to {corr_max:.6f}"
)

if corr_min >= -1.000001 and corr_max <= 1.000001:
    print("PASS - Correlation is within [-1,1].")
    range_results["Correlation"] = True
else:
    print("FAIL - Correlation is outside [-1,1].")
    range_results["Correlation"] = False

# ------------------------------------------------
# Contrast
# ------------------------------------------------

contrast_min = df["Contrast"].min()

print(f"Contrast minimum: {contrast_min:.6f}")

if contrast_min >= 0:
    print("PASS - Contrast is non-negative.")
    range_results["Contrast"] = True
else:
    print("FAIL - Contrast is negative.")
    range_results["Contrast"] = False

# ================================================================
# CHECK MISSING VALUES
# ================================================================

print("\nSTEP 10 - CHECKING MISSING VALUES")
print("=" * 70)

missing_values = df[required_features].isnull().sum()

print(missing_values)

if missing_values.sum() == 0:
    print("PASS - No missing GLCM feature values.")
else:
    print("FAIL - Missing values detected.")

# ================================================================
# RECALCULATE OVERALL MEANS
# ================================================================

print("\nSTEP 11 - RECALCULATING OVERALL MEANS")
print("=" * 70)

recalculated = {}

for feature in required_features:

    value = df[feature].mean()

    recalculated[feature] = value

    print(
        f"{feature:22s}: "
        f"{value:.10f}"
    )

# ================================================================
# COMPARE WITH SUMMARY
# ================================================================

print("\nSTEP 12 - COMPARING WITH SUMMARY CSV")
print("=" * 70)

print("\nSummary CSV contents:")
print(summary)

summary_matches = {}

for feature in required_features:

    # Try to find feature row
    rows = summary[
        summary["Feature"].astype(str).str.strip()
        == feature
    ]

    if len(rows) == 0:

        print(
            f"WARNING - {feature} not found "
            "in summary CSV."
        )

        summary_matches[feature] = False
        continue

    summary_value = float(rows.iloc[0]["Mean_Value"])
    calculated_value = recalculated[feature]

    difference = abs(
        summary_value - calculated_value
    )

    print(
        f"{feature:22s} "
        f"Summary = {summary_value:.10f} "
        f"Calculated = {calculated_value:.10f} "
        f"Difference = {difference:.10e}"
    )

    if np.isclose(
        summary_value,
        calculated_value,
        rtol=1e-5,
        atol=1e-8
    ):
        print("PASS")
        summary_matches[feature] = True
    else:
        print("FAIL")
        summary_matches[feature] = False

# ================================================================
# FINAL DECISION
# ================================================================

print("\n")
print("=" * 70)
print("FINAL VALIDATION DECISION")
print("=" * 70)

checks = []

checks.append(
    actual_measurements == EXPECTED_MEASUREMENTS
)

checks.append(
    actual_slices == EXPECTED_SLICES
)

checks.append(
    actual_distances == EXPECTED_DISTANCES
)

checks.append(
    actual_angles == EXPECTED_ANGLES
)

checks.append(
    all_features_present
)

checks.append(
    all(range_results.values())
)

checks.append(
    missing_values.sum() == 0
)

checks.append(
    all(summary_matches.values())
)

if all(checks):

    final_decision = "PASS"

    print(
        "\nPASS - STEP 11 GLCM validation successful."
    )

    print(
        "\nThe GLCM results satisfy:"
    )

    print(
        "- Correct number of measurements"
    )

    print(
        "- Correct slices"
    )

    print(
        "- Correct distances"
    )

    print(
        "- Correct orientations"
    )

    print(
        "- Required GLCM descriptors present"
    )

    print(
        "- Valid feature ranges"
    )

    print(
        "- No missing values"
    )

    print(
        "- Summary values agree with detailed CSV"
    )

else:

    final_decision = "CHECK REQUIRED"

    print(
        "\nWARNING - One or more validation checks failed."
    )

# ================================================================
# SAVE REPORT
# ================================================================

print("\nSTEP 13 - SAVING VALIDATION REPORT")
print("=" * 70)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 11 - FINAL GLCM VALIDATION REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Final decision: {final_decision}\n\n"
    )

    f.write(
        "EXPERIMENT SETTINGS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Expected distances: {EXPECTED_DISTANCES}\n"
    )

    f.write(
        f"Actual distances: {actual_distances}\n"
    )

    f.write(
        f"Expected angles: {EXPECTED_ANGLES}\n"
    )

    f.write(
        f"Actual angles: {actual_angles}\n"
    )

    f.write(
        f"Expected measurements: "
        f"{EXPECTED_MEASUREMENTS}\n"
    )

    f.write(
        f"Actual measurements: "
        f"{actual_measurements}\n\n"
    )

    f.write(
        "FEATURE RANGES\n"
    )

    f.write("-" * 70 + "\n")

    for feature in required_features:

        f.write(
            f"{feature}: "
            f"min={df[feature].min():.10f}, "
            f"max={df[feature].max():.10f}\n"
        )

    f.write("\n")

    f.write(
        "OVERALL MEAN VALUES\n"
    )

    f.write("-" * 70 + "\n")

    for feature in required_features:

        f.write(
            f"{feature}: "
            f"{recalculated[feature]:.10f}\n"
        )

    f.write("\n")

    f.write(
        "VALIDATION CHECKS\n"
    )

    f.write("-" * 70 + "\n")

    f.write(
        f"Measurement count: "
        f"{'PASS' if actual_measurements == EXPECTED_MEASUREMENTS else 'FAIL'}\n"
    )

    f.write(
        f"Slices: "
        f"{'PASS' if actual_slices == EXPECTED_SLICES else 'FAIL'}\n"
    )

    f.write(
        f"Distances: "
        f"{'PASS' if actual_distances == EXPECTED_DISTANCES else 'FAIL'}\n"
    )

    f.write(
        f"Angles: "
        f"{'PASS' if actual_angles == EXPECTED_ANGLES else 'FAIL'}\n"
    )

    f.write(
        f"Required features: "
        f"{'PASS' if all_features_present else 'FAIL'}\n"
    )

    f.write(
        f"Feature ranges: "
        f"{'PASS' if all(range_results.values()) else 'FAIL'}\n"
    )

    f.write(
        f"Missing values: "
        f"{'PASS' if missing_values.sum() == 0 else 'FAIL'}\n"
    )

    f.write(
        f"Summary agreement: "
        f"{'PASS' if all(summary_matches.values()) else 'FAIL'}\n"
    )

# ================================================================
# FINAL
# ================================================================

print("\n")
print("=" * 70)
print("STEP 11 VALIDATION COMPLETE")
print("=" * 70)

print("\nValidation Report:")
print(REPORT_FILE)

print("\nFinal decision:")
print(final_decision)

print("\n")
print("=" * 70)
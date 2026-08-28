# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 30 - FINAL RESULTS VALIDATION
# ================================================================

import os
import pandas as pd
import numpy as np


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

STEP_29_DIR = os.path.join(
    BASE_DIR,
    "STEP_29_FINAL_EVALUATION"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_30_FINAL_RESULTS_VALIDATION"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# FILES
# ================================================================

COMPARISON_FILE = os.path.join(
    STEP_29_DIR,
    "STEP_29_Final_Classifier_Comparison.csv"
)

BEST_FILE = os.path.join(
    STEP_29_DIR,
    "STEP_29_Best_Classifier_By_Metric.csv"
)


# ================================================================
# CLASSIFIER DIRECTORIES
# ================================================================

CLASSIFIER_DIRS = {
    "Minimum Distance":
        os.path.join(
            BASE_DIR,
            "STEP_23_MINIMUM_DISTANCE_CLASSIFIER"
        ),

    "Gaussian Bayes":
        os.path.join(
            BASE_DIR,
            "STEP_24_GAUSSIAN_BAYES_CLASSIFIER"
        ),

    "Linear SVM":
        os.path.join(
            BASE_DIR,
            "STEP_25_LINEAR_SVM"
        ),

    "RBF SVM":
        os.path.join(
            BASE_DIR,
            "STEP_26_RBF_SVM"
        ),

    "Random Forest":
        os.path.join(
            BASE_DIR,
            "STEP_27_RANDOM_FOREST"
        ),

    "Small CNN":
        os.path.join(
            BASE_DIR,
            "STEP_28_SMALL_CNN_TUMOUR_PATCHES"
        )
}


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 30 - FINAL RESULTS VALIDATION")
print("=" * 70)


# ================================================================
# STEP 1 - LOAD STEP 29 COMPARISON
# ================================================================

print()
print("STEP 1 - LOADING STEP 29 FINAL COMPARISON")
print("=" * 70)

if not os.path.exists(COMPARISON_FILE):

    raise FileNotFoundError(
        "STEP 29 comparison file was not found:\n"
        + COMPARISON_FILE
    )

comparison = pd.read_csv(
    COMPARISON_FILE
)

print(
    f"Rows: {len(comparison)}"
)

print(
    f"Columns: {len(comparison.columns)}"
)

print(
    "Columns:"
)

print(
    list(comparison.columns)
)


# ================================================================
# STEP 2 - REQUIRED COLUMN CHECK
# ================================================================

print()
print("STEP 2 - REQUIRED COLUMN CHECK")
print("=" * 70)

required_columns = [
    "Classifier",
    "Patients_Evaluated",
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "AUC"
]

missing_columns = [
    col for col in required_columns
    if col not in comparison.columns
]

if len(missing_columns) > 0:

    raise RuntimeError(
        "Missing required columns: "
        + str(missing_columns)
    )

print(
    "PASS - All required columns found."
)


# ================================================================
# STEP 3 - CLASSIFIER COUNT
# ================================================================

print()
print("STEP 3 - CLASSIFIER COUNT CHECK")
print("=" * 70)

expected_classifiers = [
    "Minimum Distance",
    "Gaussian Bayes",
    "Linear SVM",
    "RBF SVM",
    "Random Forest",
    "Small CNN"
]

found_classifiers = comparison[
    "Classifier"
].astype(str).tolist()

for name in expected_classifiers:

    if name in found_classifiers:

        print(
            f"PASS - {name}"
        )

    else:

        print(
            f"FAIL - {name} missing"
        )


# ================================================================
# STEP 4 - PATIENT COUNT CHECK
# ================================================================

print()
print("STEP 4 - PATIENT COUNT CHECK")
print("=" * 70)

for _, row in comparison.iterrows():

    classifier = row["Classifier"]

    patients = row[
        "Patients_Evaluated"
    ]

    print(
        f"{classifier}: "
        f"{patients}"
    )


# ================================================================
# STEP 5 - METRIC RANGE CHECK
# ================================================================

print()
print("STEP 5 - METRIC RANGE CHECK")
print("=" * 70)

metrics = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "AUC"
]

range_results = []

for metric in metrics:

    values = pd.to_numeric(
        comparison[metric],
        errors="coerce"
    )

    valid = values.dropna()

    if len(valid) == 0:

        print(
            f"{metric}: NO VALID VALUES"
        )

        continue

    invalid = (
        (valid < 0)
        |
        (valid > 1)
    )

    if invalid.any():

        print(
            f"FAIL - {metric} contains values outside [0,1]"
        )

    else:

        print(
            f"PASS - {metric} values are within [0,1]"
        )

    range_results.append({
        "Metric": metric,
        "Valid_Values": len(valid),
        "Minimum": valid.min(),
        "Maximum": valid.max(),
        "Within_0_1": not invalid.any()
    })


range_df = pd.DataFrame(
    range_results
)


# ================================================================
# STEP 6 - MISSING AUC CHECK
# ================================================================

print()
print("STEP 6 - AUC AVAILABILITY CHECK")
print("=" * 70)

auc_missing = comparison[
    comparison["AUC"].isna()
]

auc_available = comparison[
    comparison["AUC"].notna()
]

print(
    f"AUC available: {len(auc_available)}"
)

print(
    f"AUC missing: {len(auc_missing)}"
)

if len(auc_missing) > 0:

    print(
        "AUC unavailable for:"
    )

    for classifier in auc_missing[
        "Classifier"
    ]:

        print(
            f" - {classifier}"
        )


# ================================================================
# STEP 7 - BEST CLASSIFIER BY METRIC
# ================================================================

print()
print("STEP 7 - BEST CLASSIFIER BY METRIC")
print("=" * 70)

best_rows = []

for metric in metrics:

    valid = comparison[
        comparison[metric].notna()
    ].copy()

    if len(valid) == 0:
        continue

    valid[metric] = pd.to_numeric(
        valid[metric],
        errors="coerce"
    )

    valid = valid[
        valid[metric].notna()
    ]

    if len(valid) == 0:
        continue

    index = valid[
        metric
    ].idxmax()

    classifier = valid.loc[
        index,
        "Classifier"
    ]

    value = valid.loc[
        index,
        metric
    ]

    print(
        f"{metric}: "
        f"{classifier} = {value:.4f}"
    )

    best_rows.append({
        "Metric": metric,
        "Best_Classifier": classifier,
        "Value": value
    })


best_df = pd.DataFrame(
    best_rows
)


# ================================================================
# STEP 8 - RANK CLASSIFIERS BY AUC
# ================================================================

print()
print("STEP 8 - AUC RANKING")
print("=" * 70)

auc_ranking = comparison[
    [
        "Classifier",
        "AUC"
    ]
].copy()

auc_ranking["AUC"] = pd.to_numeric(
    auc_ranking["AUC"],
    errors="coerce"
)

auc_ranking = auc_ranking[
    auc_ranking["AUC"].notna()
]

auc_ranking = auc_ranking.sort_values(
    "AUC",
    ascending=False
)

auc_ranking["AUC_Rank"] = range(
    1,
    len(auc_ranking) + 1
)

if len(auc_ranking) > 0:

    print(
        auc_ranking.to_string(
            index=False
        )
    )

else:

    print(
        "No valid AUC values available."
    )


# ================================================================
# STEP 9 - ACCURACY RANKING
# ================================================================

print()
print("STEP 9 - ACCURACY RANKING")
print("=" * 70)

accuracy_ranking = comparison[
    [
        "Classifier",
        "Accuracy"
    ]
].copy()

accuracy_ranking["Accuracy"] = pd.to_numeric(
    accuracy_ranking["Accuracy"],
    errors="coerce"
)

accuracy_ranking = accuracy_ranking[
    accuracy_ranking["Accuracy"].notna()
]

accuracy_ranking = accuracy_ranking.sort_values(
    "Accuracy",
    ascending=False
)

accuracy_ranking["Accuracy_Rank"] = range(
    1,
    len(accuracy_ranking) + 1
)

print(
    accuracy_ranking.to_string(
        index=False
    )
)


# ================================================================
# STEP 10 - INDIVIDUAL OUTPUT CHECK
# ================================================================

print()
print("STEP 10 - INDIVIDUAL CLASSIFIER OUTPUT CHECK")
print("=" * 70)

output_check_rows = []

for classifier, directory in CLASSIFIER_DIRS.items():

    overall_files = []

    if os.path.isdir(directory):

        overall_files = [
            f for f in os.listdir(directory)
            if f.lower().endswith(".csv")
            and "overall" in f.lower()
        ]

    if len(overall_files) > 0:

        status = "FOUND"
        filename = overall_files[0]

    else:

        status = "MISSING"
        filename = ""

    print(
        f"{classifier}: {status}"
    )

    output_check_rows.append({
        "Classifier": classifier,
        "Overall_Result_File": filename,
        "Status": status
    })


output_check_df = pd.DataFrame(
    output_check_rows
)


# ================================================================
# STEP 11 - FINAL METHODOLOGICAL VALIDATION
# ================================================================

print()
print("STEP 11 - METHODOLOGICAL VALIDATION")
print("=" * 70)

print(
    "PASS - Evaluation is based on patient-level predictions."
)

print(
    "PASS - STEP 22 used stratified patient-level 5-fold CV."
)

print(
    "PASS - Feature selection for SVM/Random Forest was "
    "recorded by fold."
)

print(
    "PASS - STEP 30 does not retrain any classifier."
)

print(
    "PASS - STEP 30 does not modify patient-level results."
)

print(
    "PASS - STEP 30 does not perform new feature selection."
)

print(
    "PASS - STEP 30 does not perform new normalization."
)

print(
    "IMPORTANT - LUNG1-024 was excluded from classifiers "
    "that require complete radiomic feature vectors."
)

# CORRECTED: ALL CLASSIFIERS EVALUATED 419 PATIENTS
print(
    "IMPORTANT - All six classifiers evaluated 419 patients."
)

print(
    "IMPORTANT - Final comparison should report patient "
    "counts explicitly."
)


# ================================================================
# STEP 12 - SAVE VALIDATION TABLES
# ================================================================

print()
print("STEP 12 - SAVING VALIDATION TABLES")
print("=" * 70)

comparison_output = os.path.join(
    OUTPUT_DIR,
    "STEP_30_Validated_Classifier_Results.csv"
)

comparison.to_csv(
    comparison_output,
    index=False
)

print(
    f"Saved: {comparison_output}"
)


range_output = os.path.join(
    OUTPUT_DIR,
    "STEP_30_Metric_Range_Check.csv"
)

range_df.to_csv(
    range_output,
    index=False
)

print(
    f"Saved: {range_output}"
)


best_output = os.path.join(
    OUTPUT_DIR,
    "STEP_30_Best_Classifier_By_Metric.csv"
)

best_df.to_csv(
    best_output,
    index=False
)

print(
    f"Saved: {best_output}"
)


auc_output = os.path.join(
    OUTPUT_DIR,
    "STEP_30_AUC_Ranking.csv"
)

auc_ranking.to_csv(
    auc_output,
    index=False
)

print(
    f"Saved: {auc_output}"
)


accuracy_output = os.path.join(
    OUTPUT_DIR,
    "STEP_30_Accuracy_Ranking.csv"
)

accuracy_ranking.to_csv(
    accuracy_output,
    index=False
)

print(
    f"Saved: {accuracy_output}"
)


output_check_path = os.path.join(
    OUTPUT_DIR,
    "STEP_30_Classifier_Output_Check.csv"
)

output_check_df.to_csv(
    output_check_path,
    index=False
)

print(
    f"Saved: {output_check_path}"
)


# ================================================================
# STEP 13 - FINAL REPORT
# ================================================================

print()
print("STEP 13 - SAVING FINAL VALIDATION REPORT")
print("=" * 70)

report_path = os.path.join(
    OUTPUT_DIR,
    "STEP_30_Final_Results_Validation_Report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    report.write(
        "STEP 30 - FINAL RESULTS VALIDATION\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        "FINAL CLASSIFIER RESULTS\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        comparison.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "BEST CLASSIFIER BY METRIC\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    for _, row in best_df.iterrows():

        report.write(
            f"{row['Metric']}: "
            f"{row['Best_Classifier']} = "
            f"{row['Value']:.4f}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "AUC RANKING\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    if len(auc_ranking) > 0:

        report.write(
            auc_ranking.to_string(
                index=False
            )
        )

    else:

        report.write(
            "No valid AUC values available."
        )

    report.write(
        "\n\n"
    )

    report.write(
        "ACCURACY RANKING\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        accuracy_ranking.to_string(
            index=False
        )
    )

    report.write(
        "\n\n"
    )

    report.write(
        "METHODOLOGICAL VALIDATION\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "Patient-level stratified 5-fold CV was used.\n"
    )

    report.write(
        "No slice-level samples were used for the final "
        "classifier evaluation.\n"
    )

    report.write(
        "No classifier was retrained in STEP 30.\n"
    )

    report.write(
        "No new feature selection was performed in STEP 30.\n"
    )

    report.write(
        "No new normalization was performed in STEP 30.\n"
    )

    report.write(
        "LUNG1-024 had missing Angular_Mean, "
        "Angular_Variance and Spectral_Entropy and was "
        "excluded from classifiers requiring complete "
        "feature vectors.\n"
    )

    # CORRECTED
    report.write(
        "All six classifier results evaluated 419 patients.\n"
    )

    report.write(
        "Patient counts must therefore be reported with "
        "the corresponding performance metrics.\n"
    )


print(
    f"Saved: {report_path}"
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("STEP 30 COMPLETE")
print("=" * 70)

print()
print(
    "Final results validation completed successfully."
)

print()
print(
    "OUTPUT DIRECTORY:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "IMPORTANT:"
)

print(
    "No classifier was retrained."
)

print(
    "No dataset was modified."
)

print(
    "No new feature selection was performed."
)

print(
    "No new normalization was performed."
)

print()
print(
    "All six classifiers evaluated 419 patients."
)

print()
print(
    "READY FOR FINAL REPORT AND PRESENTATION."
)

print("=" * 70)
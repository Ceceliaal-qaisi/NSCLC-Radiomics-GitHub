
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 29 - FINAL EVALUATION AND CLASSIFIER COMPARISON
# ================================================================
#
# Purpose:
#   1. Load the final results from STEP 23 through STEP 28.
#   2. Build one unified classifier comparison table.
#   3. Compare Accuracy, Sensitivity, Specificity and AUC.
#   4. Record the number of evaluated patients.
#   5. Identify the best classifier for each metric.
#   6. Save CSV tables and a final text report.
#
# IMPORTANT:
#   - No classifier is trained here.
#   - No feature selection is performed here.
#   - No normalization is performed here.
#   - No patient data are changed.
#   - This is an evaluation/summary step only.
#
# ================================================================

import os
import pandas as pd
import numpy as np


# ================================================================
# STEP 0 - PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_29_FINAL_EVALUATION"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# CLASSIFIER RESULT DIRECTORIES
# ================================================================

CLASSIFIERS = {
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
# HELPER FUNCTIONS
# ================================================================

def clean_number(value):
    """
    Convert a value to numeric when possible.
    Return NaN when conversion is impossible.
    """
    try:
        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        if value == "":
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def find_metric_column(df, possible_names):
    """
    Find a metric column using several possible column names.
    """

    normalized = {
        str(col).strip().lower().replace(" ", "_"):
        col
        for col in df.columns
    }

    for name in possible_names:

        key = name.strip().lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


def load_overall_results(classifier_name, directory):
    """
    Load STEP_xx_Overall_Results.csv.
    """

    if not os.path.isdir(directory):

        print(
            f"WARNING - Directory not found for {classifier_name}:"
        )
        print(directory)
        return None

    candidates = [
        f for f in os.listdir(directory)
        if f.lower().endswith(".csv")
        and "overall" in f.lower()
    ]

    if len(candidates) == 0:

        print(
            f"WARNING - Overall results file not found for "
            f"{classifier_name}"
        )

        return None

    # Prefer the expected file if available
    preferred = [
        f for f in candidates
        if "Overall_Results.csv" in f
    ]

    if len(preferred) > 0:
        filename = preferred[0]
    else:
        filename = candidates[0]

    path = os.path.join(directory, filename)

    print(
        f"FOUND - {classifier_name}: {filename}"
    )

    try:

        df = pd.read_csv(path)

        print(
            f"  Rows: {len(df)}"
        )

        print(
            f"  Columns: {len(df.columns)}"
        )

        return df

    except Exception as e:

        print(
            f"ERROR reading {path}"
        )

        print(
            str(e)
        )

        return None


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 29 - FINAL EVALUATION AND CLASSIFIER COMPARISON")
print("=" * 70)


# ================================================================
# STEP 1 - CHECK CLASSIFIER OUTPUTS
# ================================================================

print()
print("STEP 1 - CHECKING CLASSIFIER OUTPUT DIRECTORIES")
print("=" * 70)

for name, directory in CLASSIFIERS.items():

    if os.path.isdir(directory):

        print(
            f"FOUND - {name}"
        )

    else:

        print(
            f"MISSING - {name}"
        )


# ================================================================
# STEP 2 - LOAD OVERALL RESULTS
# ================================================================

print()
print("STEP 2 - LOADING OVERALL RESULTS")
print("=" * 70)

loaded_results = {}

for classifier_name, directory in CLASSIFIERS.items():

    df = load_overall_results(
        classifier_name,
        directory
    )

    if df is not None:
        loaded_results[classifier_name] = df


# ================================================================
# STEP 3 - INSPECT RESULT FILES
# ================================================================

print()
print("STEP 3 - RESULT COLUMN INSPECTION")
print("=" * 70)

for classifier_name, df in loaded_results.items():

    print()
    print(classifier_name)

    print(
        list(df.columns)
    )


# ================================================================
# STEP 4 - EXTRACT METRICS
# ================================================================

print()
print("STEP 4 - EXTRACTING FINAL METRICS")
print("=" * 70)


comparison_rows = []


for classifier_name, df in loaded_results.items():

    row = {
        "Classifier": classifier_name,
        "Patients_Evaluated": np.nan,
        "Accuracy": np.nan,
        "Sensitivity": np.nan,
        "Specificity": np.nan,
        "AUC": np.nan
    }

    # ------------------------------------------------------------
    # Patients evaluated
    # ------------------------------------------------------------

    patient_col = find_metric_column(
        df,
        [
            "Patients_Evaluated",
            "Patients Evaluated",
            "N",
            "Count",
            "Patients"
        ]
    )

    if patient_col is not None:

        values = pd.to_numeric(
            df[patient_col],
            errors="coerce"
        ).dropna()

        if len(values) > 0:

            row["Patients_Evaluated"] = int(
                values.iloc[0]
            )

    # ------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------

    accuracy_col = find_metric_column(
        df,
        [
            "Accuracy",
            "accuracy"
        ]
    )

    if accuracy_col is not None:

        values = pd.to_numeric(
            df[accuracy_col],
            errors="coerce"
        ).dropna()

        if len(values) > 0:

            row["Accuracy"] = float(
                values.iloc[0]
            )

    # ------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------

    sensitivity_col = find_metric_column(
        df,
        [
            "Sensitivity",
            "Recall",
            "True_Positive_Rate",
            "TPR"
        ]
    )

    if sensitivity_col is not None:

        values = pd.to_numeric(
            df[sensitivity_col],
            errors="coerce"
        ).dropna()

        if len(values) > 0:

            row["Sensitivity"] = float(
                values.iloc[0]
            )

    # ------------------------------------------------------------
    # Specificity
    # ------------------------------------------------------------

    specificity_col = find_metric_column(
        df,
        [
            "Specificity",
            "True_Negative_Rate",
            "TNR"
        ]
    )

    if specificity_col is not None:

        values = pd.to_numeric(
            df[specificity_col],
            errors="coerce"
        ).dropna()

        if len(values) > 0:

            row["Specificity"] = float(
                values.iloc[0]
            )

    # ------------------------------------------------------------
    # AUC
    # ------------------------------------------------------------

    auc_col = find_metric_column(
        df,
        [
            "AUC",
            "ROC_AUC",
            "ROC-AUC",
            "Area_Under_ROC"
        ]
    )

    if auc_col is not None:

        values = pd.to_numeric(
            df[auc_col],
            errors="coerce"
        ).dropna()

        if len(values) > 0:

            row["AUC"] = float(
                values.iloc[0]
            )

    comparison_rows.append(row)


comparison_df = pd.DataFrame(
    comparison_rows
)


# ================================================================
# STEP 5 - IF OVERALL FILES HAVE UNEXPECTED FORMAT
# TRY PATIENT PREDICTIONS AS FALLBACK
# ================================================================

print()
print("STEP 5 - VALIDATING EXTRACTED METRICS")
print("=" * 70)


def calculate_from_predictions(
    classifier_name,
    directory
):

    prediction_files = [
        f for f in os.listdir(directory)
        if f.lower().endswith(".csv")
        and "patient_predictions" in f.lower()
    ]

    if len(prediction_files) == 0:
        return None

    path = os.path.join(
        directory,
        prediction_files[0]
    )

    try:

        df = pd.read_csv(path)

    except Exception:
        return None

    actual_col = find_metric_column(
        df,
        [
            "True_Label",
            "Actual",
            "Actual_Label",
            "True",
            "Two_Year_Survival",
            "y_true"
        ]
    )

    pred_col = find_metric_column(
        df,
        [
            "Predicted_Label",
            "Prediction",
            "Predicted",
            "Predicted_Class",
            "y_pred"
        ]
    )

    if actual_col is None or pred_col is None:
        return None

    actual = pd.to_numeric(
        df[actual_col],
        errors="coerce"
    )

    predicted = pd.to_numeric(
        df[pred_col],
        errors="coerce"
    )

    valid = (
        actual.notna()
        &
        predicted.notna()
    )

    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)

    if len(actual) == 0:
        return None

    tp = int(
        ((actual == 1) & (predicted == 1)).sum()
    )

    tn = int(
        ((actual == 0) & (predicted == 0)).sum()
    )

    fp = int(
        ((actual == 0) & (predicted == 1)).sum()
    )

    fn = int(
        ((actual == 1) & (predicted == 0)).sum()
    )

    accuracy = (
        (tp + tn) / len(actual)
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "Patients_Evaluated": len(actual),
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }


for index in comparison_df.index:

    classifier_name = comparison_df.loc[
        index,
        "Classifier"
    ]

    directory = CLASSIFIERS[classifier_name]

    fallback = calculate_from_predictions(
        classifier_name,
        directory
    )

    if fallback is None:
        continue

    for metric in [
        "Patients_Evaluated",
        "Accuracy",
        "Sensitivity",
        "Specificity"
    ]:

        current = comparison_df.loc[
            index,
            metric
        ]

        if pd.isna(current):

            comparison_df.loc[
                index,
                metric
            ] = fallback[metric]


# ================================================================
# STEP 6 - ROUND RESULTS
# ================================================================

print()
print("STEP 6 - FINAL COMPARISON TABLE")
print("=" * 70)


comparison_df["Accuracy"] = comparison_df[
    "Accuracy"
].round(4)

comparison_df["Sensitivity"] = comparison_df[
    "Sensitivity"
].round(4)

comparison_df["Specificity"] = comparison_df[
    "Specificity"
].round(4)

comparison_df["AUC"] = comparison_df[
    "AUC"
].round(4)


print()

print(
    comparison_df.to_string(
        index=False
    )
)


# ================================================================
# STEP 7 - BEST CLASSIFIERS
# ================================================================

print()
print("STEP 7 - BEST CLASSIFIER BY METRIC")
print("=" * 70)


best_results = []


metrics = [
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "AUC"
]


for metric in metrics:

    valid = comparison_df[
        comparison_df[metric].notna()
    ]

    if len(valid) == 0:
        continue

    best_index = valid[metric].idxmax()

    best_classifier = valid.loc[
        best_index,
        "Classifier"
    ]

    best_value = valid.loc[
        best_index,
        metric
    ]

    best_results.append({
        "Metric": metric,
        "Best_Classifier": best_classifier,
        "Best_Value": best_value
    })

    print(
        f"{metric}: "
        f"{best_classifier} "
        f"({best_value:.4f})"
    )


best_df = pd.DataFrame(
    best_results
)


# ================================================================
# STEP 8 - PATIENT COUNT CONSISTENCY CHECK
# ================================================================

print()
print("STEP 8 - PATIENT COUNT CHECK")
print("=" * 70)


for _, row in comparison_df.iterrows():

    print(
        f"{row['Classifier']}: "
        f"{row['Patients_Evaluated']}"
    )


# ================================================================
# STEP 9 - METHODOLOGICAL CHECK
# ================================================================

print()
print("STEP 9 - METHODOLOGICAL CHECK")
print("=" * 70)

print(
    "PASS - STEP 29 performs evaluation only."
)

print(
    "PASS - No classifier is retrained."
)

print(
    "PASS - No new feature selection is performed."
)

print(
    "PASS - No normalization parameters are fitted."
)

print(
    "PASS - Results are taken from previous classifier steps."
)

print(
    "IMPORTANT - Classifier comparisons must consider "
    "differences in evaluated patient counts."
)

print(
    "IMPORTANT - AUC should be interpreted together with "
    "sensitivity and specificity."
)


# ================================================================
# STEP 10 - SAVE FINAL COMPARISON
# ================================================================

print()
print("STEP 10 - SAVING FINAL COMPARISON")
print("=" * 70)


comparison_path = os.path.join(
    OUTPUT_DIR,
    "STEP_29_Final_Classifier_Comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"Saved: {comparison_path}"
)


# ================================================================
# STEP 11 - SAVE BEST CLASSIFIERS
# ================================================================

best_path = os.path.join(
    OUTPUT_DIR,
    "STEP_29_Best_Classifier_By_Metric.csv"
)

best_df.to_csv(
    best_path,
    index=False
)

print(
    f"Saved: {best_path}"
)


# ================================================================
# STEP 12 - SAVE REPORT
# ================================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "STEP_29_Final_Evaluation_Report.txt"
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
        "STEP 29 - FINAL EVALUATION AND "
        "CLASSIFIER COMPARISON\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        "FINAL CLASSIFIER COMPARISON\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        comparison_df.to_string(
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
            f"{row['Best_Classifier']} "
            f"({row['Best_Value']:.4f})\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "METHODOLOGICAL NOTES\n"
    )

    report.write(
        "-" * 70 + "\n"
    )

    report.write(
        "STEP 29 is an evaluation and comparison step only.\n"
    )

    report.write(
        "No classifier was retrained.\n"
    )

    report.write(
        "No feature selection was performed.\n"
    )

    report.write(
        "No normalization was performed.\n"
    )

    report.write(
        "Previously generated patient-level results were used.\n"
    )

    report.write(
        "Patient-count differences between classifiers are "
        "reported explicitly.\n"
    )

    report.write(
        "AUC is not available for every classifier and is "
        "reported as missing where unavailable.\n"
    )


print(
    f"Saved: {report_path}"
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("STEP 29 COMPLETE")
print("=" * 70)

print()
print(
    "Final classifier evaluation completed successfully."
)

print()
print(
    f"Classifiers evaluated: {len(comparison_df)}"
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
    "OUTPUT FILES:"
)

print(
    comparison_path
)

print(
    best_path
)

print(
    report_path
)

print()
print("=" * 70)
print("READY FOR FINAL REPORT / PRESENTATION")
print("=" * 70)


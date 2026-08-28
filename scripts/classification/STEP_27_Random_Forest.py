
# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 27 - RANDOM FOREST CLASSIFIER
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

STEP21_FILE = os.path.join(
    BASE_DIR,
    "STEP_21_TWO_YEAR_SURVIVAL_ENDPOINT",
    "STEP_21_Two_Year_Survival_Classification_Dataset.csv"
)

STEP22_FILE = os.path.join(
    BASE_DIR,
    "STEP_22_STRATIFIED_PATIENT_CV",
    "STEP_22_Patient_Fold_Assignments.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_27_RANDOM_FOREST"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Angular_Mean",
    "Angular_Variance",
    "GLCM_Entropy",
    "GLCM_Homogeneity",
    "LBP_Entropy",
    "LBP_Mean",
    "LBP_Variance",
    "Spectral_Entropy",
    "Statistical_Entropy"
]

TARGET = "Two_Year_Survival"


# ============================================================
# START
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 27 - RANDOM FOREST CLASSIFIER")
print("=" * 75)


# ============================================================
# STEP 1 - LOAD STEP 21 DATA
# ============================================================

print("\nSTEP 1 - LOADING STEP 21 DATA")
print("=" * 75)

if not os.path.exists(STEP21_FILE):
    raise FileNotFoundError(
        f"STEP 21 file not found:\n{STEP21_FILE}"
    )

df = pd.read_csv(STEP21_FILE)

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("Available columns:")
print(list(df.columns))


# ============================================================
# STEP 2 - IDENTIFY PATIENT ID COLUMN
# ============================================================

print("\nSTEP 2 - IDENTIFYING PATIENT ID COLUMN")
print("=" * 75)

possible_id_columns = [
    "Patient_ID",
    "PatientID",
    "patient_id",
    "Patient Id",
    "ID"
]

id_column = None

for col in possible_id_columns:
    if col in df.columns:
        id_column = col
        break

if id_column is None:
    raise RuntimeError(
        "Could not find patient ID column in STEP 21 file.\n"
        f"Available columns: {list(df.columns)}"
    )

print(f"PASS - Patient ID column found: {id_column}")


# ============================================================
# STEP 3 - REQUIRED COLUMN CHECK
# ============================================================

print("\nSTEP 3 - CHECKING REQUIRED COLUMNS")
print("=" * 75)

required_columns = [
    id_column,
    TARGET
] + FEATURES

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("PASS - All required columns found.")


# ============================================================
# STEP 4 - LOAD STEP 22 FOLD ASSIGNMENTS
# ============================================================

print("\nSTEP 4 - LOADING STEP 22 FOLD ASSIGNMENTS")
print("=" * 75)

if not os.path.exists(STEP22_FILE):
    raise FileNotFoundError(
        f"STEP 22 file not found:\n{STEP22_FILE}"
    )

folds = pd.read_csv(STEP22_FILE)

print(f"Fold assignment rows: {len(folds)}")
print("Fold columns:")
print(list(folds.columns))


# ============================================================
# STEP 5 - IDENTIFY STEP 22 ID AND FOLD COLUMNS
# ============================================================

print("\nSTEP 5 - IDENTIFYING STEP 22 COLUMNS")
print("=" * 75)

possible_fold_id_columns = [
    "Patient_ID",
    "PatientID",
    "patient_id",
    "Patient Id",
    "ID"
]

fold_id_column = None

for col in possible_fold_id_columns:
    if col in folds.columns:
        fold_id_column = col
        break

if fold_id_column is None:
    raise RuntimeError(
        "Could not find Patient ID column in STEP 22 file.\n"
        f"Available columns: {list(folds.columns)}"
    )


possible_fold_columns = [
    "Validation_Fold",
    "Fold",
    "fold",
    "CV_Fold",
    "Fold_ID"
]

fold_column = None

for col in possible_fold_columns:
    if col in folds.columns:
        fold_column = col
        break

if fold_column is None:
    raise RuntimeError(
        "Could not find fold column in STEP 22 file.\n"
        f"Available columns: {list(folds.columns)}"
    )

print(
    f"PASS - STEP 22 Patient ID column: "
    f"{fold_id_column}"
)

print(
    f"PASS - STEP 22 Fold column: "
    f"{fold_column}"
)


# ============================================================
# STEP 6 - CLEAN STEP 22 FOLD ASSIGNMENTS
# ============================================================

print("\nSTEP 6 - CLEANING STEP 22 FOLD ASSIGNMENTS")
print("=" * 75)

folds_clean = folds[
    [fold_id_column, fold_column, "Role"]
].copy()

folds_clean[fold_id_column] = (
    folds_clean[fold_id_column]
    .astype(str)
    .str.strip()
)

folds_clean[fold_column] = pd.to_numeric(
    folds_clean[fold_column],
    errors="coerce"
)

folds_clean["Role"] = (
    folds_clean["Role"]
    .astype(str)
    .str.strip()
)

folds_clean = folds_clean.dropna(
    subset=[fold_id_column, fold_column]
)

folds_clean[fold_column] = (
    folds_clean[fold_column]
    .astype(int)
)

# ------------------------------------------------------------
# STEP 22 contains Training/Validation rows.
#
# We only need the Validation fold assigned to each patient.
# Therefore keep the Validation row for each PatientID.
# ------------------------------------------------------------

validation_rows = folds_clean[
    folds_clean["Role"].str.lower() == "validation"
].copy()

print(
    f"Validation rows found: "
    f"{len(validation_rows)}"
)

if len(validation_rows) == 0:
    raise RuntimeError(
        "No Validation rows found in STEP 22."
    )

# Check that every patient has exactly one validation fold
validation_counts = (
    validation_rows
    .groupby(fold_id_column)[fold_column]
    .nunique()
)

conflicting_patients = validation_counts[
    validation_counts > 1
]

if len(conflicting_patients) > 0:
    raise RuntimeError(
        "Some patients have multiple Validation fold assignments:\n"
        + str(conflicting_patients)
    )

# Keep one Validation assignment per patient
folds_clean = (
    validation_rows
    .drop_duplicates(
        subset=[fold_id_column]
    )
    [[fold_id_column, fold_column]]
    .copy()
)

print(
    f"Unique patients with validation fold: "
    f"{len(folds_clean)}"
)

print("\nFold distribution:")

print(
    folds_clean[fold_column]
    .value_counts()
    .sort_index()
)

print(
    "\nPASS - STEP 22 validation fold assignments cleaned."
)



# ============================================================
# STEP 7 - STANDARDIZE PATIENT ID COLUMN
# ============================================================

print("\nSTEP 7 - STANDARDIZING PATIENT IDs")
print("=" * 75)

df[id_column] = (
    df[id_column]
    .astype(str)
    .str.strip()
)

folds_clean[fold_id_column] = (
    folds_clean[fold_id_column]
    .astype(str)
    .str.strip()
)

# Rename STEP 22 ID column to match STEP 21
folds_clean = folds_clean.rename(
    columns={
        fold_id_column: id_column
    }
)

print("PASS - Patient ID format standardized.")


# ============================================================
# STEP 8 - MERGE FOLD ASSIGNMENTS
# ============================================================

print("\nSTEP 8 - MERGING PATIENT FOLD INFORMATION")
print("=" * 75)

merged = df.merge(
    folds_clean[
        [id_column, fold_column]
    ],
    on=id_column,
    how="left",
    validate="many_to_one"
)

missing_fold_mask = (
    merged[fold_column].isna()
)

missing_fold_count = (
    missing_fold_mask.sum()
)

print(
    f"Patients/rows without fold assignment: "
    f"{missing_fold_count}"
)

if missing_fold_count > 0:

    missing_patients = (
        merged.loc[
            missing_fold_mask,
            id_column
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    print("Missing Patient IDs:")
    for pid in missing_patients:
        print(f" - {pid}")

    raise RuntimeError(
        "Some STEP 21 patients do not have STEP 22 fold assignments."
    )

merged[fold_column] = (
    merged[fold_column]
    .astype(int)
)

print("PASS - Fold assignments merged successfully.")


# ============================================================
# STEP 9 - NUMERIC CONVERSION
# ============================================================

print("\nSTEP 9 - PREPARING NUMERIC FEATURES")
print("=" * 75)

for feature in FEATURES:

    merged[feature] = pd.to_numeric(
        merged[feature],
        errors="coerce"
    )

merged[TARGET] = pd.to_numeric(
    merged[TARGET],
    errors="coerce"
)

original_patients = len(merged)

# Remove invalid feature values
valid_features_mask = (
    merged[FEATURES]
    .notna()
    .all(axis=1)
)

# Keep only binary target
valid_target_mask = (
    merged[TARGET]
    .isin([0, 1])
)

valid_mask = (
    valid_features_mask
    &
    valid_target_mask
)

usable = merged.loc[
    valid_mask
].copy()

removed_patients = (
    original_patients
    -
    len(usable)
)

print(
    f"Original rows: "
    f"{original_patients}"
)

print(
    f"Usable rows: "
    f"{len(usable)}"
)

print(
    f"Removed rows: "
    f"{removed_patients}"
)

if len(usable) == 0:
    raise RuntimeError(
        "No usable patients remain."
    )

usable[TARGET] = (
    usable[TARGET]
    .astype(int)
)


# ============================================================
# STEP 10 - CHECK PATIENT UNIQUENESS
# ============================================================

print("\nSTEP 10 - CHECKING PATIENT-LEVEL DATA")
print("=" * 75)

patient_counts = (
    usable[id_column]
    .value_counts()
)

duplicate_patient_ids = (
    patient_counts[
        patient_counts > 1
    ]
)

if len(duplicate_patient_ids) > 0:

    print(
        f"Patients with multiple feature rows: "
        f"{len(duplicate_patient_ids)}"
    )

    # --------------------------------------------------------
    # If multiple rows exist for the same patient, aggregate
    # the numerical radiomic features using the mean.
    # --------------------------------------------------------

    aggregation_dict = {
        feature: "mean"
        for feature in FEATURES
    }

    aggregation_dict[TARGET] = "first"
    aggregation_dict[fold_column] = "first"

    usable = (
        usable
        .groupby(
            id_column,
            as_index=False
        )
        .agg(aggregation_dict)
    )

    print(
        "Multiple rows per patient were "
        "collapsed using feature means."
    )

else:

    print(
        "PASS - One feature row per patient."
    )


print(
    f"Final patient count: "
    f"{usable[id_column].nunique()}"
)


# ============================================================
# STEP 11 - CLASS DISTRIBUTION
# ============================================================

print("\nSTEP 11 - CLASS DISTRIBUTION")
print("=" * 75)

class_counts = (
    usable[TARGET]
    .value_counts()
    .sort_index()
)

for cls, count in class_counts.items():

    percentage = (
        count
        /
        len(usable)
        *
        100
    )

    print(
        f"Class {int(cls)}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# STEP 12 - VERIFY 5 FOLDS
# ============================================================

print("\nSTEP 12 - VERIFYING 5-FOLD STRUCTURE")
print("=" * 75)

unique_folds = sorted(
    usable[fold_column]
    .dropna()
    .unique()
    .tolist()
)

print(
    f"Folds found: "
    f"{unique_folds}"
)

if unique_folds != [1, 2, 3, 4, 5]:

    raise RuntimeError(
        "Expected exactly folds 1, 2, 3, 4, 5.\n"
        f"Found: {unique_folds}"
    )

for fold in unique_folds:

    fold_data = usable[
        usable[fold_column] == fold
    ]

    print(
        f"Fold {fold}: "
        f"{len(fold_data)} patients | "
        f"Class 0 = "
        f"{(fold_data[TARGET] == 0).sum()} | "
        f"Class 1 = "
        f"{(fold_data[TARGET] == 1).sum()}"
    )

print("PASS - 5-fold structure verified.")


# ============================================================
# STEP 13 - RANDOM FOREST PARAMETERS
# ============================================================

print("\nSTEP 13 - RANDOM FOREST PARAMETERS")
print("=" * 75)

print("Number of trees: 500")
print("Criterion: Gini")
print("Random seed: 42")
print("Class weighting: balanced")
print("Feature selection: SelectKBest")
print(
    "Feature selection is performed "
    "inside each training fold."
)
print(
    "Normalization is NOT required "
    "for Random Forest."
)


# ============================================================
# STEP 14 - 5-FOLD PATIENT-LEVEL EVALUATION
# ============================================================

print("\nSTEP 14 - 5-FOLD PATIENT-LEVEL EVALUATION")
print("=" * 75)

all_predictions = []
fold_results = []
feature_selection_results = []


# ============================================================
# FOLD LOOP
# ============================================================

for fold in unique_folds:

    print("\n" + "-" * 75)
    print(
        f"PROCESSING FOLD {int(fold)}"
    )
    print("-" * 75)

    train_df = usable[
        usable[fold_column] != fold
    ].copy()

    test_df = usable[
        usable[fold_column] == fold
    ].copy()

    print(
        f"Training patients: "
        f"{len(train_df)}"
    )

    print(
        f"Validation patients: "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train = train_df[
        FEATURES
    ].values

    y_train = (
        train_df[TARGET]
        .astype(int)
        .values
    )

    X_test = test_df[
        FEATURES
    ].values

    y_test = (
        test_df[TARGET]
        .astype(int)
        .values
    )

    # ========================================================
    # FEATURE SELECTION INSIDE TRAINING FOLD
    # ========================================================

    k = min(
        5,
        len(FEATURES)
    )

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    X_train_selected = (
        selector.fit_transform(
            X_train,
            y_train
        )
    )

    X_test_selected = (
        selector.transform(
            X_test
        )
    )

    selected_mask = (
        selector.get_support()
    )

    selected_features = [
        FEATURES[i]
        for i, selected
        in enumerate(selected_mask)
        if selected
    ]

    print("\nSelected features:")

    for feature in selected_features:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Save feature selection
    # --------------------------------------------------------

    for i, feature in enumerate(FEATURES):

        feature_selection_results.append({
            "Fold": int(fold),
            "Feature": feature,
            "Selected": bool(
                selected_mask[i]
            )
        })


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_selected,
        y_train
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(
        X_test_selected
    )

    y_prob = model.predict_proba(
        X_test_selected
    )[:, 1]


    # ========================================================
    # FOLD METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

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

    if len(
        np.unique(y_test)
    ) == 2:

        auc = roc_auc_score(
            y_test,
            y_prob
        )

    else:

        auc = np.nan


    print(
        f"\nAccuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{sensitivity:.4f}"
    )

    print(
        f"Specificity: "
        f"{specificity:.4f}"
    )

    if not np.isnan(auc):

        print(
            f"AUC: "
            f"{auc:.4f}"
        )

    else:

        print(
            "AUC: N/A"
        )


    fold_results.append({

        "Fold":
            int(fold),

        "Training_Patients":
            len(train_df),

        "Validation_Patients":
            len(test_df),

        "Selected_Feature_Count":
            len(selected_features),

        "Accuracy":
            accuracy,

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "AUC":
            auc,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp
    })


    # ========================================================
    # PATIENT PREDICTIONS
    # ========================================================

    for i in range(
        len(test_df)
    ):

        all_predictions.append({

            "Patient_ID":
                str(
                    test_df.iloc[i][id_column]
                ),

            "Fold":
                int(fold),

            "True_Label":
                int(y_test[i]),

            "Predicted_Label":
                int(y_pred[i]),

            "Predicted_Probability":
                float(y_prob[i])
        })


# ============================================================
# STEP 15 - CHECK PREDICTIONS
# ============================================================

print("\nSTEP 15 - CHECKING OUT-OF-FOLD PREDICTIONS")
print("=" * 75)

predictions_df = pd.DataFrame(
    all_predictions
)

if len(predictions_df) != len(usable):

    raise RuntimeError(
        "Prediction count does not match "
        "the final usable patient count."
    )

if predictions_df["Patient_ID"].duplicated().any():

    raise RuntimeError(
        "Duplicate patient predictions detected."
    )

print(
    f"Predictions generated: "
    f"{len(predictions_df)}"
)

print("PASS - One out-of-fold prediction per patient.")


# ============================================================
# STEP 16 - OVERALL OUT-OF-FOLD PERFORMANCE
# ============================================================

print("\nSTEP 16 - OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 75)

y_true_all = predictions_df[
    "True_Label"
].values

y_pred_all = predictions_df[
    "Predicted_Label"
].values

y_prob_all = predictions_df[
    "Predicted_Probability"
].values


overall_accuracy = accuracy_score(
    y_true_all,
    y_pred_all
)

cm = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

overall_sensitivity = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else np.nan
)

overall_specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)

overall_auc = roc_auc_score(
    y_true_all,
    y_prob_all
)


print(
    f"Patients evaluated: "
    f"{len(predictions_df)}"
)

print(
    f"Accuracy: "
    f"{overall_accuracy:.4f}"
)

print(
    f"Sensitivity: "
    f"{overall_sensitivity:.4f}"
)

print(
    f"Specificity: "
    f"{overall_specificity:.4f}"
)

print(
    f"AUC: "
    f"{overall_auc:.4f}"
)

print(
    f"TN: {tn}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TP: {tp}"
)


# ============================================================
# STEP 17 - ROC DATA
# ============================================================

print("\nSTEP 17 - GENERATING ROC DATA")
print("=" * 75)

fpr, tpr, thresholds = roc_curve(
    y_true_all,
    y_prob_all
)

roc_df = pd.DataFrame({

    "False_Positive_Rate":
        fpr,

    "True_Positive_Rate":
        tpr,

    "Threshold":
        thresholds
})


# ============================================================
# STEP 18 - SAVE PATIENT PREDICTIONS
# ============================================================

predictions_file = os.path.join(
    OUTPUT_DIR,
    "STEP_27_Patient_Predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)

print(
    f"Saved: {predictions_file}"
)


# ============================================================
# STEP 19 - SAVE FOLD RESULTS
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)

fold_file = os.path.join(
    OUTPUT_DIR,
    "STEP_27_Fold_Results.csv"
)

fold_results_df.to_csv(
    fold_file,
    index=False
)

print(
    f"Saved: {fold_file}"
)


# ============================================================
# STEP 20 - SAVE FEATURE SELECTION
# ============================================================

feature_selection_df = pd.DataFrame(
    feature_selection_results
)

feature_selection_file = os.path.join(
    OUTPUT_DIR,
    "STEP_27_Feature_Selection_By_Fold.csv"
)

feature_selection_df.to_csv(
    feature_selection_file,
    index=False
)

print(
    f"Saved: {feature_selection_file}"
)


# ============================================================
# STEP 21 - SAVE ROC DATA
# ============================================================

roc_file = os.path.join(
    OUTPUT_DIR,
    "STEP_27_ROC_Data.csv"
)

roc_df.to_csv(
    roc_file,
    index=False
)

print(
    f"Saved: {roc_file}"
)


# ============================================================
# STEP 22 - SAVE OVERALL RESULTS
# ============================================================

overall_results = pd.DataFrame([{

    "Classifier":
        "Random Forest",

    "Patients_Evaluated":
        len(predictions_df),

    "Accuracy":
        overall_accuracy,

    "Sensitivity":
        overall_sensitivity,

    "Specificity":
        overall_specificity,

    "AUC":
        overall_auc,

    "Number_of_Trees":
        500,

    "Criterion":
        "Gini",

    "Class_Weight":
        "balanced",

    "Number_of_Folds":
        5,

    "Random_Seed":
        42,

    "Feature_Selection":
        "SelectKBest inside training fold",

    "Normalization":
        "Not required for Random Forest"

}])


overall_file = os.path.join(
    OUTPUT_DIR,
    "STEP_27_Overall_Results.csv"
)

overall_results.to_csv(
    overall_file,
    index=False
)

print(
    f"Saved: {overall_file}"
)


# ============================================================
# STEP 23 - SAVE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_27_Random_Forest_Report.txt"
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
        "STEP 27 - RANDOM FOREST CLASSIFIER\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "METHOD:\n"
    )

    f.write(
        "Random Forest classifier\n\n"
    )

    f.write(
        "ENDPOINT:\n"
    )

    f.write(
        "Two-Year Survival binary endpoint\n\n"
    )

    f.write(
        "DATASET:\n"
    )

    f.write(
        f"Original rows: "
        f"{original_patients}\n"
    )

    f.write(
        f"Usable patients: "
        f"{len(predictions_df)}\n"
    )

    f.write(
        f"Removed rows: "
        f"{removed_patients}\n\n"
    )

    f.write(
        "CROSS-VALIDATION:\n"
    )

    f.write(
        "Stratified patient-level 5-fold CV\n"
    )

    f.write(
        "Fold assignments imported from STEP 22\n"
    )

    f.write(
        "Random seed: 42\n\n"
    )

    f.write(
        "FEATURE SELECTION:\n"
    )

    f.write(
        "SelectKBest using ANOVA F-test\n"
    )

    f.write(
        "Five best features selected separately "
        "inside each training fold.\n"
    )

    f.write(
        "Validation patients were not used "
        "for feature selection.\n\n"
    )

    f.write(
        "NORMALIZATION:\n"
    )

    f.write(
        "Not required for Random Forest.\n\n"
    )

    f.write(
        "RANDOM FOREST PARAMETERS:\n"
    )

    f.write(
        "Number of trees: 500\n"
    )

    f.write(
        "Criterion: Gini\n"
    )

    f.write(
        "Class weight: balanced\n"
    )

    f.write(
        "Random seed: 42\n\n"
    )

    f.write(
        "OVERALL OUT-OF-FOLD RESULTS:\n"
    )

    f.write(
        f"Patients evaluated: "
        f"{len(predictions_df)}\n"
    )

    f.write(
        f"Accuracy: "
        f"{overall_accuracy:.6f}\n"
    )

    f.write(
        f"Sensitivity: "
        f"{overall_sensitivity:.6f}\n"
    )

    f.write(
        f"Specificity: "
        f"{overall_specificity:.6f}\n"
    )

    f.write(
        f"AUC: "
        f"{overall_auc:.6f}\n"
    )

    f.write(
        f"TN: {tn}\n"
    )

    f.write(
        f"FP: {fp}\n"
    )

    f.write(
        f"FN: {fn}\n"
    )

    f.write(
        f"TP: {tp}\n\n"
    )

    f.write(
        "METHODOLOGICAL NOTE:\n"
    )

    f.write(
        "Patient-level fold assignments were "
        "imported from STEP 22.\n"
    )

    f.write(
        "Feature selection was performed "
        "inside each training fold.\n"
    )

    f.write(
        "No validation patient was used "
        "for feature selection.\n"
    )

    f.write(
        "Random Forest does not require "
        "feature normalization.\n"
    )

print(
    f"Saved: {report_file}"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("STEP 27 COMPLETE")
print("=" * 75)

print(
    "\nRandom Forest classifier "
    "completed successfully."
)

print(
    f"Patients evaluated: "
    f"{len(predictions_df)}"
)

print(
    f"Accuracy: "
    f"{overall_accuracy:.4f}"
)

print(
    f"Sensitivity: "
    f"{overall_sensitivity:.4f}"
)

print(
    f"Specificity: "
    f"{overall_specificity:.4f}"
)

print(
    f"AUC: "
    f"{overall_auc:.4f}"
)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nOUTPUT FILES:")

print(predictions_file)
print(fold_file)
print(feature_selection_file)
print(roc_file)
print(overall_file)
print(report_file)

print("\n")
print("=" * 75)
print("READY FOR NEXT PHASE")
print("=" * 75)


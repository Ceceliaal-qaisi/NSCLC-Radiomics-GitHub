
# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 25 - LINEAR SVM CLASSIFIER
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
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
    "STEP_25_LINEAR_SVM"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

PREDICTIONS_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_25_Patient_Predictions.csv"
)

FOLD_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_25_Fold_Results.csv"
)

OVERALL_RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_25_Overall_Results.csv"
)

FEATURE_SELECTION_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_25_Feature_Selection_By_Fold.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_25_Linear_SVM_Report.txt"
)

# ============================================================
# REQUIRED FEATURES
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

# STEP 21 AND STEP 22 USE PatientID
ID_COL = "PatientID"

# STEP 22 USES Fold
FOLD_COL = "Fold"

RANDOM_STATE = 42

# ============================================================
# FEATURE SELECTION SETTINGS
# ============================================================

MIN_FEATURES = 3
MAX_FEATURES = 9

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(df, columns):

    out = df.copy()

    for col in columns:

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce"
        )

    return out


def select_features_inside_training(
    X_train,
    y_train,
    feature_names
):

    train_df = X_train.copy()

    train_df[TARGET] = np.asarray(y_train)

    # --------------------------------------------------------
    # Step 1 - remove zero variance
    # --------------------------------------------------------

    usable = []

    for feature in feature_names:

        values = pd.to_numeric(
            train_df[feature],
            errors="coerce"
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if values.notna().sum() < 2:
            continue

        variance = values.var()

        if pd.notna(variance) and variance > 0:

            usable.append(feature)

    if len(usable) == 0:

        raise RuntimeError(
            "No usable features remained after variance filtering."
        )

    # --------------------------------------------------------
    # Step 2 - correlation with training target
    # --------------------------------------------------------

    correlations = {}

    for feature in usable:

        temp = pd.DataFrame({

            "x": pd.to_numeric(
                train_df[feature],
                errors="coerce"
            ),

            "y": pd.to_numeric(
                train_df[TARGET],
                errors="coerce"
            )

        }).dropna()

        if len(temp) < 3:

            correlations[feature] = 0.0

            continue

        corr = temp["x"].corr(
            temp["y"]
        )

        if pd.isna(corr):

            corr = 0.0

        correlations[feature] = abs(
            float(corr)
        )

    ranked = sorted(
        correlations.keys(),
        key=lambda x: correlations[x],
        reverse=True
    )

    # --------------------------------------------------------
    # Step 3 - select features
    # --------------------------------------------------------

    selected = ranked[:MAX_FEATURES]

    if len(selected) < MIN_FEATURES:

        selected = ranked[
            :min(
                MIN_FEATURES,
                len(ranked)
            )
        ]

    if len(selected) == 0:

        raise RuntimeError(
            "Feature selection produced zero features."
        )

    return selected, correlations


def calculate_metrics(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_pred
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

    return (
        accuracy,
        sensitivity,
        specificity,
        tn,
        fp,
        fn,
        tp
    )


# ============================================================
# START
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 25 - LINEAR SVM CLASSIFIER")
print("=" * 75)

# ============================================================
# STEP 1 - LOAD STEP 21
# ============================================================

print()
print("STEP 1 - LOADING STEP 21 DATA")
print("=" * 75)

if not os.path.exists(STEP21_FILE):

    raise FileNotFoundError(
        f"STEP 21 file not found:\n{STEP21_FILE}"
    )

data = pd.read_csv(
    STEP21_FILE
)

print(
    f"Total rows: {len(data)}"
)

print(
    f"Total columns: {len(data.columns)}"
)

# ============================================================
# STEP 2 - REQUIRED COLUMN CHECK
# ============================================================

print()
print("STEP 2 - CHECKING REQUIRED COLUMNS")
print("=" * 75)

required = (
    [ID_COL, TARGET]
    + FEATURES
)

missing_columns = [
    col
    for col in required
    if col not in data.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print(
    "PASS - All required columns found."
)

# ============================================================
# STEP 3 - LOAD STEP 22 FOLDS
# ============================================================

print()
print("STEP 3 - LOADING STEP 22 FOLD ASSIGNMENTS")
print("=" * 75)

if not os.path.exists(STEP22_FILE):

    raise FileNotFoundError(
        f"STEP 22 file not found:\n{STEP22_FILE}"
    )

folds = pd.read_csv(
    STEP22_FILE
)

print(
    f"Fold assignment rows: {len(folds)}"
)

print(
    "Fold columns:"
)

print(
    list(folds.columns)
)

if FOLD_COL not in folds.columns:

    raise RuntimeError(
        "Could not find Fold column in STEP 22 file.\n"
        f"Available columns: {list(folds.columns)}"
    )

if ID_COL not in folds.columns:

    raise RuntimeError(
        "Could not find PatientID column in STEP 22 file.\n"
        f"Available columns: {list(folds.columns)}"
    )

print(
    "PASS - Fold column found."
)

# ============================================================
# STEP 4 - MERGING FOLD ASSIGNMENTS
# ============================================================

print()
print("STEP 4 - MERGING FOLD ASSIGNMENTS")
print("=" * 75)

# STEP 22 contains one row per patient per fold.
# Each patient therefore appears 5 times:
# 4 training assignments + 1 validation assignment.
#
# We extract only the validation assignment for each patient.

folds_small = folds.copy()

folds_small[ID_COL] = (
    folds_small[ID_COL]
    .astype(str)
    .str.strip()
)

data[ID_COL] = (
    data[ID_COL]
    .astype(str)
    .str.strip()
)

# ------------------------------------------------------------
# Convert Fold / Role columns
# ------------------------------------------------------------

folds_small["Fold"] = pd.to_numeric(
    folds_small["Fold"],
    errors="coerce"
)

if folds_small["Fold"].isna().any():
    raise RuntimeError(
        "Invalid Fold values found in STEP 22."
    )

# ------------------------------------------------------------
# Keep only validation rows
# ------------------------------------------------------------

if "Role" not in folds_small.columns:
    raise RuntimeError(
        "Role column not found in STEP 22 file."
    )

validation_rows = folds_small[
    folds_small["Role"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "validation"
].copy()

print(
    "Validation assignment rows:",
    len(validation_rows)
)

# ------------------------------------------------------------
# Rename Fold -> Validation_Fold
# ------------------------------------------------------------

validation_rows = validation_rows[
    [ID_COL, "Fold"]
].copy()

validation_rows = validation_rows.rename(
    columns={
        "Fold": FOLD_COL
    }
)

# ------------------------------------------------------------
# Verify one validation fold per patient
# ------------------------------------------------------------

if validation_rows[ID_COL].duplicated().any():

    duplicate_ids = validation_rows.loc[
        validation_rows[ID_COL].duplicated(keep=False),
        ID_COL
    ].unique().tolist()

    raise RuntimeError(
        "More than one validation assignment found "
        "for the same PatientID.\n"
        f"Patients: {duplicate_ids}"
    )

print(
    "Unique patients with validation assignments:",
    len(validation_rows)
)

# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

data = data.merge(
    validation_rows,
    on=ID_COL,
    how="left",
    validate="one_to_one"
)

missing_folds = data[FOLD_COL].isna().sum()

print(
    f"Patients without fold assignment: {missing_folds}"
)

if missing_folds > 0:

    print(
        data.loc[
            data[FOLD_COL].isna(),
            ID_COL
        ].tolist()
    )

    raise RuntimeError(
        "Some patients do not have a validation fold assignment."
    )

data[FOLD_COL] = pd.to_numeric(
    data[FOLD_COL],
    errors="coerce"
).astype(int)

print(
    "PASS - Validation fold assignments merged successfully."
)

print(
    "Fold distribution:"
)

for fold in sorted(data[FOLD_COL].unique()):

    print(
        f"Fold {fold}: "
        f"{(data[FOLD_COL] == fold).sum()} patients"
    )

# ============================================================
# STEP 5 - NUMERIC VALIDITY CHECK
# ============================================================

print()
print("STEP 5 - CHECKING FEATURE VALUES")
print("=" * 75)

data = safe_numeric(
    data,
    FEATURES + [TARGET]
)

invalid_mask = (
    data[FEATURES]
    .isna()
    .any(axis=1)
)

invalid_patients = data.loc[
    invalid_mask,
    ID_COL
].tolist()

print(
    f"Patients with missing/invalid features: "
    f"{len(invalid_patients)}"
)

if invalid_patients:

    print(
        "These patients will NOT be evaluated:"
    )

    for pid in invalid_patients:

        print(
            f" - {pid}"
        )

usable = data.loc[
    ~invalid_mask
].copy()

print(
    f"Usable patients for classification: "
    f"{len(usable)}"
)

# ============================================================
# STEP 6 - TARGET CHECK
# ============================================================

print()
print("STEP 6 - TARGET CHECK")
print("=" * 75)

usable[TARGET] = pd.to_numeric(
    usable[TARGET],
    errors="coerce"
)

usable = usable.loc[
    usable[TARGET].isin([0, 1])
].copy()

usable[TARGET] = usable[
    TARGET
].astype(int)

print(
    f"Class 0: "
    f"{(usable[TARGET] == 0).sum()}"
)

print(
    f"Class 1: "
    f"{(usable[TARGET] == 1).sum()}"
)

# ============================================================
# STEP 7 - VERIFY 5 FOLDS
# ============================================================

print()
print("STEP 7 - VERIFYING 5-FOLD STRUCTURE")
print("=" * 75)

unique_folds = sorted(
    usable[FOLD_COL]
    .unique()
    .tolist()
)

print(
    f"Folds found: {unique_folds}"
)

if unique_folds != [1, 2, 3, 4, 5]:

    raise RuntimeError(
        "Expected exactly folds 1,2,3,4,5."
    )

for fold in unique_folds:

    subset = usable[
        usable[FOLD_COL] == fold
    ]

    print(
        f"Fold {fold}: "
        f"{len(subset)} patients | "
        f"Class 0 = "
        f"{(subset[TARGET] == 0).sum()} | "
        f"Class 1 = "
        f"{(subset[TARGET] == 1).sum()}"
    )

print(
    "PASS - 5-fold structure verified."
)

# ============================================================
# STEP 8 - CROSS-VALIDATION
# ============================================================

print()
print("STEP 8 - RUNNING 5-FOLD LINEAR SVM")
print("=" * 75)

all_predictions = []

fold_results = []

feature_selection_records = []

for fold in unique_folds:

    print()
    print("-" * 75)
    print(
        f"FOLD {fold}"
    )
    print("-" * 75)

    train = usable[
        usable[FOLD_COL] != fold
    ].copy()

    validation = usable[
        usable[FOLD_COL] == fold
    ].copy()

    print(
        f"Training patients: {len(train)}"
    )

    print(
        f"Validation patients: "
        f"{len(validation)}"
    )

    X_train = train[
        FEATURES
    ].copy()

    y_train = train[
        TARGET
    ].astype(int).values

    X_val = validation[
        FEATURES
    ].copy()

    y_val = validation[
        TARGET
    ].astype(int).values

    # --------------------------------------------------------
    # FEATURE SELECTION - TRAINING FOLD ONLY
    # --------------------------------------------------------

    selected_features, correlations = (
        select_features_inside_training(
            X_train,
            y_train,
            FEATURES
        )
    )

    print()
    print(
        "Selected features inside training fold:"
    )

    for feature in selected_features:

        print(
            f" - {feature} "
            f"(abs correlation = "
            f"{correlations.get(feature, 0):.6f})"
        )

    for feature in FEATURES:

        feature_selection_records.append({

            "Fold": fold,

            "Feature": feature,

            "Selected": int(
                feature in selected_features
            ),

            "Training_Absolute_Correlation":
                correlations.get(
                    feature,
                    np.nan
                )
        })

    # --------------------------------------------------------
    # SELECT FEATURES
    # --------------------------------------------------------

    X_train_selected = X_train[
        selected_features
    ].astype(float)

    X_val_selected = X_val[
        selected_features
    ].astype(float)

    # --------------------------------------------------------
    # STANDARDIZATION - TRAINING FOLD ONLY
    # --------------------------------------------------------

    train_mean = X_train_selected.mean(
        axis=0
    )

    train_std = X_train_selected.std(
        axis=0,
        ddof=0
    )

    train_std = train_std.replace(
        0,
        1.0
    )

    X_train_scaled = (
        X_train_selected - train_mean
    ) / train_std

    X_val_scaled = (
        X_val_selected - train_mean
    ) / train_std

    # --------------------------------------------------------
    # LINEAR SVM
    # --------------------------------------------------------

    model = SVC(
        kernel="linear",
        probability=True,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        X_val_scaled
    )

    y_prob = model.predict_proba(
        X_val_scaled
    )[:, 1]

    (
        accuracy,
        sensitivity,
        specificity,
        tn,
        fp,
        fn,
        tp
    ) = calculate_metrics(
        y_val,
        y_pred
    )

    try:

        auc = roc_auc_score(
            y_val,
            y_prob
        )

    except ValueError:

        auc = np.nan

    print()

    print(
        f"Fold {fold} Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Fold {fold} Sensitivity: "
        f"{sensitivity:.4f}"
    )

    print(
        f"Fold {fold} Specificity: "
        f"{specificity:.4f}"
    )

    print(
        f"Fold {fold} AUC: "
        f"{auc:.4f}"
    )

    # --------------------------------------------------------
    # PATIENT PREDICTIONS
    # --------------------------------------------------------

    for i in range(
        len(validation)
    ):

        all_predictions.append({

            "PatientID":
                validation.iloc[i][ID_COL],

            "True_Label":
                int(y_val[i]),

            "Predicted_Label":
                int(y_pred[i]),

            "Predicted_Probability_Class_1":
                float(y_prob[i]),

            "Validation_Fold":
                int(fold),

            "Selected_Features":
                ",".join(
                    selected_features
                )
        })

    # --------------------------------------------------------
    # FOLD RESULTS
    # --------------------------------------------------------

    fold_results.append({

        "Fold": fold,

        "Training_Patients":
            len(train),

        "Validation_Patients":
            len(validation),

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

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp
    })

# ============================================================
# STEP 9 - COMBINE PATIENT PREDICTIONS
# ============================================================

print()
print(
    "STEP 9 - COMBINING PATIENT PREDICTIONS"
)
print("=" * 75)

predictions_df = pd.DataFrame(
    all_predictions
)

if len(predictions_df) != len(usable):

    raise RuntimeError(
        "Prediction count does not match usable patient count."
    )

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"Saved: {PREDICTIONS_FILE}"
)

# ============================================================
# STEP 10 - OVERALL PERFORMANCE
# ============================================================

print()
print(
    "STEP 10 - OVERALL PERFORMANCE"
)
print("=" * 75)

y_true_all = predictions_df[
    "True_Label"
].values

y_pred_all = predictions_df[
    "Predicted_Label"
].values

y_prob_all = predictions_df[
    "Predicted_Probability_Class_1"
].values

(
    overall_accuracy,
    overall_sensitivity,
    overall_specificity,
    overall_tn,
    overall_fp,
    overall_fn,
    overall_tp
) = calculate_metrics(
    y_true_all,
    y_pred_all
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

# ============================================================
# STEP 11 - SAVE FOLD RESULTS
# ============================================================

print()
print(
    "STEP 11 - SAVING FOLD RESULTS"
)
print("=" * 75)

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_df.to_csv(
    FOLD_RESULTS_FILE,
    index=False
)

print(
    f"Saved: {FOLD_RESULTS_FILE}"
)

# ============================================================
# STEP 12 - SAVE FEATURE SELECTION LOG
# ============================================================

print()
print(
    "STEP 12 - SAVING FEATURE SELECTION LOG"
)
print("=" * 75)

feature_selection_df = pd.DataFrame(
    feature_selection_records
)

feature_selection_df.to_csv(
    FEATURE_SELECTION_FILE,
    index=False
)

print(
    f"Saved: {FEATURE_SELECTION_FILE}"
)

# ============================================================
# STEP 13 - SAVE OVERALL RESULTS
# ============================================================

print()
print(
    "STEP 13 - SAVING OVERALL RESULTS"
)
print("=" * 75)

overall_df = pd.DataFrame([{

    "Classifier":
        "Linear SVM",

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

    "TN":
        overall_tn,

    "FP":
        overall_fp,

    "FN":
        overall_fn,

    "TP":
        overall_tp,

    "Number_of_Folds":
        5,

    "Random_Seed":
        RANDOM_STATE,

    "Feature_Selection":
        "Inside training folds",

    "Normalization":
        "Training-fold mean/std only",

    "Kernel":
        "Linear"
}])

overall_df.to_csv(
    OVERALL_RESULTS_FILE,
    index=False
)

print(
    f"Saved: {OVERALL_RESULTS_FILE}"
)

# ============================================================
# STEP 14 - SAVE REPORT
# ============================================================

print()
print(
    "STEP 14 - SAVING REPORT"
)
print("=" * 75)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 25 - LINEAR SVM CLASSIFIER\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "METHODOLOGY\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "Endpoint: Two-Year Survival\n"
    )

    f.write(
        "Classifier: Support Vector Machine\n"
    )

    f.write(
        "Kernel: Linear\n"
    )

    f.write(
        "Cross-validation: Stratified patient-level 5-fold CV\n"
    )

    f.write(
        "Random seed: 42\n"
    )

    f.write(
        "Feature selection: Performed inside each training fold\n"
    )

    f.write(
        "Normalization: Training-fold mean/std only\n"
    )

    f.write(
        "Data leakage prevention: Yes\n\n"
    )

    f.write(
        "DATASET\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        f"Original STEP 21 patients: {len(data)}\n"
    )

    f.write(
        f"Usable patients evaluated: "
        f"{len(predictions_df)}\n"
    )

    f.write(
        f"Excluded patients: "
        f"{len(data) - len(predictions_df)}\n"
    )

    f.write(
        f"Class 0: "
        f"{(y_true_all == 0).sum()}\n"
    )

    f.write(
        f"Class 1: "
        f"{(y_true_all == 1).sum()}\n\n"
    )

    f.write(
        "OVERALL RESULTS\n"
    )

    f.write(
        "-" * 75 + "\n"
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
        f"TN: {overall_tn}\n"
    )

    f.write(
        f"FP: {overall_fp}\n"
    )

    f.write(
        f"FN: {overall_fn}\n"
    )

    f.write(
        f"TP: {overall_tp}\n\n"
    )

    f.write(
        "FOLD RESULTS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        fold_results_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "FEATURE SELECTION\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    for fold in unique_folds:

        selected = feature_selection_df[
            (
                feature_selection_df["Fold"]
                == fold
            )
            &
            (
                feature_selection_df["Selected"]
                == 1
            )
        ]["Feature"].tolist()

        f.write(
            f"Fold {fold}: "
            + ", ".join(selected)
            + "\n"
        )

    f.write(
        "\n"
    )

    f.write(
        "NOTE\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "Feature selection was performed independently "
        "inside each training fold.\n"
    )

    f.write(
        "Validation patients were not used for feature selection "
        "or normalization.\n"
    )

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 25 COMPLETE")
print("=" * 75)

print()

print(
    "Linear SVM classifier completed successfully."
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
    PREDICTIONS_FILE
)

print(
    FOLD_RESULTS_FILE
)

print(
    OVERALL_RESULTS_FILE
)

print(
    FEATURE_SELECTION_FILE
)

print(
    REPORT_FILE
)

print()
print("=" * 75)
print("READY FOR NEXT CLASSIFIER")
print("=" * 75)


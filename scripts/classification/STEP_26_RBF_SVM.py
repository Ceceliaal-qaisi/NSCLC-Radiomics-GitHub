import os
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 26 - RBF SVM CLASSIFIER
# ============================================================
#
# Methodological requirements:
# - Two-year survival binary endpoint
# - Patient-level samples
# - Same stratified 5-fold CV from STEP 22
# - No data leakage
# - Feature selection INSIDE each training fold
# - Standardization INSIDE each training fold
# - RBF SVM comparator
# - Patient-level predictions
# - Fold-level and overall results
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 26 - RBF SVM CLASSIFIER")
print("=" * 75)


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
    "STEP_26_RBF_SVM"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

predictions_file = os.path.join(
    OUTPUT_DIR,
    "STEP_26_Patient_Predictions.csv"
)

fold_file = os.path.join(
    OUTPUT_DIR,
    "STEP_26_Fold_Results.csv"
)

feature_selection_file = os.path.join(
    OUTPUT_DIR,
    "STEP_26_Feature_Selection_By_Fold.csv"
)

roc_file = os.path.join(
    OUTPUT_DIR,
    "STEP_26_ROC_Data.csv"
)

overall_file = os.path.join(
    OUTPUT_DIR,
    "STEP_26_Overall_Results.csv"
)

report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_26_RBF_SVM_Report.txt"
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

# IMPORTANT:
# STEP 21 uses PatientID, not Patient_ID.
ID_COLUMN = "PatientID"

# STEP 22 uses Fold, not Validation_Fold.
FOLD_COLUMN = "Fold"

RANDOM_STATE = 42


# ============================================================
# STEP 1 - LOAD STEP 21 DATA
# ============================================================

print("\nSTEP 1 - LOADING STEP 21 DATA")
print("=" * 75)

if not os.path.exists(STEP21_FILE):
    raise FileNotFoundError(
        f"STEP 21 file not found:\n{STEP21_FILE}"
    )

df = pd.read_csv(
    STEP21_FILE
)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Total columns: {len(df.columns)}"
)


# ============================================================
# STEP 2 - REQUIRED COLUMN CHECK
# ============================================================

print("\nSTEP 2 - CHECKING REQUIRED COLUMNS")
print("=" * 75)

required_columns = (
    [ID_COLUMN, TARGET]
    + FEATURES
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )

print(
    "PASS - All required columns found."
)

print(
    f"Patient ID column: {ID_COLUMN}"
)


# ============================================================
# STEP 3 - LOAD STEP 22 FOLD ASSIGNMENTS
# ============================================================

print("\nSTEP 3 - LOADING STEP 22 FOLD ASSIGNMENTS")
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


# ============================================================
# STEP 3A - VERIFY STEP 22 COLUMNS
# ============================================================

if ID_COLUMN not in folds.columns:

    # Allow automatic conversion if STEP 22 has Patient_ID
    if "Patient_ID" in folds.columns:

        folds = folds.rename(
            columns={
                "Patient_ID": ID_COLUMN
            }
        )

    else:

        raise RuntimeError(
            "Could not find patient ID column in STEP 22 file.\n"
            f"Available columns: {list(folds.columns)}"
        )


if FOLD_COLUMN not in folds.columns:

    possible_fold_columns = [
        "Validation_Fold",
        "fold",
        "CV_Fold",
        "Fold_ID"
    ]

    detected_fold = None

    for col in possible_fold_columns:

        if col in folds.columns:

            detected_fold = col
            break

    if detected_fold is None:

        raise RuntimeError(
            "Could not find fold column in STEP 22 file.\n"
            f"Available columns: {list(folds.columns)}"
        )

    folds = folds.rename(
        columns={
            detected_fold: FOLD_COLUMN
        }
    )


print(
    f"PASS - Patient ID column: {ID_COLUMN}"
)

print(
    f"PASS - Fold column: {FOLD_COLUMN}"
)


# ============================================================
# STEP 4 - PROCESS STEP 22 FOLD ASSIGNMENTS
# ============================================================

print("\nSTEP 4 - PROCESSING PATIENT FOLD ASSIGNMENTS")
print("=" * 75)

# STEP 22 contains multiple rows per patient:
# one training/validation record for each fold.
#
# We need ONE final fold assignment per patient.
#
# The validation row identifies the patient's actual fold.
# Therefore we use Role == Validation when available.

folds[ID_COLUMN] = (
    folds[ID_COLUMN]
    .astype(str)
    .str.strip()
)

# ------------------------------------------------------------
# Check Role column
# ------------------------------------------------------------

if "Role" in folds.columns:

    validation_rows = folds[
        folds["Role"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("validation")
    ].copy()

    print(
        f"Validation rows found: {len(validation_rows)}"
    )

else:

    validation_rows = pd.DataFrame()


# ============================================================
# EXTRACT ONE FOLD PER PATIENT
# ============================================================

if len(validation_rows) > 0:

    folds_small = validation_rows[
        [
            ID_COLUMN,
            FOLD_COLUMN
        ]
    ].copy()

else:

    # Fallback:
    # If Role is unavailable, use unique patient/fold pairs.
    folds_small = folds[
        [
            ID_COLUMN,
            FOLD_COLUMN
        ]
    ].drop_duplicates()


# ------------------------------------------------------------
# Convert fold to numeric
# ------------------------------------------------------------

folds_small[FOLD_COLUMN] = pd.to_numeric(
    folds_small[FOLD_COLUMN],
    errors="coerce"
)

folds_small = folds_small.dropna(
    subset=[FOLD_COLUMN]
)

folds_small[FOLD_COLUMN] = (
    folds_small[FOLD_COLUMN]
    .astype(int)
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_patients = folds_small[
    folds_small[ID_COLUMN].duplicated(
        keep=False
    )
][ID_COLUMN].unique()


if len(duplicate_patients) > 0:

    # If duplicate PatientID remains, verify that each patient
    # has only one fold.

    duplicate_check = (
        folds_small
        .groupby(ID_COLUMN)[FOLD_COLUMN]
        .nunique()
    )

    conflicting = duplicate_check[
        duplicate_check > 1
    ]

    if len(conflicting) > 0:

        raise RuntimeError(
            "Some patients have conflicting fold assignments:\n"
            + str(conflicting.index.tolist())
        )

    folds_small = (
        folds_small
        .drop_duplicates(
            subset=[ID_COLUMN]
        )
    )


print(
    f"Unique patients with fold assignments: "
    f"{len(folds_small)}"
)


# ============================================================
# STEP 5 - MERGE FOLD INFORMATION
# ============================================================

print("\nSTEP 5 - MERGING PATIENT FOLD INFORMATION")
print("=" * 75)

df[ID_COLUMN] = (
    df[ID_COLUMN]
    .astype(str)
    .str.strip()
)

if df[ID_COLUMN].duplicated().any():

    duplicate_ids = (
        df.loc[
            df[ID_COLUMN].duplicated(
                keep=False
            ),
            ID_COLUMN
        ]
        .unique()
        .tolist()
    )

    raise RuntimeError(
        "Duplicate PatientID found in STEP 21 dataset:\n"
        + str(duplicate_ids)
    )


merged = df.merge(
    folds_small,
    on=ID_COLUMN,
    how="left",
    validate="one_to_one"
)


missing_fold_mask = (
    merged[FOLD_COLUMN]
    .isna()
)

missing_fold_count = (
    missing_fold_mask.sum()
)

print(
    f"Patients without fold assignment: "
    f"{missing_fold_count}"
)

if missing_fold_count > 0:

    missing_patients = (
        merged.loc[
            missing_fold_mask,
            ID_COLUMN
        ]
        .tolist()
    )

    print(
        "Missing fold patients:"
    )

    for pid in missing_patients:
        print(
            f" - {pid}"
        )

    raise RuntimeError(
        "Some patients do not have fold assignments."
    )


merged[FOLD_COLUMN] = (
    pd.to_numeric(
        merged[FOLD_COLUMN],
        errors="coerce"
    )
    .astype(int)
)

print(
    "PASS - Fold information merged successfully."
)


# ============================================================
# STEP 6 - PREPARE NUMERIC FEATURES
# ============================================================

print("\nSTEP 6 - PREPARING NUMERIC FEATURES")
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


# Replace infinite values

merged[FEATURES] = (
    merged[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


merged[TARGET] = (
    merged[TARGET]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


before = len(merged)


# ============================================================
# VALIDITY MASK
# ============================================================

valid_mask = (
    merged[FEATURES]
    .notna()
    .all(axis=1)
    &
    merged[TARGET]
    .notna()
)


usable = merged.loc[
    valid_mask
].copy()


removed = (
    before
    - len(usable)
)


print(
    f"Original patients: {before}"
)

print(
    f"Usable patients: {len(usable)}"
)

print(
    f"Removed patients with missing values: {removed}"
)


if len(usable) == 0:

    raise RuntimeError(
        "No usable patients remain."
    )


# ============================================================
# STEP 7 - TARGET CHECK
# ============================================================

print("\nSTEP 7 - TARGET CHECK")
print("=" * 75)

usable[TARGET] = (
    usable[TARGET]
    .astype(int)
)


invalid_target = ~usable[
    TARGET
].isin([0, 1])


if invalid_target.any():

    print(
        "WARNING - Invalid target values found."
    )

    print(
        usable.loc[
            invalid_target,
            [ID_COLUMN, TARGET]
        ]
    )

    usable = usable.loc[
        ~invalid_target
    ].copy()


if usable[TARGET].nunique() != 2:

    raise RuntimeError(
        "Two-class classification requires both class 0 and class 1."
    )


class_counts = (
    usable[TARGET]
    .value_counts()
    .sort_index()
)


print(
    f"Class 0: {class_counts.get(0, 0)}"
)

print(
    f"Class 1: {class_counts.get(1, 0)}"
)


# ============================================================
# STEP 8 - VERIFY 5 FOLDS
# ============================================================

print("\nSTEP 8 - VERIFYING 5-FOLD STRUCTURE")
print("=" * 75)

unique_folds = sorted(
    usable[FOLD_COLUMN]
    .unique()
    .tolist()
)


print(
    f"Folds found: {unique_folds}"
)


if unique_folds != [1, 2, 3, 4, 5]:

    raise RuntimeError(
        "Expected exactly folds 1,2,3,4,5.\n"
        f"Found: {unique_folds}"
    )


for fold in unique_folds:

    subset = usable[
        usable[FOLD_COLUMN] == fold
    ]

    class0 = (
        subset[TARGET] == 0
    ).sum()

    class1 = (
        subset[TARGET] == 1
    ).sum()

    print(
        f"Fold {fold}: "
        f"{len(subset)} patients | "
        f"Class 0 = {class0} | "
        f"Class 1 = {class1}"
    )


print(
    "PASS - 5-fold structure verified."
)


# ============================================================
# STEP 9 - RBF SVM PARAMETERS
# ============================================================

print("\nSTEP 9 - RBF SVM PARAMETERS")
print("=" * 75)

print(
    "Kernel: RBF"
)

print(
    "C: 1.0"
)

print(
    "Gamma: scale"
)

print(
    "Probability estimates: True"
)

print(
    "Feature selection: SelectKBest"
)

print(
    "Features selected inside each training fold."
)

print(
    "Normalization inside each training fold."
)


# ============================================================
# STEP 10 - CROSS-VALIDATION
# ============================================================

print("\nSTEP 10 - RUNNING 5-FOLD RBF SVM")
print("=" * 75)


all_predictions = []

fold_results = []

feature_selection_results = []


# ============================================================
# FOLD LOOP
# ============================================================

for fold in unique_folds:

    print("\n")
    print("-" * 75)
    print(
        f"PROCESSING FOLD {fold}"
    )
    print("-" * 75)


    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    train_df = usable[
        usable[FOLD_COLUMN] != fold
    ].copy()


    test_df = usable[
        usable[FOLD_COLUMN] == fold
    ].copy()


    print(
        f"Training patients: {len(train_df)}"
    )

    print(
        f"Validation patients: {len(test_df)}"
    )


    X_train = (
        train_df[FEATURES]
        .astype(float)
        .values
    )

    y_train = (
        train_df[TARGET]
        .astype(int)
        .values
    )


    X_test = (
        test_df[FEATURES]
        .astype(float)
        .values
    )

    y_test = (
        test_df[TARGET]
        .astype(int)
        .values
    )


    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    print()
    print(
        "FEATURE SELECTION - TRAINING FOLD ONLY"
    )


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


    print(
        "Selected features:"
    )


    for feature in selected_features:

        print(
            f" - {feature}"
        )


    # --------------------------------------------------------
    # SAVE FEATURE SELECTION
    # --------------------------------------------------------

    scores = selector.scores_


    for i, feature in enumerate(FEATURES):

        score = scores[i]

        if pd.isna(score):

            score = np.nan

        feature_selection_results.append({
            "Fold": int(fold),
            "Feature": feature,
            "Selected": int(
                selected_mask[i]
            ),
            "ANOVA_F_Score": score
        })


    # ========================================================
    # STANDARDIZATION
    # ========================================================

    print()
    print(
        "STANDARDIZATION - TRAINING FOLD ONLY"
    )


    scaler = StandardScaler()


    X_train_scaled = (
        scaler.fit_transform(
            X_train_selected
        )
    )


    X_test_scaled = (
        scaler.transform(
            X_test_selected
        )
    )


    # ========================================================
    # TRAIN RBF SVM
    # ========================================================

    print()
    print(
        "TRAINING RBF SVM"
    )


    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(
        X_test_scaled
    )


    y_prob = (
        model.predict_proba(
            X_test_scaled
        )[:, 1]
    )


    # ========================================================
    # METRICS
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


    tn, fp, fn, tp = (
        cm.ravel()
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


    if len(
        np.unique(y_test)
    ) == 2:

        auc = roc_auc_score(
            y_test,
            y_prob
        )

    else:

        auc = np.nan


    print()
    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Sensitivity: {sensitivity:.4f}"
    )

    print(
        f"Specificity: {specificity:.4f}"
    )

    if not np.isnan(auc):

        print(
            f"AUC: {auc:.4f}"
        )

    else:

        print(
            "AUC: N/A"
        )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    fold_results.append({

        "Fold": int(fold),

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
    # SAVE PATIENT PREDICTIONS
    # ========================================================

    for i in range(
        len(test_df)
    ):

        all_predictions.append({

            "PatientID":
                test_df.iloc[i][ID_COLUMN],

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
# STEP 11 - COMBINE PREDICTIONS
# ============================================================

print("\nSTEP 11 - COMBINING PATIENT PREDICTIONS")
print("=" * 75)


predictions_df = pd.DataFrame(
    all_predictions
)


if len(predictions_df) != len(usable):

    raise RuntimeError(
        "Prediction count does not match usable patient count.\n"
        f"Predictions: {len(predictions_df)}\n"
        f"Usable: {len(usable)}"
    )


print(
    f"Total patient predictions: "
    f"{len(predictions_df)}"
)


# ============================================================
# STEP 12 - OVERALL OUT-OF-FOLD PERFORMANCE
# ============================================================

print("\nSTEP 12 - OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 75)


y_true_all = (
    predictions_df[
        "True_Label"
    ].values
)


y_pred_all = (
    predictions_df[
        "Predicted_Label"
    ].values
)


y_prob_all = (
    predictions_df[
        "Predicted_Probability"
    ].values
)


accuracy = accuracy_score(
    y_true_all,
    y_pred_all
)


cm = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=[0, 1]
)


tn, fp, fn, tp = (
    cm.ravel()
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


auc = roc_auc_score(
    y_true_all,
    y_prob_all
)


print(
    f"Patients evaluated: "
    f"{len(predictions_df)}"
)


print(
    f"Accuracy: "
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


print(
    f"AUC: "
    f"{auc:.4f}"
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
# STEP 13 - ROC DATA
# ============================================================

print("\nSTEP 13 - GENERATING ROC DATA")
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
# STEP 14 - SAVE PATIENT PREDICTIONS
# ============================================================

print("\nSTEP 14 - SAVING PATIENT PREDICTIONS")
print("=" * 75)


predictions_df.to_csv(
    predictions_file,
    index=False
)


print(
    f"Saved: {predictions_file}"
)


# ============================================================
# STEP 15 - SAVE FOLD RESULTS
# ============================================================

print("\nSTEP 15 - SAVING FOLD RESULTS")
print("=" * 75)


fold_results_df = pd.DataFrame(
    fold_results
)


fold_results_df.to_csv(
    fold_file,
    index=False
)


print(
    f"Saved: {fold_file}"
)


# ============================================================
# STEP 16 - SAVE FEATURE SELECTION
# ============================================================

print("\nSTEP 16 - SAVING FEATURE SELECTION")
print("=" * 75)


feature_selection_df = pd.DataFrame(
    feature_selection_results
)


feature_selection_df.to_csv(
    feature_selection_file,
    index=False
)


print(
    f"Saved: {feature_selection_file}"
)


# ============================================================
# STEP 17 - SAVE ROC DATA
# ============================================================

print("\nSTEP 17 - SAVING ROC DATA")
print("=" * 75)


roc_df.to_csv(
    roc_file,
    index=False
)


print(
    f"Saved: {roc_file}"
)


# ============================================================
# STEP 18 - SAVE OVERALL RESULTS
# ============================================================

print("\nSTEP 18 - SAVING OVERALL RESULTS")
print("=" * 75)


overall_results = pd.DataFrame([{

    "Classifier":
        "RBF SVM",

    "Patients_Evaluated":
        len(predictions_df),

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
        tp,

    "Kernel":
        "RBF",

    "C":
        1.0,

    "Gamma":
        "scale",

    "Number_of_Folds":
        5,

    "Random_Seed":
        RANDOM_STATE,

    "Feature_Selection":
        "SelectKBest ANOVA F-test inside training fold",

    "Selected_Features_Per_Fold":
        5,

    "Normalization":
        "StandardScaler fitted inside training fold",

    "Data_Leakage_Prevention":
        "Yes"

}])


overall_results.to_csv(
    overall_file,
    index=False
)


print(
    f"Saved: {overall_file}"
)


# ============================================================
# STEP 19 - SAVE REPORT
# ============================================================

print("\nSTEP 19 - SAVING REPORT")
print("=" * 75)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 26 - RBF SVM CLASSIFIER\n"
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
        "Kernel: RBF\n"
    )

    f.write(
        "C: 1.0\n"
    )

    f.write(
        "Gamma: scale\n"
    )

    f.write(
        "Cross-validation: Stratified patient-level 5-fold CV\n"
    )

    f.write(
        "Fold assignments: STEP 22\n"
    )

    f.write(
        "Random seed: 42\n"
    )

    f.write(
        "Feature selection: SelectKBest ANOVA F-test\n"
    )

    f.write(
        "Feature selection performed inside each training fold.\n"
    )

    f.write(
        "Normalization: StandardScaler\n"
    )

    f.write(
        "Normalization fitted using training patients only.\n"
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
        f"Original STEP 21 patients: {before}\n"
    )

    f.write(
        f"Usable patients evaluated: {len(predictions_df)}\n"
    )

    f.write(
        f"Excluded patients: {removed}\n"
    )

    f.write(
        f"Class 0: {(y_true_all == 0).sum()}\n"
    )

    f.write(
        f"Class 1: {(y_true_all == 1).sum()}\n\n"
    )


    f.write(
        "OVERALL OUT-OF-FOLD RESULTS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        f"Patients evaluated: {len(predictions_df)}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Sensitivity: {sensitivity:.6f}\n"
    )

    f.write(
        f"Specificity: {specificity:.6f}\n"
    )

    f.write(
        f"AUC: {auc:.6f}\n"
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
        "FEATURE SELECTION BY FOLD\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for fold in unique_folds:

        selected = (
            feature_selection_df[
                (
                    feature_selection_df[
                        "Fold"
                    ]
                    == fold
                )
                &
                (
                    feature_selection_df[
                        "Selected"
                    ]
                    == 1
                )
            ]
            ["Feature"]
            .tolist()
        )


        f.write(
            f"Fold {fold}: "
            + ", ".join(
                selected
            )
            + "\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "METHODOLOGICAL NOTE\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "The STEP 22 file contains multiple rows per patient "
        "because each patient has a training/validation role "
        "for each cross-validation fold.\n"
    )

    f.write(
        "The validation row was used to determine the single "
        "patient-level fold assignment.\n"
    )

    f.write(
        "Feature selection was performed independently inside "
        "each training fold.\n"
    )

    f.write(
        "Standardization was fitted independently inside each "
        "training fold.\n"
    )

    f.write(
        "Validation patients were never used during feature "
        "selection or normalization.\n"
    )

    f.write(
        "Therefore, the reported predictions are out-of-fold "
        "patient-level predictions.\n"
    )


print(
    f"Saved: {report_file}"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("STEP 26 COMPLETE")
print("=" * 75)


print(
    "\nRBF SVM classifier completed successfully."
)


print(
    f"Patients evaluated: "
    f"{len(predictions_df)}"
)


print(
    f"Accuracy: "
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


print(
    f"AUC: "
    f"{auc:.4f}"
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
    predictions_file
)

print(
    fold_file
)

print(
    feature_selection_file
)

print(
    roc_file
)

print(
    overall_file
)

print(
    report_file
)


print("\n")
print("=" * 75)
print("READY FOR NEXT PHASE")
print("=" * 75)
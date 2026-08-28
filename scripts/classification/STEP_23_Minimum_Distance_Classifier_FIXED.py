
import os
import numpy as np
import pandas as pd

# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 23 - MINIMUM-DISTANCE CLASSIFIER
# ============================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

DATA_FILE = os.path.join(
    BASE_DIR,
    "STEP_21_TWO_YEAR_SURVIVAL_ENDPOINT",
    "STEP_21_Two_Year_Survival_Classification_Dataset.csv"
)

FOLD_FILE = os.path.join(
    BASE_DIR,
    "STEP_22_STRATIFIED_PATIENT_CV",
    "STEP_22_Patient_Fold_Assignments.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_23_MINIMUM_DISTANCE_CLASSIFIER"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
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
PATIENT_ID_CANDIDATES = [
    "PatientID",
    "Patient_ID",
    "patient_id",
    "patientid"
]

N_FOLDS = 5
RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Find the first available column from a list of candidates.
    """
    for column in candidates:
        if column in df.columns:
            return column

    return None


def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum(
        (y_true == 1) &
        (y_pred == 1)
    )

    tn = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    fp = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1) &
        (y_pred == 0)
    )

    total = len(y_true)

    accuracy = (
        (tp + tn) / total
        if total > 0 else np.nan
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0 else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else np.nan
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else np.nan
    )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 23 - MINIMUM-DISTANCE CLASSIFIER")
print("=" * 75)

print("\nClassifier: Euclidean Minimum-Distance")
print("Cross-validation: Stratified Patient-Level 5-Fold")
print("Random state:", RANDOM_STATE)


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

print("\n")
print("STEP 1 - LOADING VERIFIED DATA")
print("=" * 75)

if not os.path.isfile(DATA_FILE):

    raise FileNotFoundError(
        "\nSTEP 21 dataset not found:\n"
        + DATA_FILE
    )

if not os.path.isfile(FOLD_FILE):

    raise FileNotFoundError(
        "\nSTEP 22 fold assignment file not found:\n"
        + FOLD_FILE
    )

data = pd.read_csv(DATA_FILE)

folds = pd.read_csv(FOLD_FILE)

print("STEP 21 dataset loaded successfully.")
print("Original rows:", len(data))
print("STEP 21 columns:", len(data.columns))

print("\nSTEP 22 fold assignment file loaded successfully.")
print("Fold file rows:", len(folds))
print("Fold file columns:", len(folds.columns))


# ============================================================
# STEP 2 - IDENTIFY PATIENT ID COLUMN
# ============================================================

print("\n")
print("STEP 2 - PATIENT ID COLUMN CHECK")
print("=" * 75)

patient_col_data = find_column(
    data,
    PATIENT_ID_CANDIDATES
)

if patient_col_data is None:

    print("Available STEP 21 columns:")

    for column in data.columns:
        print("-", column)

    raise RuntimeError(
        "\nCould not find Patient ID column in STEP 21 dataset."
    )

print(
    "Detected STEP 21 Patient ID column:",
    patient_col_data
)


patient_col_fold = find_column(
    folds,
    PATIENT_ID_CANDIDATES
)

if patient_col_fold is None:

    raise RuntimeError(
        "\nCould not find Patient ID column in STEP 22 fold file."
    )

print(
    "Detected STEP 22 Patient ID column:",
    patient_col_fold
)


# ============================================================
# STEP 3 - REQUIRED COLUMN CHECK
# ============================================================

print("\n")
print("STEP 3 - REQUIRED COLUMN CHECK")
print("=" * 75)

required_data_columns = [
    patient_col_data,
    TARGET
] + FEATURES

missing_data_columns = [
    column
    for column in required_data_columns
    if column not in data.columns
]

if missing_data_columns:

    print("MISSING COLUMNS:")

    for column in missing_data_columns:
        print("-", column)

    raise RuntimeError(
        "\nRequired columns are missing from STEP 21 dataset."
    )

print("PASS - Patient ID found")
print("PASS - Two_Year_Survival found")

for feature in FEATURES:
    print("FOUND -", feature)


# ============================================================
# STEP 4 - STANDARDIZE PATIENT ID
# ============================================================

print("\n")
print("STEP 4 - PATIENT ID VALIDATION")
print("=" * 75)

data[patient_col_data] = (
    data[patient_col_data]
    .astype(str)
    .str.strip()
)

folds[patient_col_fold] = (
    folds[patient_col_fold]
    .astype(str)
    .str.strip()
)

if data[patient_col_data].duplicated().any():

    duplicated_ids = (
        data.loc[
            data[patient_col_data].duplicated(
                keep=False
            ),
            patient_col_data
        ]
        .unique()
    )

    raise RuntimeError(
        "\nDuplicate Patient IDs detected in STEP 21.\n"
        f"Number of duplicated IDs: {len(duplicated_ids)}"
    )

print(
    "Unique STEP 21 patients:",
    data[patient_col_data].nunique()
)


# ============================================================
# STEP 5 - TARGET VALIDATION
# ============================================================

print("\n")
print("STEP 5 - TARGET VALIDATION")
print("=" * 75)

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce"
)

if data[TARGET].isna().any():

    raise RuntimeError(
        "\nMissing or invalid Two_Year_Survival values detected."
    )

if not set(
    data[TARGET].astype(int).unique()
).issubset({0, 1}):

    raise RuntimeError(
        "\nTarget contains values other than 0 and 1."
    )

data[TARGET] = data[TARGET].astype(int)

print("PASS - Target contains only classes 0 and 1.")

print(
    "Class 0:",
    int((data[TARGET] == 0).sum())
)

print(
    "Class 1:",
    int((data[TARGET] == 1).sum())
)


# ============================================================
# STEP 6 - FEATURE VALIDATION
# ============================================================

print("\n")
print("STEP 6 - FEATURE VALIDATION")
print("=" * 75)

for feature in FEATURES:

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )

    print(
        feature,
        "- missing/non-numeric:",
        int(data[feature].isna().sum())
    )


# ============================================================
# STEP 7 - IDENTIFY UNUSABLE PATIENTS
# ============================================================

print("\n")
print("STEP 7 - IDENTIFYING UNUSABLE PATIENTS")
print("=" * 75)

missing_feature_mask = (
    data[FEATURES]
    .isna()
    .any(axis=1)
)

removed_patients = data[
    missing_feature_mask
].copy()

usable_data = data[
    ~missing_feature_mask
].copy()

print(
    "Original STEP 21 patients:",
    len(data)
)

print(
    "Patients with missing/non-numeric features:",
    len(removed_patients)
)

print(
    "Usable patients:",
    len(usable_data)
)

if len(removed_patients) > 0:

    print("\nRemoved / unusable patient(s):")

    for _, row in removed_patients.iterrows():

        missing_features = []

        for feature in FEATURES:

            if pd.isna(row[feature]):
                missing_features.append(feature)

        print(
            "Patient:",
            row[patient_col_data]
        )

        print(
            "Missing features:",
            ", ".join(missing_features)
        )

# ------------------------------------------------------------
# Expected result from STEP 23.1:
#
# Original patients = 420
# Unusable patients = 1
# Usable patients = 419
# ------------------------------------------------------------

if len(usable_data) != 419:

    raise RuntimeError(
        "\nUnexpected usable patient count.\n"
        f"Expected: 419\n"
        f"Found: {len(usable_data)}\n"
        "Check STEP 21 and STEP 23.1 before continuing."
    )

print(
    "\nPASS - Exactly 419 usable patients identified."
)


# ============================================================
# STEP 8 - CHECK STEP 22 FOLD FILE
# ============================================================

print("\n")
print("STEP 8 - STEP 22 FOLD ASSIGNMENT CHECK")
print("=" * 75)

fold_candidates = [
    "Fold",
    "fold",
    "Fold_ID",
    "fold_id",
    "Fold_Number",
    "fold_number",
    "CV_Fold",
    "cv_fold",
    "Patient_Fold",
    "patient_fold"
]

fold_col = find_column(
    folds,
    fold_candidates
)

if fold_col is None:

    raise RuntimeError(
        "\nCould not find fold column in STEP 22 file."
    )

print(
    "Detected fold column:",
    fold_col
)

folds[fold_col] = pd.to_numeric(
    folds[fold_col],
    errors="coerce"
)

if folds[fold_col].isna().any():

    raise RuntimeError(
        "\nMissing or invalid fold values detected."
    )

folds[fold_col] = folds[fold_col].astype(int)

unique_folds = sorted(
    folds[fold_col].unique()
)

print(
    "Fold values:",
    unique_folds
)

if unique_folds != [1, 2, 3, 4, 5]:

    raise RuntimeError(
        "\nExpected exactly 5 folds numbered 1-5."
    )


# ============================================================
# STEP 9 - CREATE CLEAN FOLD TABLE
# ============================================================

print("\n")
print("STEP 9 - PREPARING PATIENT-LEVEL FOLD TABLE")
print("=" * 75)

fold_table = folds[
    [patient_col_fold, fold_col]
].copy()

fold_table = fold_table.rename(
    columns={
        patient_col_fold: "Patient_ID",
        fold_col: "CV_Fold"
    }
)

fold_table["Patient_ID"] = (
    fold_table["Patient_ID"]
    .astype(str)
    .str.strip()
)

# ------------------------------------------------------------
# IMPORTANT:
# STEP 22 contains TWO rows per patient:
# one Training row and one Validation row for every fold.
#
# We only need the validation assignment.
# Each patient must have exactly one validation fold.
# ------------------------------------------------------------

if "Role" in folds.columns:

    validation_rows = folds[
        folds["Role"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "validation"
    ].copy()

    validation_rows = validation_rows[
        [patient_col_fold, fold_col]
    ].copy()

    validation_rows = validation_rows.rename(
        columns={
            patient_col_fold: "Patient_ID",
            fold_col: "CV_Fold"
        }
    )

    validation_rows["Patient_ID"] = (
        validation_rows["Patient_ID"]
        .astype(str)
        .str.strip()
    )

    validation_rows["CV_Fold"] = (
        pd.to_numeric(
            validation_rows["CV_Fold"],
            errors="coerce"
        )
        .astype(int)
    )

    fold_table = validation_rows.copy()

else:

    raise RuntimeError(
        "\nSTEP 22 file does not contain Role column."
    )


# ============================================================
# STEP 10 - VALIDATE ONE VALIDATION FOLD PER PATIENT
# ============================================================

print("\n")
print("STEP 10 - VALIDATING PATIENT-LEVEL FOLD ASSIGNMENTS")
print("=" * 75)

validation_counts = (
    fold_table["Patient_ID"]
    .value_counts()
)

if not (
    validation_counts == 1
).all():

    raise RuntimeError(
        "\nSome patients do not have exactly one validation fold."
    )

print(
    "PASS - Every STEP 22 patient has exactly one validation fold."
)

print(
    "Patients in STEP 22:",
    len(fold_table)
)


# ============================================================
# STEP 11 - MATCH USABLE PATIENTS WITH FOLDS
# ============================================================

print("\n")
print("STEP 11 - MATCHING USABLE PATIENTS WITH STEP 22")
print("=" * 75)

usable_ids = set(
    usable_data[patient_col_data]
)

fold_ids = set(
    fold_table["Patient_ID"]
)

common_ids = (
    usable_ids
    .intersection(fold_ids)
)

print(
    "Usable STEP 21 patients:",
    len(usable_ids)
)

print(
    "STEP 22 validation patients:",
    len(fold_ids)
)

print(
    "Common patients:",
    len(common_ids)
)

print(
    "Usable patients without fold:",
    len(usable_ids - fold_ids)
)

print(
    "Fold patients not usable:",
    len(fold_ids - usable_ids)
)

if len(common_ids) != len(usable_ids):

    raise RuntimeError(
        "\nNot all usable patients have valid fold assignments."
    )

print(
    "PASS - All 419 usable patients have fold assignments."
)


# ============================================================
# STEP 12 - MERGE DATA WITH FOLD ASSIGNMENTS
# ============================================================

print("\n")
print("STEP 12 - MERGING USABLE DATA WITH FOLDS")
print("=" * 75)

usable_data = usable_data.rename(
    columns={
        patient_col_data: "Patient_ID"
    }
)

model_data = usable_data.merge(
    fold_table,
    on="Patient_ID",
    how="inner",
    validate="one_to_one"
)

print(
    "Merged patients:",
    len(model_data)
)

if len(model_data) != 419:

    raise RuntimeError(
        "\nPatient count changed during fold merge.\n"
        f"Expected: 419\n"
        f"Found: {len(model_data)}"
    )

print(
    "PASS - 419 patients retained after fold merge."
)


# ============================================================
# STEP 13 - FINAL NUMERIC CHECK
# ============================================================

print("\n")
print("STEP 13 - FINAL NUMERIC DATA CHECK")
print("=" * 75)

for feature in FEATURES:

    model_data[feature] = pd.to_numeric(
        model_data[feature],
        errors="coerce"
    )

model_data[TARGET] = pd.to_numeric(
    model_data[TARGET],
    errors="coerce"
)

if model_data[
    FEATURES + [TARGET]
].isna().any().any():

    raise RuntimeError(
        "\nUnexpected missing values remain in model data."
    )

print(
    "PASS - No missing feature values remain."
)


# ============================================================
# STEP 14 - SAVE USABLE DATA CHECK
# ============================================================

usable_check_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_Usable_Patient_Check.csv"
)

model_data[
    [
        "Patient_ID",
        "CV_Fold",
        TARGET
    ] + FEATURES
].to_csv(
    usable_check_file,
    index=False
)

print(
    "\nSaved usable patient dataset:"
)

print(
    usable_check_file
)


# ============================================================
# STEP 15 - MINIMUM-DISTANCE CLASSIFIER
# ============================================================

print("\n")
print("STEP 15 - MINIMUM-DISTANCE CLASSIFIER")
print("=" * 75)

print(
    "Distance metric: Euclidean"
)

print(
    "Class prototypes: calculated from training patients only."
)

print(
    "Normalization: calculated from training patients only."
)

print(
    "Feature selection: none in STEP 23."
)


all_predictions = []
all_true = []

all_dist_class0 = []
all_dist_class1 = []

all_patient_ids = []
all_folds = []

fold_results = []


# ============================================================
# STEP 16 - 5-FOLD PATIENT-LEVEL CROSS-VALIDATION
# ============================================================

print("\n")
print("STEP 16 - STRATIFIED PATIENT-LEVEL 5-FOLD CV")
print("=" * 75)

for fold_number in range(1, N_FOLDS + 1):

    print("\n")
    print("-" * 70)
    print(
        f"FOLD {fold_number}"
    )
    print("-" * 70)

    train = model_data[
        model_data["CV_Fold"] != fold_number
    ].copy()

    test = model_data[
        model_data["CV_Fold"] == fold_number
    ].copy()

    print(
        "Training patients:",
        len(train)
    )

    print(
        "Validation patients:",
        len(test)
    )

    if len(test) == 0:

        raise RuntimeError(
            f"\nFold {fold_number} contains no validation patients."
        )

    # --------------------------------------------------------
    # TRAINING MATRICES
    # --------------------------------------------------------

    X_train = (
        train[FEATURES]
        .values
        .astype(float)
    )

    X_test = (
        test[FEATURES]
        .values
        .astype(float)
    )

    y_train = (
        train[TARGET]
        .values
        .astype(int)
    )

    y_test = (
        test[TARGET]
        .values
        .astype(int)
    )

    # --------------------------------------------------------
    # NORMALIZATION
    #
    # Mean and standard deviation are calculated ONLY from
    # the training patients.
    # --------------------------------------------------------

    train_mean = np.mean(
        X_train,
        axis=0
    )

    train_std = np.std(
        X_train,
        axis=0
    )

    train_std[
        train_std == 0
    ] = 1.0

    X_train_scaled = (
        X_train - train_mean
    ) / train_std

    X_test_scaled = (
        X_test - train_mean
    ) / train_std

    # --------------------------------------------------------
    # CLASS PROTOTYPES
    # --------------------------------------------------------

    class0_samples = X_train_scaled[
        y_train == 0
    ]

    class1_samples = X_train_scaled[
        y_train == 1
    ]

    if len(class0_samples) == 0:

        raise RuntimeError(
            f"\nNo class 0 training patients in fold {fold_number}."
        )

    if len(class1_samples) == 0:

        raise RuntimeError(
            f"\nNo class 1 training patients in fold {fold_number}."
        )

    prototype_0 = np.mean(
        class0_samples,
        axis=0
    )

    prototype_1 = np.mean(
        class1_samples,
        axis=0
    )

    print(
        "Training Class 0:",
        len(class0_samples)
    )

    print(
        "Training Class 1:",
        len(class1_samples)
    )

    # --------------------------------------------------------
    # EUCLIDEAN DISTANCES
    # --------------------------------------------------------

    distances_0 = np.sqrt(
        np.sum(
            (
                X_test_scaled -
                prototype_0
            ) ** 2,
            axis=1
        )
    )

    distances_1 = np.sqrt(
        np.sum(
            (
                X_test_scaled -
                prototype_1
            ) ** 2,
            axis=1
        )
    )

    # --------------------------------------------------------
    # MINIMUM-DISTANCE DECISION
    # --------------------------------------------------------

    predictions = np.where(
        distances_0 <= distances_1,
        0,
        1
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    print(
        "Accuracy:",
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        "Sensitivity:",
        f"{metrics['Sensitivity']:.4f}"
    )

    print(
        "Specificity:",
        f"{metrics['Specificity']:.4f}"
    )

    print(
        "Precision:",
        f"{metrics['Precision']:.4f}"
    )

    print(
        "Confusion Matrix:",
        f"TN={metrics['TN']},",
        f"FP={metrics['FP']},",
        f"FN={metrics['FN']},",
        f"TP={metrics['TP']}"
    )

    # --------------------------------------------------------
    # SAVE FOLD RESULTS
    # --------------------------------------------------------

    fold_results.append({

        "Fold":
            fold_number,

        "Training_Patients":
            len(train),

        "Validation_Patients":
            len(test),

        "Class0_Train":
            len(class0_samples),

        "Class1_Train":
            len(class1_samples),

        "Class0_Validation":
            int(np.sum(y_test == 0)),

        "Class1_Validation":
            int(np.sum(y_test == 1)),

        "Accuracy":
            metrics["Accuracy"],

        "Sensitivity":
            metrics["Sensitivity"],

        "Specificity":
            metrics["Specificity"],

        "Precision":
            metrics["Precision"],

        "TP":
            metrics["TP"],

        "TN":
            metrics["TN"],

        "FP":
            metrics["FP"],

        "FN":
            metrics["FN"]

    })

    # --------------------------------------------------------
    # STORE PATIENT-LEVEL RESULTS
    # --------------------------------------------------------

    all_predictions.extend(
        predictions.tolist()
    )

    all_true.extend(
        y_test.tolist()
    )

    all_dist_class0.extend(
        distances_0.tolist()
    )

    all_dist_class1.extend(
        distances_1.tolist()
    )

    all_patient_ids.extend(
        test["Patient_ID"].tolist()
    )

    all_folds.extend(
        [fold_number] * len(test)
    )


# ============================================================
# STEP 17 - FINAL PATIENT COUNT CHECK
# ============================================================

print("\n")
print("STEP 17 - FINAL PATIENT COUNT CHECK")
print("=" * 75)

print(
    "Total evaluated patients:",
    len(all_true)
)

if len(all_true) != 419:

    raise RuntimeError(
        "\nFinal evaluated patient count is not 419.\n"
        f"Found: {len(all_true)}"
    )

print(
    "PASS - Exactly 419 patients evaluated."
)


# ============================================================
# STEP 18 - CHECK EACH PATIENT WAS EVALUATED ONCE
# ============================================================

print("\n")
print("STEP 18 - PATIENT EVALUATION UNIQUENESS CHECK")
print("=" * 75)

if len(set(all_patient_ids)) != 419:

    raise RuntimeError(
        "\nSome patients were evaluated more than once "
        "or were missing from evaluation."
    )

print(
    "PASS - Every patient evaluated exactly once."
)


# ============================================================
# STEP 19 - OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("STEP 19 - OVERALL CROSS-VALIDATION RESULTS")
print("=" * 75)

overall_metrics = calculate_metrics(
    all_true,
    all_predictions
)

print(
    "Total predictions:",
    len(all_predictions)
)

print(
    "Accuracy:",
    f"{overall_metrics['Accuracy']:.6f}"
)

print(
    "Sensitivity:",
    f"{overall_metrics['Sensitivity']:.6f}"
)

print(
    "Specificity:",
    f"{overall_metrics['Specificity']:.6f}"
)

print(
    "Precision:",
    f"{overall_metrics['Precision']:.6f}"
)

print("\nConfusion Matrix:")

print(
    "TN:",
    overall_metrics["TN"],
    "FP:",
    overall_metrics["FP"]
)

print(
    "FN:",
    overall_metrics["FN"],
    "TP:",
    overall_metrics["TP"]
)


# ============================================================
# STEP 20 - SAVE PATIENT PREDICTIONS
# ============================================================

print("\n")
print("STEP 20 - SAVING PATIENT PREDICTIONS")
print("=" * 75)

prediction_df = pd.DataFrame({

    "Patient_ID":
        all_patient_ids,

    "Fold":
        all_folds,

    "True_Class":
        all_true,

    "Predicted_Class":
        all_predictions,

    "Distance_Class_0":
        all_dist_class0,

    "Distance_Class_1":
        all_dist_class1

})

prediction_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_Patient_Predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    "Saved:"
)

print(
    prediction_file
)


# ============================================================
# STEP 21 - SAVE FOLD RESULTS
# ============================================================

print("\n")
print("STEP 21 - SAVING FOLD RESULTS")
print("=" * 75)

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_Fold_Results.csv"
)

fold_results_df.to_csv(
    fold_results_file,
    index=False
)

print(
    "Saved:"
)

print(
    fold_results_file
)


# ============================================================
# STEP 22 - SAVE OVERALL RESULTS
# ============================================================

print("\n")
print("STEP 22 - SAVING OVERALL RESULTS")
print("=" * 75)

overall_df = pd.DataFrame([{

    "Classifier":
        "Minimum-Distance",

    "Patients":
        len(all_true),

    "Accuracy":
        overall_metrics["Accuracy"],

    "Sensitivity":
        overall_metrics["Sensitivity"],

    "Specificity":
        overall_metrics["Specificity"],

    "Precision":
        overall_metrics["Precision"],

    "TP":
        overall_metrics["TP"],

    "TN":
        overall_metrics["TN"],

    "FP":
        overall_metrics["FP"],

    "FN":
        overall_metrics["FN"]

}])

overall_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_Overall_Results.csv"
)

overall_df.to_csv(
    overall_file,
    index=False
)

print(
    "Saved:"
)

print(
    overall_file
)


# ============================================================
# STEP 23 - SAVE FEATURES USED
# ============================================================

print("\n")
print("STEP 23 - SAVING FEATURE LIST")
print("=" * 75)

features_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_Features_Used.txt"
)

with open(
    features_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 23 - MINIMUM-DISTANCE CLASSIFIER\n\n"
    )

    f.write(
        "Features used:\n\n"
    )

    for feature in FEATURES:

        f.write(
            f"- {feature}\n"
        )

    f.write(
        "\nNo feature selection was performed in STEP 23.\n"
    )

    f.write(
        "Normalization parameters were calculated from "
        "training patients only within each fold.\n"
    )

print(
    "Saved:"
)

print(
    features_file
)


# ============================================================
# STEP 24 - SAVE COMPLETE REPORT
# ============================================================

print("\n")
print("STEP 24 - SAVING METHODOLOGY REPORT")
print("=" * 75)

report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_Minimum_Distance_Report.txt"
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
        "STEP 23 - MINIMUM-DISTANCE CLASSIFIER\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "CLASSIFICATION TASK\n"
    )

    f.write(
        "Two-Year Survival Binary Classification\n\n"
    )

    f.write(
        "PATIENT DATA\n"
    )

    f.write(
        "Original STEP 21 patients: 420\n"
    )

    f.write(
        "Removed / unusable patients: 1\n"
    )

    f.write(
        "Patients evaluated: 419\n\n"
    )

    f.write(
        "REMOVAL CRITERION\n"
    )

    f.write(
        "A patient was considered unusable if any of the "
        "nine required radiomic features was missing "
        "or non-numeric.\n\n"
    )

    f.write(
        "CLASSIFIER\n"
    )

    f.write(
        "Minimum-Distance Classifier\n"
    )

    f.write(
        "Euclidean distance to class prototypes\n\n"
    )

    f.write(
        "CROSS-VALIDATION\n"
    )

    f.write(
        "Stratified patient-level 5-fold cross-validation\n"
    )

    f.write(
        "Random state: 42\n\n"
    )

    f.write(
        "NORMALIZATION\n"
    )

    f.write(
        "Feature mean and standard deviation were "
        "calculated using training patients only "
        "within each fold.\n\n"
    )

    f.write(
        "CLASS PROTOTYPES\n"
    )

    f.write(
        "Class prototypes were calculated from the "
        "training patients only within each fold.\n\n"
    )

    f.write(
        "FEATURE SELECTION\n"
    )

    f.write(
        "No feature selection was performed in STEP 23.\n"
    )

    f.write(
        "Any future feature selection must be performed "
        "inside the training portion of each fold.\n\n"
    )

    f.write(
        "FEATURES\n"
    )

    for feature in FEATURES:

        f.write(
            f"- {feature}\n"
        )

    f.write("\n")

    f.write(
        "OVERALL RESULTS\n"
    )

    f.write(
        f"Patients evaluated: {len(all_true)}\n"
    )

    f.write(
        f"Accuracy: "
        f"{overall_metrics['Accuracy']:.6f}\n"
    )

    f.write(
        f"Sensitivity: "
        f"{overall_metrics['Sensitivity']:.6f}\n"
    )

    f.write(
        f"Specificity: "
        f"{overall_metrics['Specificity']:.6f}\n"
    )

    f.write(
        f"Precision: "
        f"{overall_metrics['Precision']:.6f}\n\n"
    )

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        f"TN = {overall_metrics['TN']}\n"
    )

    f.write(
        f"FP = {overall_metrics['FP']}\n"
    )

    f.write(
        f"FN = {overall_metrics['FN']}\n"
    )

    f.write(
        f"TP = {overall_metrics['TP']}\n"
    )

print(
    "Saved:"
)

print(
    report_file
)


# ============================================================
# STEP 25 - FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("STEP 23 COMPLETE")
print("=" * 75)

print(
    "\nOriginal STEP 21 patients:",
    len(data)
)

print(
    "Removed / unusable patients:",
    len(removed_patients)
)

print(
    "Usable patients:",
    len(model_data)
)

print(
    "Patients evaluated:",
    len(all_true)
)

print(
    "\nMinimum-Distance classifier completed successfully."
)

print(
    "\nAccuracy:",
    f"{overall_metrics['Accuracy']:.4f}"
)

print(
    "Sensitivity:",
    f"{overall_metrics['Sensitivity']:.4f}"
)

print(
    "Specificity:",
    f"{overall_metrics['Specificity']:.4f}"
)

print(
    "Precision:",
    f"{overall_metrics['Precision']:.4f}"
)

print("\nOUTPUT DIRECTORY:")
print(
    OUTPUT_DIR
)

print("\nOUTPUT FILES:")
print(
    prediction_file
)
print(
    fold_results_file
)
print(
    overall_file
)
print(
    features_file
)
print(
    report_file
)
print(
    usable_check_file
)

print("\n")
print("=" * 75)
print("READY FOR NEXT CLASSIFIER")
print("=" * 75)


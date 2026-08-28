
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

# IMPORTANT:
# STEP 21 uses PatientID, not Patient_ID.
PATIENT_ID = "PatientID"

TARGET = "Two_Year_Survival"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

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

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else np.nan
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

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else np.nan
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


# ============================================================
# STEP 1 - LOAD VERIFIED DATA
# ============================================================

print("\nSTEP 1 - LOADING VERIFIED DATA")
print("=" * 75)

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"STEP 21 dataset not found:\n{DATA_FILE}"
    )

if not os.path.exists(FOLD_FILE):

    raise FileNotFoundError(
        f"STEP 22 fold file not found:\n{FOLD_FILE}"
    )


data = pd.read_csv(DATA_FILE)

folds = pd.read_csv(FOLD_FILE)


print("Radiomic dataset loaded.")
print("Rows:", len(data))
print("Columns:", len(data.columns))

print("\nFold assignment file loaded.")
print("Rows:", len(folds))
print("Columns:", len(folds.columns))


# ============================================================
# STEP 2 - CHECK REQUIRED COLUMNS
# ============================================================

print("\nSTEP 2 - REQUIRED COLUMN CHECK")
print("=" * 75)

required_data_columns = [
    PATIENT_ID,
    TARGET
] + FEATURES


missing_data_columns = [
    c
    for c in required_data_columns
    if c not in data.columns
]


if missing_data_columns:

    print("MISSING COLUMNS:")

    for c in missing_data_columns:
        print("-", c)

    raise RuntimeError(
        "Required columns are missing from STEP 21 dataset."
    )


print("PASS - PatientID found")
print("PASS - Target found")


for feature in FEATURES:

    print("FOUND -", feature)


# ============================================================
# STEP 3 - PATIENT ID CHECK
# ============================================================

print("\nSTEP 3 - PATIENT ID CHECK")
print("=" * 75)

data[PATIENT_ID] = (
    data[PATIENT_ID]
    .astype(str)
    .str.strip()
)


print(
    "Total patients:",
    data[PATIENT_ID].nunique()
)


print(
    "Duplicate PatientID:",
    data[PATIENT_ID].duplicated().sum()
)


if data[PATIENT_ID].duplicated().any():

    raise RuntimeError(
        "Duplicate PatientID detected."
    )


# ============================================================
# STEP 4 - FIND PATIENT ID IN FOLD FILE
# ============================================================

print("\nSTEP 4 - FOLD ASSIGNMENT CHECK")
print("=" * 75)

print("STEP 22 columns:")

for c in folds.columns:

    print("-", c)


fold_id_candidates = [
    "Patient_ID",
    "PatientID",
    "patient_id",
    "patientid"
]


fold_id_col = find_column(
    folds,
    fold_id_candidates
)


if fold_id_col is None:

    raise RuntimeError(
        "Could not find Patient ID column in STEP 22 file."
    )


folds[fold_id_col] = (
    folds[fold_id_col]
    .astype(str)
    .str.strip()
)


print(
    "\nDetected Patient ID column:",
    fold_id_col
)


# ============================================================
# STEP 5 - FIND FOLD COLUMN
# ============================================================

print("\nSTEP 5 - FINDING FOLD COLUMN")
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
    "patient_fold",
    "Stratified_Fold",
    "stratified_fold"
]


fold_col = find_column(
    folds,
    fold_candidates
)


# Automatic numeric detection if standard name is absent

if fold_col is None:

    for col in folds.columns:

        if col == fold_id_col:
            continue

        numeric_values = pd.to_numeric(
            folds[col],
            errors="coerce"
        )

        if numeric_values.notna().all():

            unique_values = sorted(
                numeric_values.astype(int).unique()
            )

            if (
                len(unique_values) == 5
                and set(unique_values).issubset(
                    {1, 2, 3, 4, 5}
                )
            ):

                fold_col = col
                break


if fold_col is None:

    print("\nCould not detect fold column.")

    raise RuntimeError(
        "Could not find fold column in STEP 22 file."
    )


print(
    "Detected Fold column:",
    fold_col
)


# ============================================================
# STEP 6 - CHECK FOLD VALUES
# ============================================================

print("\nSTEP 6 - VALIDATING FOLDS")
print("=" * 75)

folds[fold_col] = pd.to_numeric(
    folds[fold_col],
    errors="coerce"
)


if folds[fold_col].isna().any():

    raise RuntimeError(
        "Missing or invalid fold values detected."
    )


folds[fold_col] = (
    folds[fold_col]
    .astype(int)
)


unique_folds = sorted(
    folds[fold_col].unique()
)


print(
    "Fold values:",
    unique_folds
)


if unique_folds != [1, 2, 3, 4, 5]:

    raise RuntimeError(
        "Expected exactly 5 folds numbered 1-5."
    )


print("\nFold distribution:")


for fold_number in unique_folds:

    count = np.sum(
        folds[fold_col] == fold_number
    )

    print(
        f"Fold {fold_number}: "
        f"{count} rows"
    )


# ============================================================
# STEP 7 - CHECK ROLE COLUMN
# ============================================================

print("\nSTEP 7 - VALIDATING PATIENT-LEVEL FOLD ASSIGNMENTS")
print("=" * 75)

ROLE_COLUMN = "Role"


if ROLE_COLUMN not in folds.columns:

    raise RuntimeError(
        "Role column not found in STEP 22 fold assignment file."
    )


print(
    "Role values:",
    sorted(
        folds[ROLE_COLUMN]
        .astype(str)
        .str.strip()
        .unique()
    )
)


# ============================================================
# STEP 8 - USE VALIDATION ASSIGNMENTS ONLY
#
# STEP 22 contains 5 rows per patient:
# one Validation row and four Training rows.
#
# For STEP 23 we need exactly one CV fold per patient.
# Therefore only Validation rows are used.
# ============================================================

print("\nSTEP 8 - EXTRACTING VALIDATION FOLD ASSIGNMENTS")
print("=" * 75)


folds[ROLE_COLUMN] = (
    folds[ROLE_COLUMN]
    .astype(str)
    .str.strip()
)


validation_folds = folds[
    folds[ROLE_COLUMN].str.lower() == "validation"
].copy()


print(
    "Validation assignment rows:",
    len(validation_folds)
)


print(
    "Validation patients:",
    validation_folds[fold_id_col].nunique()
)


if len(validation_folds) == 0:

    raise RuntimeError(
        "No Validation rows found in STEP 22 file."
    )


if validation_folds[fold_id_col].duplicated().any():

    duplicated_validation_ids = (
        validation_folds.loc[
            validation_folds[fold_id_col].duplicated(
                keep=False
            ),
            fold_id_col
        ]
        .unique()
    )

    raise RuntimeError(
        "\nA patient has more than one validation fold.\n"
        f"Duplicated validation IDs: "
        f"{len(duplicated_validation_ids)}"
    )


print(
    "PASS - Exactly one validation fold per patient."
)


# ============================================================
# STEP 9 - PATIENT MATCHING
# ============================================================

print("\nSTEP 9 - PATIENT MATCHING")
print("=" * 75)


data_ids = set(
    data[PATIENT_ID]
)


fold_ids = set(
    validation_folds[fold_id_col]
)


common_ids = (
    data_ids.intersection(fold_ids)
)


print(
    "Dataset patients:",
    len(data_ids)
)


print(
    "Fold patients:",
    len(fold_ids)
)


print(
    "Common patients:",
    len(common_ids)
)


print(
    "Dataset-only patients:",
    len(data_ids - fold_ids)
)


print(
    "Fold-only patients:",
    len(fold_ids - data_ids)
)


if len(common_ids) != len(data_ids):

    missing_ids = sorted(
        data_ids - fold_ids
    )

    print(
        "\nPatients without fold assignment:"
    )

    for patient_id in missing_ids[:20]:

        print("-", patient_id)

    raise RuntimeError(
        "Not all dataset patients have validation fold assignments."
    )


print(
    "PASS - All dataset patients have fold assignments."
)


# ============================================================
# STEP 10 - BUILD ONE PATIENT / ONE FOLD TABLE
# ============================================================

print("\nSTEP 10 - BUILDING PATIENT FOLD TABLE")
print("=" * 75)


fold_table = validation_folds[
    [
        fold_id_col,
        fold_col
    ]
].copy()


fold_table = fold_table.rename(
    columns={
        fold_id_col: PATIENT_ID,
        fold_col: "CV_Fold"
    }
)


# Make sure fold is integer

fold_table["CV_Fold"] = (
    fold_table["CV_Fold"]
    .astype(int)
)


print(
    "Patient fold table rows:",
    len(fold_table)
)


print(
    "Unique patients:",
    fold_table[PATIENT_ID].nunique()
)


if (
    len(fold_table)
    != fold_table[PATIENT_ID].nunique()
):

    raise RuntimeError(
        "Patient fold table contains duplicate patients."
    )


# ============================================================
# STEP 11 - MERGE FOLD ASSIGNMENTS
# ============================================================

print("\nSTEP 11 - MERGING FOLD ASSIGNMENTS")
print("=" * 75)


model_data = data.merge(
    fold_table,
    on=PATIENT_ID,
    how="inner",
    validate="one_to_one"
)


print(
    "Merged rows:",
    len(model_data)
)


if len(model_data) != len(data):

    raise RuntimeError(
        "Patient count changed during fold merge."
    )


print(
    "PASS - Patient-level fold merge successful."
)


# ============================================================
# STEP 12 - NUMERIC CONVERSION
# ============================================================

print("\nSTEP 12 - NUMERIC FEATURE PREPARATION")
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


before = len(model_data)


model_data = model_data.dropna(
    subset=FEATURES + [TARGET]
).copy()


after = len(model_data)


print(
    "Rows before numeric cleaning:",
    before
)


print(
    "Rows after numeric cleaning:",
    after
)


print(
    "Rows removed:",
    before - after
)


model_data[TARGET] = (
    model_data[TARGET]
    .astype(int)
)


if not set(
    model_data[TARGET].unique()
).issubset({0, 1}):

    raise RuntimeError(
        "Target contains values other than 0 and 1."
    )


# ============================================================
# STEP 13 - CLASS DISTRIBUTION
# ============================================================

print("\nSTEP 13 - CLASS DISTRIBUTION")
print("=" * 75)


class_counts = (
    model_data[TARGET]
    .value_counts()
    .sort_index()
)


total = len(model_data)


for cls in [0, 1]:

    count = int(
        class_counts.get(cls, 0)
    )

    percentage = (
        100 * count / total
        if total > 0
        else 0
    )

    print(
        f"Class {cls}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# STEP 14 - VERIFY FOLD STRATIFICATION
# ============================================================

print("\nSTEP 14 - VERIFYING FOLD STRATIFICATION")
print("=" * 75)


for fold_number in [1, 2, 3, 4, 5]:

    fold_data = model_data[
        model_data["CV_Fold"] == fold_number
    ]

    class_0 = int(
        (fold_data[TARGET] == 0).sum()
    )

    class_1 = int(
        (fold_data[TARGET] == 1).sum()
    )

    print(
        f"Fold {fold_number}: "
        f"{len(fold_data)} patients "
        f"(Class 0 = {class_0}, "
        f"Class 1 = {class_1})"
    )


# ============================================================
# STEP 15 - MINIMUM-DISTANCE CLASSIFIER
# ============================================================

print("\nSTEP 15 - MINIMUM-DISTANCE CLASSIFIER")
print("=" * 75)


print(
    "Classifier: Euclidean minimum-distance"
)


print(
    "Class prototypes are calculated from training patients only."
)


print(
    "Feature normalization is calculated from training patients only."
)


all_predictions = []

all_true = []

all_dist_class0 = []

all_dist_class1 = []

all_patient_ids = []

all_fold_numbers = []


fold_results = []


# ============================================================
# STEP 16 - 5-FOLD PATIENT-LEVEL CV
# ============================================================

print("\nSTEP 16 - 5-FOLD CROSS-VALIDATION")
print("=" * 75)


for fold_number in [1, 2, 3, 4, 5]:

    print("\n" + "-" * 70)
    print(f"FOLD {fold_number}")
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
        "Testing patients:",
        len(test)
    )


    # --------------------------------------------------------
    # TRAINING DATA
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
    # NORMALIZATION FROM TRAINING FOLD ONLY
    # --------------------------------------------------------

    train_mean = np.mean(
        X_train,
        axis=0
    )


    train_std = np.std(
        X_train,
        axis=0
    )


    # Avoid division by zero

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
            f"No class 0 training samples "
            f"in fold {fold_number}."
        )


    if len(class1_samples) == 0:

        raise RuntimeError(
            f"No class 1 training samples "
            f"in fold {fold_number}."
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
        "Class 0 training patients:",
        len(class0_samples)
    )


    print(
        "Class 1 training patients:",
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
    # MINIMUM DISTANCE DECISION
    # --------------------------------------------------------

    predictions = np.where(
        distances_0 <= distances_1,
        0,
        1
    )


    # --------------------------------------------------------
    # STORE RESULTS
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


    fold_results.append({

        "Fold":
            fold_number,

        "Training_Patients":
            len(train),

        "Testing_Patients":
            len(test),

        "Class0_Train":
            len(class0_samples),

        "Class1_Train":
            len(class1_samples),

        "Class0_Test":
            int((y_test == 0).sum()),

        "Class1_Test":
            int((y_test == 1).sum()),

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
        test[PATIENT_ID]
        .astype(str)
        .tolist()
    )


    all_fold_numbers.extend(
        [fold_number] * len(test)
    )


# ============================================================
# STEP 17 - OVERALL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 17 - OVERALL CROSS-VALIDATION RESULTS")
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
# STEP 18 - SAVE PATIENT PREDICTIONS
# ============================================================

print("\nSTEP 18 - SAVING PATIENT PREDICTIONS")
print("=" * 75)


prediction_df = pd.DataFrame({

    "PatientID":
        all_patient_ids,

    "Fold":
        all_fold_numbers,

    "True_Class":
        all_true,

    "Predicted_Class":
        all_predictions,

    "Distance_Class_0":
        all_dist_class0,

    "Distance_Class_1":
        all_dist_class1

})


# Verify one prediction per patient

if prediction_df["PatientID"].duplicated().any():

    raise RuntimeError(
        "Duplicate patient predictions detected."
    )


if len(prediction_df) != len(model_data):

    raise RuntimeError(
        "Number of predictions does not match "
        "number of evaluated patients."
    )


prediction_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_Patient_Predictions.csv"
)


prediction_df.to_csv(
    prediction_file,
    index=False
)


print("Saved:")
print(prediction_file)


# ============================================================
# STEP 19 - SAVE FOLD RESULTS
# ============================================================

print("\nSTEP 19 - SAVING FOLD RESULTS")
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


print("Saved:")
print(fold_results_file)


# ============================================================
# STEP 20 - SAVE OVERALL RESULTS
# ============================================================

print("\nSTEP 20 - SAVING OVERALL RESULTS")
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


print("Saved:")
print(overall_file)


# ============================================================
# STEP 21 - SAVE MODEL INFORMATION / REPORT
# ============================================================

print("\nSTEP 21 - SAVING METHODOLOGY REPORT")
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
        "=" * 70 + "\n\n"
    )


    f.write(
        "CLASSIFICATION TASK\n"
    )


    f.write(
        "Two-Year Survival Binary Classification\n\n"
    )


    f.write(
        "PATIENT IDENTIFIER\n"
    )


    f.write(
        "PatientID\n\n"
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
        "Random seed: 42\n\n"
    )


    f.write(
        "NORMALIZATION\n"
    )


    f.write(
        "Mean and standard deviation fitted on training "
        "patients only in each fold.\n\n"
    )


    f.write(
        "CLASS PROTOTYPES\n"
    )


    f.write(
        "Class prototypes calculated from training "
        "patients only in each fold.\n\n"
    )


    f.write(
        "FEATURE SELECTION\n"
    )


    f.write(
        "No feature selection performed in STEP 23.\n"
    )


    f.write(
        "Feature selection, if required, must be performed "
        "inside each training fold.\n\n"
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
        "RESULTS\n"
    )


    f.write(
        f"Patients evaluated: "
        f"{len(all_true)}\n"
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
        f"{overall_metrics['Precision']:.6f}\n"
    )


    f.write("\n")


    f.write(
        "CONFUSION MATRIX\n"
    )


    f.write(
        f"TN = "
        f"{overall_metrics['TN']}\n"
    )


    f.write(
        f"FP = "
        f"{overall_metrics['FP']}\n"
    )


    f.write(
        f"FN = "
        f"{overall_metrics['FN']}\n"
    )


    f.write(
        f"TP = "
        f"{overall_metrics['TP']}\n"
    )


    f.write("\n")


    f.write(
        "LEAKAGE PREVENTION\n"
    )


    f.write(
        "Training-fold normalization parameters were "
        "calculated using training patients only.\n"
    )


    f.write(
        "Class prototypes were calculated using training "
        "patients only.\n"
    )


    f.write(
        "Validation patients were not used for prototype "
        "calculation or normalization fitting.\n"
    )


    f.write(
        "Patient-level fold assignments were taken from "
        "STEP 22 validation rows only.\n"
    )


print("Saved:")
print(report_file)


# ============================================================
# STEP 22 - SAVE FEATURE INFORMATION
# ============================================================

print("\nSTEP 22 - SAVING FEATURE INFORMATION")
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
        "STEP 23 - FEATURES USED BY MINIMUM-DISTANCE CLASSIFIER\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for feature in FEATURES:

        f.write(
            feature + "\n"
        )


    f.write("\n")


    f.write(
        "No feature selection was performed in STEP 23.\n"
    )


print("Saved:")
print(features_file)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n")
print("=" * 75)
print("STEP 23 COMPLETE")
print("=" * 75)


print(
    "\nMinimum-Distance classifier completed successfully."
)


print(
    "\nPatients evaluated:",
    len(all_true)
)


print(
    "Accuracy:",
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


print(
    "\nOUTPUT DIRECTORY:"
)


print(
    OUTPUT_DIR
)


print(
    "\nOUTPUT FILES:"
)


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
    report_file
)


print(
    features_file
)


print("\n")


print("=" * 75)
print("READY FOR NEXT CLASSIFIER")
print("=" * 75)


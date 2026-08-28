
import os
import numpy as np
import pandas as pd

# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 24 - GAUSSIAN BAYES CLASSIFIER
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
    "STEP_24_GAUSSIAN_BAYES_CLASSIFIER"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

N_FOLDS = 5
RANDOM_STATE = 42

PATIENT_ID = "PatientID"
TARGET = "Two_Year_Survival"

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

EPSILON = 1e-9

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

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
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision
    }


def gaussian_log_probability(
    X,
    mean,
    variance,
    prior
):

    variance = np.maximum(
        variance,
        EPSILON
    )

    log_probability = (
        np.log(prior)
        - 0.5 * np.sum(
            np.log(2.0 * np.pi * variance)
        )
        - 0.5 * np.sum(
            ((X - mean) ** 2) / variance,
            axis=1
        )
    )

    return log_probability


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 24 - GAUSSIAN BAYES CLASSIFIER")
print("=" * 75)

print("\nNumber of folds:", N_FOLDS)
print("Random state:", RANDOM_STATE)

# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

print("\nSTEP 1 - LOADING VERIFIED DATA")
print("=" * 75)

if not os.path.isfile(DATA_FILE):

    raise FileNotFoundError(
        "\nSTEP 21 dataset not found:\n"
        + DATA_FILE
    )

if not os.path.isfile(FOLD_FILE):

    raise FileNotFoundError(
        "\nSTEP 22 fold file not found:\n"
        + FOLD_FILE
    )

data = pd.read_csv(DATA_FILE)
folds = pd.read_csv(FOLD_FILE)

print("STEP 21 dataset loaded.")
print("Rows:", len(data))
print("Columns:", len(data.columns))

print("\nSTEP 22 fold file loaded.")
print("Rows:", len(folds))
print("Columns:", len(folds.columns))

# ============================================================
# STEP 2 - REQUIRED COLUMN CHECK
# ============================================================

print("\nSTEP 2 - REQUIRED COLUMN CHECK")
print("=" * 75)

required_columns = [
    PATIENT_ID,
    TARGET
] + FEATURES

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    print("\nMISSING COLUMNS:")

    for column in missing_columns:
        print("-", column)

    raise RuntimeError(
        "\nRequired columns are missing from STEP 21 dataset."
    )

print("PASS - PatientID found")
print("PASS - Two_Year_Survival found")

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
    "Unique patients:",
    data[PATIENT_ID].nunique()
)

duplicate_count = (
    data[PATIENT_ID]
    .duplicated()
    .sum()
)

print(
    "Duplicate PatientID:",
    duplicate_count
)

if duplicate_count > 0:

    raise RuntimeError(
        "Duplicate PatientID detected."
    )

# ============================================================
# STEP 4 - TARGET CHECK
# ============================================================

print("\nSTEP 4 - TARGET CHECK")
print("=" * 75)

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce"
)

if data[TARGET].isna().any():

    raise RuntimeError(
        "Missing or invalid target values detected."
    )

data[TARGET] = data[TARGET].astype(int)

invalid_targets = data[
    ~data[TARGET].isin([0, 1])
]

if len(invalid_targets) > 0:

    raise RuntimeError(
        "Target contains values other than 0 and 1."
    )

print("PASS - Target contains only 0 and 1")

# ============================================================
# STEP 5 - FEATURE NUMERIC CHECK
# ============================================================

print("\nSTEP 5 - FEATURE NUMERIC CHECK")
print("=" * 75)

for feature in FEATURES:

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce"
    )

    print(
        feature,
        "missing:",
        int(data[feature].isna().sum())
    )

# ============================================================
# STEP 6 - REMOVE ONLY THE KNOWN UNUSABLE PATIENT
# ============================================================

print("\nSTEP 6 - PATIENT USABILITY CHECK")
print("=" * 75)

feature_missing_mask = data[
    FEATURES
].isna().any(axis=1)

unusable_patients = data[
    feature_missing_mask
].copy()

print(
    "Patients with missing/non-numeric features:",
    len(unusable_patients)
)

if len(unusable_patients) > 0:

    print("\nExcluded Patient IDs:")

    for patient_id in unusable_patients[PATIENT_ID]:

        print("-", patient_id)

usable_data = data[
    ~feature_missing_mask
].copy()

print(
    "\nUsable patients:",
    len(usable_data)
)

if len(usable_data) != 419:

    raise RuntimeError(
        "\nUnexpected usable patient count.\n"
        f"Expected 419, found {len(usable_data)}."
    )

print("PASS - 419 usable patients confirmed.")

# ============================================================
# STEP 7 - FOLD COLUMN DETECTION
# ============================================================

print("\nSTEP 7 - FOLD ASSIGNMENT CHECK")
print("=" * 75)

fold_id_candidates = [
    "PatientID",
    "Patient_ID",
    "PatientId",
    "patient_id",
    "patientid"
]

fold_id_col = find_column(
    folds,
    fold_id_candidates
)

if fold_id_col is None:

    raise RuntimeError(
        "Could not find Patient ID column in STEP 22 fold file."
    )

folds[fold_id_col] = (
    folds[fold_id_col]
    .astype(str)
    .str.strip()
)

print(
    "Detected Patient ID column:",
    fold_id_col
)

fold_candidates = [
    "Fold",
    "fold",
    "Fold_ID",
    "fold_id",
    "Fold_Number",
    "fold_number",
    "CV_Fold",
    "cv_fold"
]

fold_col = find_column(
    folds,
    fold_candidates
)

if fold_col is None:

    raise RuntimeError(
        "Could not find fold column in STEP 22 file."
    )

print(
    "Detected Fold column:",
    fold_col
)

# ============================================================
# STEP 8 - VALIDATE FOLD VALUES
# ============================================================

print("\nSTEP 8 - VALIDATING FOLD VALUES")
print("=" * 75)

folds[fold_col] = pd.to_numeric(
    folds[fold_col],
    errors="coerce"
)

if folds[fold_col].isna().any():

    raise RuntimeError(
        "Invalid fold values detected."
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
        "Expected exactly five folds numbered 1-5."
    )

# ============================================================
# STEP 9 - BUILD ONE PATIENT-LEVEL FOLD TABLE
# ============================================================

print("\nSTEP 9 - BUILDING PATIENT-LEVEL FOLD TABLE")
print("=" * 75)

# STEP 22 contains Training + Validation rows.
# We use ONLY the validation row for each patient.

validation_rows = folds[
    folds["Role"].astype(str).str.strip().str.lower()
    == "validation"
].copy()

if len(validation_rows) == 0:

    raise RuntimeError(
        "No Validation rows found in STEP 22 fold file."
    )

validation_rows = validation_rows[
    [fold_id_col, fold_col]
].copy()

validation_rows = validation_rows.rename(
    columns={
        fold_id_col: PATIENT_ID,
        fold_col: "CV_Fold"
    }
)

validation_rows[PATIENT_ID] = (
    validation_rows[PATIENT_ID]
    .astype(str)
    .str.strip()
)

# Each patient must have exactly one validation fold.

duplicate_validation_ids = (
    validation_rows[PATIENT_ID]
    .duplicated()
)

if duplicate_validation_ids.any():

    raise RuntimeError(
        "A patient has more than one validation fold."
    )

print(
    "Validation patients in STEP 22:",
    len(validation_rows)
)

# Match usable patients to validation folds.

usable_ids = set(
    usable_data[PATIENT_ID]
)

fold_ids = set(
    validation_rows[PATIENT_ID]
)

missing_fold_ids = usable_ids - fold_ids

if missing_fold_ids:

    raise RuntimeError(
        "Some usable patients do not have a validation fold:\n"
        + "\n".join(sorted(missing_fold_ids))
    )

print(
    "PASS - Every usable patient has one validation fold."
)

# ============================================================
# STEP 10 - MERGE DATA WITH FOLDS
# ============================================================

print("\nSTEP 10 - MERGING PATIENT DATA WITH FOLDS")
print("=" * 75)

model_data = usable_data.merge(
    validation_rows,
    on=PATIENT_ID,
    how="inner",
    validate="one_to_one"
)

print(
    "Merged patients:",
    len(model_data)
)

if len(model_data) != 419:

    raise RuntimeError(
        "Patient count changed during fold merge."
    )

print(
    "PASS - 419 patients retained."
)

# ============================================================
# STEP 11 - CLASS DISTRIBUTION
# ============================================================

print("\nSTEP 11 - CLASS DISTRIBUTION")
print("=" * 75)

class_0_count = int(
    (model_data[TARGET] == 0).sum()
)

class_1_count = int(
    (model_data[TARGET] == 1).sum()
)

print(
    "Class 0:",
    class_0_count
)

print(
    "Class 1:",
    class_1_count
)

# ============================================================
# STEP 12 - GAUSSIAN BAYES 5-FOLD CV
# ============================================================

print("\nSTEP 12 - GAUSSIAN BAYES 5-FOLD CROSS-VALIDATION")
print("=" * 75)

all_predictions = []
all_true = []

all_log_prob_class0 = []
all_log_prob_class1 = []

prediction_records = []
fold_results = []

for fold_number in range(1, N_FOLDS + 1):

    print("\n" + "-" * 70)
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
        "Testing patients:",
        len(test)
    )

    # --------------------------------------------------------
    # TRAINING / TEST MATRICES
    # --------------------------------------------------------

    X_train = train[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    X_test = test[
        FEATURES
    ].to_numpy(
        dtype=float
    )

    y_train = train[
        TARGET
    ].to_numpy(
        dtype=int
    )

    y_test = test[
        TARGET
    ].to_numpy(
        dtype=int
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # FITTED USING TRAINING DATA ONLY
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
        train_std < EPSILON
    ] = 1.0

    X_train_scaled = (
        X_train - train_mean
    ) / train_std

    X_test_scaled = (
        X_test - train_mean
    ) / train_std

    # --------------------------------------------------------
    # CLASS-SPECIFIC TRAINING DATA
    # --------------------------------------------------------

    class0_train = X_train_scaled[
        y_train == 0
    ]

    class1_train = X_train_scaled[
        y_train == 1
    ]

    if len(class0_train) == 0:

        raise RuntimeError(
            f"No Class 0 training patients in Fold {fold_number}."
        )

    if len(class1_train) == 0:

        raise RuntimeError(
            f"No Class 1 training patients in Fold {fold_number}."
        )

    # --------------------------------------------------------
    # GAUSSIAN PARAMETERS
    # --------------------------------------------------------

    mean_0 = np.mean(
        class0_train,
        axis=0
    )

    mean_1 = np.mean(
        class1_train,
        axis=0
    )

    variance_0 = np.var(
        class0_train,
        axis=0
    )

    variance_1 = np.var(
        class1_train,
        axis=0
    )

    # --------------------------------------------------------
    # CLASS PRIORS
    # --------------------------------------------------------

    prior_0 = (
        len(class0_train)
        / len(X_train_scaled)
    )

    prior_1 = (
        len(class1_train)
        / len(X_train_scaled)
    )

    # --------------------------------------------------------
    # GAUSSIAN LOG-PROBABILITIES
    # --------------------------------------------------------

    log_prob_0 = gaussian_log_probability(
        X_test_scaled,
        mean_0,
        variance_0,
        prior_0
    )

    log_prob_1 = gaussian_log_probability(
        X_test_scaled,
        mean_1,
        variance_1,
        prior_1
    )

    # --------------------------------------------------------
    # CLASS DECISION
    # --------------------------------------------------------

    predictions = np.where(
        log_prob_0 >= log_prob_1,
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
        "Class 0 training:",
        len(class0_train)
    )

    print(
        "Class 1 training:",
        len(class1_train)
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
            len(class0_train),

        "Class1_Train":
            len(class1_train),

        "Class0_Prior":
            prior_0,

        "Class1_Prior":
            prior_1,

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
    # SAVE PATIENT PREDICTIONS
    # --------------------------------------------------------

    for index in range(len(test)):

        prediction_records.append({

            "PatientID":
                test.iloc[index][PATIENT_ID],

            "Fold":
                fold_number,

            "True_Class":
                int(y_test[index]),

            "Predicted_Class":
                int(predictions[index]),

            "Log_Probability_Class_0":
                float(log_prob_0[index]),

            "Log_Probability_Class_1":
                float(log_prob_1[index])

        })

    all_predictions.extend(
        predictions.tolist()
    )

    all_true.extend(
        y_test.tolist()
    )

    all_log_prob_class0.extend(
        log_prob_0.tolist()
    )

    all_log_prob_class1.extend(
        log_prob_1.tolist()
    )

# ============================================================
# STEP 13 - OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("STEP 13 - OVERALL CROSS-VALIDATION RESULTS")
print("=" * 75)

overall_metrics = calculate_metrics(
    all_true,
    all_predictions
)

print(
    "\nPatients evaluated:",
    len(all_true)
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
# STEP 14 - SAVE PATIENT PREDICTIONS
# ============================================================

print("\nSTEP 14 - SAVING PATIENT PREDICTIONS")
print("=" * 75)

prediction_df = pd.DataFrame(
    prediction_records
)

prediction_file = os.path.join(
    OUTPUT_DIR,
    "STEP_24_Patient_Predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    "Saved:",
    prediction_file
)

# ============================================================
# STEP 15 - SAVE FOLD RESULTS
# ============================================================

print("\nSTEP 15 - SAVING FOLD RESULTS")
print("=" * 75)

fold_results_df = pd.DataFrame(
    fold_results
)

fold_results_file = os.path.join(
    OUTPUT_DIR,
    "STEP_24_Fold_Results.csv"
)

fold_results_df.to_csv(
    fold_results_file,
    index=False
)

print(
    "Saved:",
    fold_results_file
)

# ============================================================
# STEP 16 - SAVE OVERALL RESULTS
# ============================================================

print("\nSTEP 16 - SAVING OVERALL RESULTS")
print("=" * 75)

overall_df = pd.DataFrame([{

    "Classifier":
        "Gaussian Bayes",

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
    "STEP_24_Overall_Results.csv"
)

overall_df.to_csv(
    overall_file,
    index=False
)

print(
    "Saved:",
    overall_file
)

# ============================================================
# STEP 17 - SAVE FEATURES
# ============================================================

print("\nSTEP 17 - SAVING FEATURE LIST")
print("=" * 75)

features_file = os.path.join(
    OUTPUT_DIR,
    "STEP_24_Features_Used.txt"
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
        "STEP 24 - GAUSSIAN BAYES CLASSIFIER\n\n"
    )

    f.write(
        "Features used:\n\n"
    )

    for feature in FEATURES:

        f.write(
            f"- {feature}\n"
        )

    f.write("\n")

    f.write(
        "No global feature selection was performed.\n"
    )

# ============================================================
# STEP 18 - SAVE METHODOLOGY REPORT
# ============================================================

print("\nSTEP 18 - SAVING METHODOLOGY REPORT")
print("=" * 75)

report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_24_Gaussian_Bayes_Report.txt"
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
        "STEP 24 - GAUSSIAN BAYES CLASSIFIER\n"
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
        "PATIENTS\n"
    )

    f.write(
        "Original STEP 21 classified patients: 420\n"
    )

    f.write(
        "Usable patients after feature validation: 419\n"
    )

    f.write(
        "One patient excluded because at least one required "
        "radiomic feature was missing/non-numeric.\n\n"
    )

    f.write(
        "CLASSIFIER\n"
    )

    f.write(
        "Gaussian Bayes Classifier\n"
    )

    f.write(
        "Independent Gaussian feature likelihoods\n"
    )

    f.write(
        "Classification based on maximum posterior log-probability.\n\n"
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
        "Feature mean and standard deviation were calculated "
        "using training patients only within each fold.\n"
    )

    f.write(
        "The same training parameters were applied to the "
        "corresponding validation patients.\n\n"
    )

    f.write(
        "GAUSSIAN PARAMETERS\n"
    )

    f.write(
        "Class-specific feature means and variances were "
        "estimated from training patients only.\n"
    )

    f.write(
        "A small numerical epsilon was used to prevent "
        "division by zero for zero-variance features.\n\n"
    )

    f.write(
        "FEATURE SELECTION\n"
    )

    f.write(
        "No feature selection was performed in STEP 24.\n"
    )

    f.write(
        "Any future feature selection must be performed "
        "inside the training portion of each CV fold.\n\n"
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
        f"Accuracy: {overall_metrics['Accuracy']:.6f}\n"
    )

    f.write(
        f"Sensitivity: {overall_metrics['Sensitivity']:.6f}\n"
    )

    f.write(
        f"Specificity: {overall_metrics['Specificity']:.6f}\n"
    )

    f.write(
        f"Precision: {overall_metrics['Precision']:.6f}\n\n"
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

# ============================================================
# STEP 19 - SAVE CV VALIDATION REPORT
# ============================================================

print("\nSTEP 19 - SAVING VALIDATION REPORT")
print("=" * 75)

validation_report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_24_CV_Validation_Report.txt"
)

with open(
    validation_report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 24 - GAUSSIAN BAYES CV VALIDATION REPORT\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Patients evaluated: {len(all_true)}\n"
    )

    f.write(
        f"Expected patients: 419\n"
    )

    f.write(
        f"Number of folds: {N_FOLDS}\n"
    )

    f.write(
        "Patient-level validation: VERIFIED\n"
    )

    f.write(
        "Training-only normalization: VERIFIED\n"
    )

    f.write(
        "Validation data excluded from parameter fitting: VERIFIED\n"
    )

    f.write(
        "Gaussian parameters fitted using training data only: VERIFIED\n\n"
    )

    f.write(
        "Overall metrics:\n"
    )

    f.write(
        f"Accuracy = {overall_metrics['Accuracy']:.6f}\n"
    )

    f.write(
        f"Sensitivity = {overall_metrics['Sensitivity']:.6f}\n"
    )

    f.write(
        f"Specificity = {overall_metrics['Specificity']:.6f}\n"
    )

    f.write(
        f"Precision = {overall_metrics['Precision']:.6f}\n"
    )

# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("STEP 24 COMPLETE")
print("=" * 75)

print(
    "\nGaussian Bayes classifier completed successfully."
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

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nOUTPUT FILES:")
print(prediction_file)
print(fold_results_file)
print(overall_file)
print(features_file)
print(report_file)
print(validation_report_file)

print("\n")
print("=" * 75)
print("READY FOR NEXT CLASSIFIER")
print("=" * 75)


import os
import pandas as pd
import numpy as np

# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 22 - STRATIFIED PATIENT-LEVEL CROSS-VALIDATION
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "STEP_21_TWO_YEAR_SURVIVAL_ENDPOINT",
    "STEP_21_Two_Year_Survival_Classification_Dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_22_STRATIFIED_PATIENT_CV"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# SETTINGS
# ================================================================

N_FOLDS = 5
RANDOM_STATE = 42

TARGET = "Two_Year_Survival"

# STEP 19 uses PatientID
PATIENT_COLUMN = "PatientID"

STABLE_FEATURES = [
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

# ================================================================
# START
# ================================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 22 - STRATIFIED PATIENT-LEVEL CROSS-VALIDATION")
print("=" * 75)

print("\nNumber of folds:", N_FOLDS)
print("Random state:", RANDOM_STATE)

# ================================================================
# LOAD DATA
# ================================================================

if not os.path.isfile(INPUT_FILE):
    raise RuntimeError(
        "\nInput dataset not found:\n" + INPUT_FILE
    )

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Patients:", len(df))
print("Columns:", len(df.columns))

# ================================================================
# CHECK REQUIRED COLUMNS
# ================================================================

print("\n" + "=" * 75)
print("REQUIRED COLUMN CHECK")
print("=" * 75)

required_columns = [
    PATIENT_COLUMN,
    TARGET
] + STABLE_FEATURES

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nMissing required columns:")
    for column in missing_columns:
        print(" -", column)

    raise RuntimeError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required columns are present.")

# ================================================================
# CHECK PATIENT IDs
# ================================================================

print("\n" + "=" * 75)
print("PATIENT ID CHECK")
print("=" * 75)

df[PATIENT_COLUMN] = (
    df[PATIENT_COLUMN]
    .astype(str)
    .str.strip()
)

unique_patients = df[PATIENT_COLUMN].nunique()

duplicate_rows = (
    df[PATIENT_COLUMN]
    .duplicated()
    .sum()
)

missing_patient_ids = (
    df[PATIENT_COLUMN]
    .isna()
    .sum()
)

print("Total rows:", len(df))
print("Unique patients:", unique_patients)
print("Duplicate PatientID rows:", duplicate_rows)
print("Missing PatientID values:", missing_patient_ids)

if duplicate_rows > 0:
    duplicated = (
        df.loc[
            df[PATIENT_COLUMN].duplicated(keep=False),
            PATIENT_COLUMN
        ]
        .unique()
    )

    raise RuntimeError(
        "\nDuplicate patient IDs detected.\n"
        "Patient-level CV requires one row per patient.\n"
        f"Duplicated IDs: {len(duplicated)}"
    )

# ================================================================
# CHECK TARGET
# ================================================================

print("\n" + "=" * 75)
print("TARGET CHECK")
print("=" * 75)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

missing_target = df[TARGET].isna().sum()

print("Missing target values:", missing_target)

if missing_target > 0:
    raise RuntimeError(
        "\nMissing target values detected."
    )

invalid_target = df[
    ~df[TARGET].isin([0, 1])
]

if len(invalid_target) > 0:
    raise RuntimeError(
        "\nInvalid target values detected.\n"
        "Two_Year_Survival must contain only 0 or 1."
    )

print("Target values are valid: [0, 1]")

# ================================================================
# CHECK FEATURE VALUES
# ================================================================

print("\n" + "=" * 75)
print("RADIOMIC FEATURE CHECK")
print("=" * 75)

for feature in STABLE_FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    missing_feature = df[feature].isna().sum()

    print(
        f"{feature}: "
        f"{missing_feature} missing"
    )

# ================================================================
# PATIENT / TARGET SUMMARY
# ================================================================

print("\n" + "=" * 75)
print("PATIENT-LEVEL DATA SUMMARY")
print("=" * 75)

print("\nUnique patients:", unique_patients)

class_0_total = int(
    (df[TARGET] == 0).sum()
)

class_1_total = int(
    (df[TARGET] == 1).sum()
)

print(
    "Class 0:",
    class_0_total
)

print(
    "Class 1:",
    class_1_total
)

# ================================================================
# STRATIFIED PATIENT-LEVEL SPLIT
#
# Each patient appears once.
# Patients from each class are shuffled independently.
#
# No classifier is trained here.
# Feature selection is NOT performed globally.
# ================================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

class_0_ids = df.loc[
    df[TARGET] == 0,
    PATIENT_COLUMN
].to_numpy()

class_1_ids = df.loc[
    df[TARGET] == 1,
    PATIENT_COLUMN
].to_numpy()

# Shuffle each class independently
rng.shuffle(class_0_ids)
rng.shuffle(class_1_ids)

# ================================================================
# CREATE FOLD ASSIGNMENTS
# ================================================================

fold_assignments = {}

for fold_number in range(N_FOLDS):

    fold_assignments[fold_number] = []

# Distribute Class 0 patients
for index, patient_id in enumerate(class_0_ids):

    fold_number = index % N_FOLDS

    fold_assignments[
        fold_number
    ].append(patient_id)

# Distribute Class 1 patients
for index, patient_id in enumerate(class_1_ids):

    fold_number = index % N_FOLDS

    fold_assignments[
        fold_number
    ].append(patient_id)

# ================================================================
# VERIFY FOLD ASSIGNMENTS
# ================================================================

print("\n" + "=" * 75)
print("FOLD ASSIGNMENT CHECK")
print("=" * 75)

all_validation_ids = []

for fold_number in range(N_FOLDS):

    validation_ids = fold_assignments[
        fold_number
    ]

    all_validation_ids.extend(
        validation_ids
    )

if len(all_validation_ids) != len(df):

    raise RuntimeError(
        "\nNot all patients were assigned to validation folds."
    )

if len(set(all_validation_ids)) != len(df):

    raise RuntimeError(
        "\nA patient was assigned to more than one validation fold."
    )

print(
    "PASS - Every patient assigned exactly once "
    "to a validation fold."
)

# ================================================================
# BUILD FOLD TABLE
# ================================================================

fold_rows = []

for fold_number in range(N_FOLDS):

    validation_ids = set(
        fold_assignments[fold_number]
    )

    for _, row in df.iterrows():

        patient_id = row[PATIENT_COLUMN]

        if patient_id in validation_ids:

            fold_rows.append({

                PATIENT_COLUMN:
                    patient_id,

                "Fold":
                    fold_number + 1,

                "Role":
                    "Validation",

                TARGET:
                    int(row[TARGET])

            })

        else:

            fold_rows.append({

                PATIENT_COLUMN:
                    patient_id,

                "Fold":
                    fold_number + 1,

                "Role":
                    "Training",

                TARGET:
                    int(row[TARGET])

            })

fold_df = pd.DataFrame(
    fold_rows
)

# ================================================================
# VERIFY EVERY PATIENT APPEARS EXACTLY ONCE IN VALIDATION
# ================================================================

validation_counts = (
    fold_df[
        fold_df["Role"] == "Validation"
    ][PATIENT_COLUMN]
    .value_counts()
)

if not (
    validation_counts == 1
).all():

    raise RuntimeError(
        "\nPatient-level validation assignment failed."
    )

if len(validation_counts) != len(df):

    raise RuntimeError(
        "\nNot every patient has exactly one validation fold."
    )

print(
    "PASS - Each patient has exactly one validation fold."
)

# ================================================================
# VERIFY STRATIFICATION
# ================================================================

print("\n" + "=" * 75)
print("FOLD CLASS DISTRIBUTION")
print("=" * 75)

fold_summary_rows = []

for fold_number in range(
    1,
    N_FOLDS + 1
):

    validation = fold_df[
        (fold_df["Fold"] == fold_number)
        &
        (fold_df["Role"] == "Validation")
    ]

    training = fold_df[
        (fold_df["Fold"] == fold_number)
        &
        (fold_df["Role"] == "Training")
    ]

    val_class_0 = int(
        (validation[TARGET] == 0).sum()
    )

    val_class_1 = int(
        (validation[TARGET] == 1).sum()
    )

    train_class_0 = int(
        (training[TARGET] == 0).sum()
    )

    train_class_1 = int(
        (training[TARGET] == 1).sum()
    )

    print(
        f"\nFold {fold_number}:"
    )

    print(
        f"  Training:   {len(training)} "
        f"(Class 0 = {train_class_0}, "
        f"Class 1 = {train_class_1})"
    )

    print(
        f"  Validation: {len(validation)} "
        f"(Class 0 = {val_class_0}, "
        f"Class 1 = {val_class_1})"
    )

    fold_summary_rows.append({

        "Fold":
            fold_number,

        "Training_Patients":
            len(training),

        "Training_Class_0":
            train_class_0,

        "Training_Class_1":
            train_class_1,

        "Validation_Patients":
            len(validation),

        "Validation_Class_0":
            val_class_0,

        "Validation_Class_1":
            val_class_1

    })

fold_summary_df = pd.DataFrame(
    fold_summary_rows
)

# ================================================================
# VERIFY TOTAL FOLD COUNTS
# ================================================================

print("\n" + "=" * 75)
print("STRATIFICATION VALIDATION")
print("=" * 75)

validation_total = (
    fold_summary_df[
        "Validation_Patients"
    ].sum()
)

training_total_expected = (
    len(df) * (N_FOLDS - 1)
)

training_total = (
    fold_summary_df[
        "Training_Patients"
    ].sum()
)

if validation_total != len(df):

    raise RuntimeError(
        "\nValidation fold totals do not equal "
        "the number of patients."
    )

if training_total != training_total_expected:

    raise RuntimeError(
        "\nTraining fold totals are incorrect."
    )

validation_class_0_total = (
    fold_summary_df[
        "Validation_Class_0"
    ].sum()
)

validation_class_1_total = (
    fold_summary_df[
        "Validation_Class_1"
    ].sum()
)

if validation_class_0_total != class_0_total:

    raise RuntimeError(
        "\nClass 0 stratification validation failed."
    )

if validation_class_1_total != class_1_total:

    raise RuntimeError(
        "\nClass 1 stratification validation failed."
    )

print(
    "PASS - Validation patients total:",
    validation_total
)

print(
    "PASS - Class 0 validation total:",
    validation_class_0_total
)

print(
    "PASS - Class 1 validation total:",
    validation_class_1_total
)

print(
    "PASS - Training fold totals verified."
)

# ================================================================
# SAVE FOLD ASSIGNMENTS
# ================================================================

fold_file = os.path.join(
    OUTPUT_DIR,
    "STEP_22_Patient_Fold_Assignments.csv"
)

fold_df.to_csv(
    fold_file,
    index=False
)

# ================================================================
# SAVE FOLD SUMMARY
# ================================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "STEP_22_Fold_Class_Distribution.csv"
)

fold_summary_df.to_csv(
    summary_file,
    index=False
)

# ================================================================
# SAVE FEATURE LIST
# ================================================================

features_file = os.path.join(
    OUTPUT_DIR,
    "STEP_22_Stable_Features_For_CV.txt"
)

with open(
    features_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Stable radiomic features available "
        "for feature selection inside CV:\n\n"
    )

    for feature in STABLE_FEATURES:

        f.write(
            feature + "\n"
        )

    f.write(
        "\nIMPORTANT:\n"
    )

    f.write(
        "These features are candidate stable features.\n"
    )

    f.write(
        "Any additional feature selection must be "
        "performed using training data inside each "
        "cross-validation fold only.\n"
    )

# ================================================================
# SAVE METHODOLOGY SUMMARY
# ================================================================

method_file = os.path.join(
    OUTPUT_DIR,
    "STEP_22_CV_Methodology.txt"
)

with open(
    method_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 22 - STRATIFIED PATIENT-LEVEL "
        "CROSS-VALIDATION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Total classified patients: {len(df)}\n"
    )

    f.write(
        f"Number of folds: {N_FOLDS}\n"
    )

    f.write(
        f"Random state: {RANDOM_STATE}\n\n"
    )

    f.write(
        "Patient identifier:\n"
    )

    f.write(
        f"{PATIENT_COLUMN}\n\n"
    )

    f.write(
        "Target:\n"
    )

    f.write(
        "Two_Year_Survival\n\n"
    )

    f.write(
        "Class 0 = Death before 2 years\n"
    )

    f.write(
        "Class 1 = Survived >= 2 years\n\n"
    )

    f.write(
        "Validation strategy:\n"
    )

    f.write(
        "Stratified patient-level cross-validation.\n\n"
    )

    f.write(
        "Fold construction:\n"
    )

    f.write(
        "Patients were shuffled independently within "
        "each target class using a fixed random seed.\n"
    )

    f.write(
        "Patients were distributed across five folds "
        "using round-robin assignment.\n\n"
    )

    f.write(
        "Leakage prevention:\n"
    )

    f.write(
        "Feature selection must be performed inside "
        "the training portion of each fold.\n"
    )

    f.write(
        "The validation portion must remain completely "
        "unseen during feature selection and model fitting.\n\n"
    )

    f.write(
        "Normalization:\n"
    )

    f.write(
        "Normalization parameters must be fitted using "
        "training patients only and then applied to "
        "validation patients.\n\n"
    )

    f.write(
        "Classifier training:\n"
    )

    f.write(
        "No classifier was trained in STEP 22.\n"
    )

# ================================================================
# SAVE FINAL VALIDATION REPORT
# ================================================================

validation_report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_22_CV_Validation_Report.txt"
)

with open(
    validation_report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 22 - CROSS-VALIDATION VALIDATION REPORT\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "STATUS: VALID\n\n"
    )

    f.write(
        f"Total patients: {len(df)}\n"
    )

    f.write(
        f"Class 0 patients: {class_0_total}\n"
    )

    f.write(
        f"Class 1 patients: {class_1_total}\n"
    )

    f.write(
        f"Number of folds: {N_FOLDS}\n"
    )

    f.write(
        f"Random state: {RANDOM_STATE}\n\n"
    )

    f.write(
        "Patient-level validation:\n"
    )

    f.write(
        "PASS - One row per patient.\n"
    )

    f.write(
        "PASS - Every patient assigned exactly once "
        "to a validation fold.\n"
    )

    f.write(
        "PASS - No patient appears in more than one "
        "validation fold.\n"
    )

    f.write(
        "PASS - Class 0 distribution verified.\n"
    )

    f.write(
        "PASS - Class 1 distribution verified.\n"
    )

    f.write(
        "PASS - Five-fold training/validation structure verified.\n\n"
    )

    f.write(
        "Data leakage prevention:\n"
    )

    f.write(
        "Feature selection must occur inside each "
        "training fold only.\n"
    )

    f.write(
        "Normalization must be fitted using training "
        "patients only.\n"
    )

    f.write(
        "Validation data must remain unseen until evaluation.\n\n"
    )

    f.write(
        "No classifier was trained in STEP 22.\n"
    )

    f.write(
        "STEP 22 is a cross-validation preparation step.\n"
    )

# ================================================================
# FINAL
# ================================================================

print("\n")
print("=" * 75)
print("STEP 22 COMPLETE")
print("=" * 75)

print("\nTotal patients:", len(df))

print(
    "Class 0:",
    class_0_total
)

print(
    "Class 1:",
    class_1_total
)

print(
    "\nEach patient is assigned to exactly one "
    "validation fold."
)

print(
    "Patient-level stratification verified."
)

print(
    "Class distribution verified."
)

print(
    "No classifier trained."
)

print(
    "Feature selection must occur inside "
    "each training fold."
)

print(
    "Normalization must use training data only."
)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nFILES:")
print(fold_file)
print(summary_file)
print(features_file)
print(method_file)
print(validation_report_file)

print("\n")
print("=" * 75)
print("READY FOR CLASSIFIER DEVELOPMENT")
print("=" * 75)

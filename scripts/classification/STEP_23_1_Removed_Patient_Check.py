
import os
import pandas as pd
import numpy as np

# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 23.1 - REMOVED PATIENT CHECK
# ============================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

DATA_FILE = os.path.join(
    BASE_DIR,
    "STEP_21_TWO_YEAR_SURVIVAL_ENDPOINT",
    "STEP_21_Two_Year_Survival_Classification_Dataset.csv"
)

STEP23_PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "STEP_23_MINIMUM_DISTANCE_CLASSIFIER",
    "STEP_23_Patient_Predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_23_MINIMUM_DISTANCE_CLASSIFIER"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

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

# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 23.1 - REMOVED PATIENT CHECK")
print("=" * 75)

# ============================================================
# STEP 1 - LOAD STEP 21
# ============================================================

print("\nSTEP 1 - LOADING STEP 21 DATA")
print("=" * 75)

if not os.path.isfile(DATA_FILE):
    raise FileNotFoundError(
        "\nSTEP 21 dataset not found:\n"
        + DATA_FILE
    )

df = pd.read_csv(DATA_FILE)

print("STEP 21 dataset loaded successfully.")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

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
    if column not in df.columns
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

df[PATIENT_ID] = (
    df[PATIENT_ID]
    .astype(str)
    .str.strip()
)

unique_patients = df[PATIENT_ID].nunique()

print("Total rows:", len(df))
print("Unique patients:", unique_patients)

duplicate_count = (
    df[PATIENT_ID]
    .duplicated()
    .sum()
)

print("Duplicate PatientID rows:", duplicate_count)

if duplicate_count > 0:

    duplicated_ids = (
        df.loc[
            df[PATIENT_ID].duplicated(keep=False),
            PATIENT_ID
        ]
        .unique()
    )

    print("\nDuplicated Patient IDs:")

    for patient_id in duplicated_ids:
        print("-", patient_id)

    raise RuntimeError(
        "\nDuplicate PatientID detected."
    )

# ============================================================
# STEP 4 - CHECK TARGET
# ============================================================

print("\nSTEP 4 - TARGET CHECK")
print("=" * 75)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

print(
    "Missing / invalid target values:",
    int(df[TARGET].isna().sum())
)

print(
    "Class 0:",
    int((df[TARGET] == 0).sum())
)

print(
    "Class 1:",
    int((df[TARGET] == 1).sum())
)

invalid_target_mask = ~df[TARGET].isin([0, 1])

if invalid_target_mask.any():

    invalid_targets = df.loc[
        invalid_target_mask,
        [PATIENT_ID, TARGET]
    ]

    print("\nInvalid target patients:")

    print(
        invalid_targets.to_string(
            index=False
        )
    )

# ============================================================
# STEP 5 - CHECK ALL 9 FEATURES
# ============================================================

print("\nSTEP 5 - CHECKING THE 9 CLASSIFICATION FEATURES")
print("=" * 75)

numeric_columns = []

for feature in FEATURES:

    numeric_column = feature + "_numeric"

    numeric_columns.append(numeric_column)

    df[numeric_column] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    missing_count = int(
        df[numeric_column].isna().sum()
    )

    print(
        f"{feature}: "
        f"{missing_count} missing/non-numeric"
    )

# ============================================================
# STEP 6 - IDENTIFY REMOVED PATIENTS
# ============================================================

print("\nSTEP 6 - IDENTIFYING REMOVED PATIENTS")
print("=" * 75)

feature_missing_mask = (
    df[numeric_columns]
    .isna()
    .any(axis=1)
)

target_invalid_mask = (
    ~df[TARGET].isin([0, 1])
)

removed_mask = (
    feature_missing_mask |
    target_invalid_mask
)

removed = df.loc[
    removed_mask
].copy()

usable = df.loc[
    ~removed_mask
].copy()

print("Total STEP 21 patients:", len(df))
print("Usable patients:", len(usable))
print("Removed / unusable patients:", len(removed))

# ============================================================
# STEP 7 - REMOVED PATIENT DETAILS
# ============================================================

print("\nSTEP 7 - REMOVED PATIENT DETAILS")
print("=" * 75)

if len(removed) == 0:

    print(
        "NO PATIENTS WERE REMOVED "
        "BASED ON FEATURE/TARGET VALIDITY."
    )

else:

    for _, row in removed.iterrows():

        print("\n---------------------------------------")

        print(
            "PatientID:",
            row[PATIENT_ID]
        )

        print(
            "Two_Year_Survival:",
            row[TARGET]
        )

        missing_features = []

        for feature in FEATURES:

            numeric_value = row[
                feature + "_numeric"
            ]

            if pd.isna(numeric_value):

                missing_features.append(
                    feature
                )

        if missing_features:

            print("Missing / invalid features:")

            for feature in missing_features:

                print(
                    "  -",
                    feature
                )

        else:

            print(
                "No missing radiomic features."
            )

        if pd.isna(row[TARGET]):

            print(
                "Invalid / missing target."
            )

        elif row[TARGET] not in [0, 1]:

            print(
                "Invalid target value:",
                row[TARGET]
            )

# ============================================================
# STEP 8 - CHECK STEP 23 PREDICTIONS
# ============================================================

print("\nSTEP 8 - CHECKING STEP 23 PREDICTIONS")
print("=" * 75)

if not os.path.isfile(STEP23_PREDICTIONS_FILE):

    print(
        "STEP 23 prediction file not found."
    )

    predictions_df = None

else:

    predictions_df = pd.read_csv(
        STEP23_PREDICTIONS_FILE
    )

    print(
        "STEP 23 predictions loaded."
    )

    print(
        "STEP 23 prediction rows:",
        len(predictions_df)
    )

    prediction_id_candidates = [
        "PatientID",
        "Patient_ID"
    ]

    prediction_id_column = None

    for candidate in prediction_id_candidates:

        if candidate in predictions_df.columns:

            prediction_id_column = candidate
            break

    if prediction_id_column is None:

        raise RuntimeError(
            "\nCould not identify Patient ID "
            "column in STEP 23 predictions."
        )

    predictions_df[prediction_id_column] = (
        predictions_df[prediction_id_column]
        .astype(str)
        .str.strip()
    )

    step23_patient_ids = set(
        predictions_df[
            prediction_id_column
        ]
    )

    removed_patient_ids = set(
        removed[PATIENT_ID]
    )

    overlap = (
        step23_patient_ids
        .intersection(
            removed_patient_ids
        )
    )

    print(
        "Removed patients appearing in STEP 23:",
        len(overlap)
    )

    if len(overlap) > 0:

        print(
            "\nWARNING - REMOVED PATIENTS "
            "FOUND IN STEP 23:"
        )

        for patient_id in sorted(overlap):

            print(
                "-",
                patient_id
            )

    else:

        print(
            "PASS - No removed patient "
            "appears in STEP 23 predictions."
        )

# ============================================================
# STEP 9 - SAVE REMOVED PATIENT REPORT
# ============================================================

print("\nSTEP 9 - SAVING REMOVED PATIENT REPORT")
print("=" * 75)

removed_records = []

for _, row in removed.iterrows():

    missing_features = []

    for feature in FEATURES:

        if pd.isna(
            row[feature + "_numeric"]
        ):

            missing_features.append(
                feature
            )

    target_status = "Valid"

    if pd.isna(row[TARGET]):

        target_status = "Missing / Invalid"

    elif row[TARGET] not in [0, 1]:

        target_status = "Invalid"

    removed_records.append({

        "PatientID":
            row[PATIENT_ID],

        "Two_Year_Survival":
            row[TARGET],

        "Missing_or_Invalid_Features":
            "; ".join(missing_features),

        "Target_Status":
            target_status

    })

removed_report_df = pd.DataFrame(
    removed_records
)

removed_report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_1_Removed_Patient_Check.csv"
)

removed_report_df.to_csv(
    removed_report_file,
    index=False
)

print(
    "Saved:",
    removed_report_file
)

# ============================================================
# STEP 10 - SAVE VALIDATION REPORT
# ============================================================

validation_report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_23_1_Removed_Patient_Validation_Report.txt"
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
        "STEP 23.1 - REMOVED PATIENT CHECK\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"STEP 21 total patients: {len(df)}\n"
    )

    f.write(
        f"Usable patients: {len(usable)}\n"
    )

    f.write(
        f"Removed / unusable patients: {len(removed)}\n\n"
    )

    f.write(
        "PATIENT IDENTIFIER\n"
    )

    f.write(
        "PatientID\n\n"
    )

    f.write(
        "TARGET\n"
    )

    f.write(
        "Two_Year_Survival\n\n"
    )

    f.write(
        "CLASSIFICATION FEATURES\n"
    )

    for feature in FEATURES:

        f.write(
            f"- {feature}\n"
        )

    f.write("\n")

    f.write(
        "REMOVED PATIENTS\n"
    )

    if len(removed) == 0:

        f.write(
            "None based on feature/target validity.\n"
        )

    else:

        for _, row in removed.iterrows():

            missing_features = []

            for feature in FEATURES:

                if pd.isna(
                    row[feature + "_numeric"]
                ):

                    missing_features.append(
                        feature
                    )

            f.write(
                f"\nPatientID: "
                f"{row[PATIENT_ID]}\n"
            )

            f.write(
                f"Two_Year_Survival: "
                f"{row[TARGET]}\n"
            )

            f.write(
                "Missing / invalid features: "
                + (
                    ", ".join(missing_features)
                    if missing_features
                    else "None"
                )
                + "\n"
            )

    f.write("\n")

    if predictions_df is not None:

        f.write(
            "STEP 23 PREDICTION CHECK\n"
        )

        f.write(
            f"STEP 23 predicted patients: "
            f"{len(step23_patient_ids)}\n"
        )

        f.write(
            f"Removed patients found in STEP 23: "
            f"{len(overlap)}\n"
        )

        if len(overlap) == 0:

            f.write(
                "PASS - No removed patient "
                "was used by STEP 23.\n"
            )

        else:

            f.write(
                "WARNING - One or more removed "
                "patients were found in STEP 23.\n"
            )

    f.write("\n")

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "This step is a validation/check step only.\n"
    )

    f.write(
        "No classifier was trained in STEP 23.1.\n"
    )

# ============================================================
# STEP 11 - FINAL CONSISTENCY CHECK
# ============================================================

print("\nSTEP 11 - FINAL CONSISTENCY CHECK")
print("=" * 75)

print(
    "STEP 21 patients:",
    len(df)
)

print(
    "Usable patients:",
    len(usable)
)

print(
    "Removed / unusable:",
    len(removed)
)

if predictions_df is not None:

    print(
        "STEP 23 predicted patients:",
        len(step23_patient_ids)
    )

    if len(overlap) == 0:

        print(
            "\nPASS - STEP 23 does not contain "
            "any removed patient."
        )

    else:

        print(
            "\nFAIL - Removed patient(s) "
            "appear in STEP 23."
        )

# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 75)
print("STEP 23.1 COMPLETE")
print("=" * 75)

print(
    "\nTotal STEP 21 patients:",
    len(df)
)

print(
    "Usable patients:",
    len(usable)
)

print(
    "Removed / unusable patients:",
    len(removed)
)

print("\nOUTPUT FILES:")

print(
    removed_report_file
)

print(
    validation_report_file
)

print("\n")
print("=" * 75)
print("DO NOT START BAYES YET")
print("=" * 75)

print(
    "\nSend me the COMPLETE CMD output."
)



import os
import pandas as pd
import numpy as np

# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 21 - TWO-YEAR SURVIVAL BINARY ENDPOINT
# ============================================================
#
# PURPOSE:
# Create the binary classification endpoint required for
# Phase 3:
#
#   Class 0 = Death before 2 years
#   Class 1 = Survival for at least 2 years
#
# The endpoint is constructed from:
#   Survival.time
#   deadstatus.event
#
# IMPORTANT:
# Patients who are alive but have less than 2 years of
# follow-up are censored and are NOT assigned a class.
#
# This avoids incorrectly calling a patient who survived
# 300 days "dead before 2 years".
#
# ============================================================


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

STEP19_DIR = os.path.join(
    BASE_DIR,
    "STEP_19_MERGED_RADIOMICS_CLINICAL"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_21_TWO_YEAR_SURVIVAL_ENDPOINT"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# INPUT FILE
# ============================================================

INPUT_FILE = os.path.join(
    STEP19_DIR,
    "STEP_19_Final_Radiomics_Clinical_Dataset.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_DATASET = os.path.join(
    OUTPUT_DIR,
    "STEP_21_Two_Year_Survival_Classification_Dataset.csv"
)

CENSORED_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_21_Censored_Patients.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_21_Target_Summary.txt"
)

DISTRIBUTION_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_21_Class_Distribution.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

TWO_YEAR_DAYS = 730
TIENT_COL = "Patient_ID"
PATIENT_COL = "PatientID"
SURVIVAL_COL = "Survival.time"
DEATH_COL = "deadstatus.event"

TARGET_COL = "Two_Year_Survival"


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 21 - TWO-YEAR SURVIVAL BINARY ENDPOINT")
print("=" * 75)


# ============================================================
# STEP 1 - LOAD STEP 19
# ============================================================

print("\n" + "=" * 75)
print("STEP 1 - LOADING STEP 19 DATASET")
print("=" * 75)

if not os.path.exists(INPUT_FILE):

    raise RuntimeError(
        f"\nSTEP 19 input file not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)

print(f"Dataset loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# STEP 2 - REQUIRED COLUMN CHECK
# ============================================================

print("\n" + "=" * 75)
print("STEP 2 - REQUIRED COLUMN CHECK")
print("=" * 75)

required_columns = [
    PATIENT_COL,
    SURVIVAL_COL,
    DEATH_COL
]

for column in required_columns:

    if column in df.columns:

        print(f"FOUND - {column}")

    else:

        print(f"MISSING - {column}")

        raise RuntimeError(
            f"Required column missing: {column}"
        )


# ============================================================
# STEP 3 - PATIENT ID CHECK
# ============================================================

print("\n" + "=" * 75)
print("STEP 3 - PATIENT ID CHECK")
print("=" * 75)

df[PATIENT_COL] = (
    df[PATIENT_COL]
    .astype(str)
    .str.strip()
)

unique_patients = df[PATIENT_COL].nunique()

duplicate_rows = df[PATIENT_COL].duplicated().sum()

print(f"Total rows: {len(df)}")
print(f"Unique patients: {unique_patients}")
print(f"Duplicate Patient_ID rows: {duplicate_rows}")


if duplicate_rows > 0:

    raise RuntimeError(
        "Duplicate Patient_ID values detected. "
        "STEP 21 requires one row per patient."
    )


# ============================================================
# STEP 4 - CONVERT SURVIVAL AND DEATH STATUS
# ============================================================

print("\n" + "=" * 75)
print("STEP 4 - CONVERTING SURVIVAL VARIABLES")
print("=" * 75)

df[SURVIVAL_COL] = pd.to_numeric(
    df[SURVIVAL_COL],
    errors="coerce"
)

df[DEATH_COL] = pd.to_numeric(
    df[DEATH_COL],
    errors="coerce"
)


print(
    f"{SURVIVAL_COL} valid: "
    f"{df[SURVIVAL_COL].notna().sum()}"
)

print(
    f"{SURVIVAL_COL} missing: "
    f"{df[SURVIVAL_COL].isna().sum()}"
)

print(
    f"{DEATH_COL} valid: "
    f"{df[DEATH_COL].notna().sum()}"
)

print(
    f"{DEATH_COL} missing: "
    f"{df[DEATH_COL].isna().sum()}"
)


# ============================================================
# STEP 5 - DEATH STATUS CHECK
# ============================================================

print("\n" + "=" * 75)
print("STEP 5 - DEATH STATUS CHECK")
print("=" * 75)

print(
    "\nUnique deadstatus.event values:"
)

print(
    df[DEATH_COL]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# VALIDITY CHECK
# ============================================================

invalid_death_status = df[
    ~df[DEATH_COL].isin([0, 1])
    & df[DEATH_COL].notna()
]

if len(invalid_death_status) > 0:

    print(
        "\nWARNING:"
    )

    print(
        f"Found {len(invalid_death_status)} "
        "patients with invalid death-status values."
    )

    raise RuntimeError(
        "deadstatus.event must contain only 0 or 1."
    )


# ============================================================
# STEP 6 - CREATE TWO-YEAR ENDPOINT
# ============================================================

print("\n" + "=" * 75)
print("STEP 6 - CREATING TWO-YEAR SURVIVAL ENDPOINT")
print("=" * 75)

print(
    f"Two-year threshold: {TWO_YEAR_DAYS} days"
)

print(
    "\nEndpoint definition:"
)

print(
    "Class 0 = Death before 730 days"
)

print(
    "Class 1 = Alive at 730 days OR survival time >= 730 days"
)

print(
    "Censored = Alive with less than 730 days follow-up"
)


# ------------------------------------------------------------
# Initialize target
#
# NaN means:
#   endpoint cannot be determined reliably
# ------------------------------------------------------------

df[TARGET_COL] = np.nan


# ------------------------------------------------------------
# CLASS 0
#
# Death occurred before 2 years.
# ------------------------------------------------------------

class_0_mask = (
    (df[DEATH_COL] == 1)
    &
    (df[SURVIVAL_COL] < TWO_YEAR_DAYS)
)


df.loc[
    class_0_mask,
    TARGET_COL
] = 0


# ------------------------------------------------------------
# CLASS 1
#
# Patient survived at least 2 years.
#
# This includes:
#
#   death after 730 days
#   alive with >=730 days follow-up
#
# ------------------------------------------------------------

class_1_mask = (
    (df[SURVIVAL_COL] >= TWO_YEAR_DAYS)
)


df.loc[
    class_1_mask,
    TARGET_COL
] = 1


# ============================================================
# STEP 7 - IDENTIFY CENSORED / UNDETERMINED PATIENTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 7 - CENSORED / UNDETERMINED CASES")
print("=" * 75)


censored_mask = (
    df[TARGET_COL].isna()
)


censored_df = df.loc[
    censored_mask
].copy()


print(
    f"Censored / undetermined patients: "
    f"{len(censored_df)}"
)


if len(censored_df) > 0:

    print(
        "\nThese patients are not assigned a binary class "
        "because they do not have enough follow-up to determine "
        "two-year survival."
    )


# ============================================================
# STEP 8 - TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("STEP 8 - TWO-YEAR SURVIVAL CLASS DISTRIBUTION")
print("=" * 75)


class_0_count = int(
    (df[TARGET_COL] == 0).sum()
)

class_1_count = int(
    (df[TARGET_COL] == 1).sum()
)

censored_count = int(
    df[TARGET_COL].isna().sum()
)

total_count = len(df)

classified_count = (
    class_0_count
    +
    class_1_count
)


print(
    f"Total patients: {total_count}"
)

print(
    f"Class 0 - Death before 2 years: "
    f"{class_0_count}"
)

print(
    f"Class 1 - Survival >= 2 years: "
    f"{class_1_count}"
)

print(
    f"Censored / undetermined: "
    f"{censored_count}"
)

print(
    f"Patients with binary endpoint: "
    f"{classified_count}"
)


if classified_count > 0:

    class_0_percent = (
        class_0_count
        /
        classified_count
        *
        100
    )

    class_1_percent = (
        class_1_count
        /
        classified_count
        *
        100
    )

else:

    class_0_percent = 0
    class_1_percent = 0


print(
    f"\nClass 0 percentage among classified patients: "
    f"{class_0_percent:.2f}%"
)

print(
    f"Class 1 percentage among classified patients: "
    f"{class_1_percent:.2f}%"
)


# ============================================================
# STEP 9 - REMOVE UNDETERMINED PATIENTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 9 - BUILDING FINAL CLASSIFICATION DATASET")
print("=" * 75)


classification_df = df[
    df[TARGET_COL].notna()
].copy()


classification_df[TARGET_COL] = (
    classification_df[TARGET_COL]
    .astype(int)
)


print(
    f"Final classification patients: "
    f"{len(classification_df)}"
)

print(
    f"Final columns: "
    f"{len(classification_df.columns)}"
)


# ============================================================
# STEP 10 - FINAL TARGET VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("STEP 10 - FINAL TARGET VALIDATION")
print("=" * 75)


target_values = sorted(
    classification_df[TARGET_COL]
    .unique()
)


print(
    f"Target values: {target_values}"
)


if target_values != [0, 1]:

    raise RuntimeError(
        "Final target is not binary [0, 1]."
    )


if classification_df[PATIENT_COL].duplicated().any():

    raise RuntimeError(
        "Duplicate patients detected in final dataset."
    )


print(
    "PASS - Binary target confirmed."
)

print(
    "PASS - One row per patient confirmed."
)


# ============================================================
# STEP 11 - SAVE CLASSIFIED DATASET
# ============================================================

print("\n" + "=" * 75)
print("STEP 11 - SAVING CLASSIFICATION DATASET")
print("=" * 75)


classification_df.to_csv(
    OUTPUT_DATASET,
    index=False
)


print(
    f"Saved:\n{OUTPUT_DATASET}"
)


# ============================================================
# STEP 12 - SAVE CENSORED PATIENTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 12 - SAVING CENSORED PATIENTS")
print("=" * 75)


censored_df.to_csv(
    CENSORED_FILE,
    index=False
)


print(
    f"Saved:\n{CENSORED_FILE}"
)


# ============================================================
# STEP 13 - SAVE CLASS DISTRIBUTION
# ============================================================

distribution_df = pd.DataFrame(
    {
        "Class": [
            0,
            1
        ],

        "Description": [
            "Death before 2 years",
            "Survival >= 2 years"
        ],

        "Patients": [
            class_0_count,
            class_1_count
        ],

        "Percentage_of_Classified": [
            class_0_percent,
            class_1_percent
        ]
    }
)


distribution_df.to_csv(
    DISTRIBUTION_FILE,
    index=False
)


print(
    f"Saved:\n{DISTRIBUTION_FILE}"
)


# ============================================================
# STEP 14 - SAVE SUMMARY REPORT
# ============================================================

summary_text = f"""
PROJECT 7 - RADIOMICS
STEP 21 - TWO-YEAR SURVIVAL BINARY ENDPOINT
===========================================================================

INPUT DATASET
------------------------------------------------------------------------
Input:
{INPUT_FILE}

Total patients:
{total_count}

Unique patients:
{unique_patients}

Duplicate Patient_ID rows:
{duplicate_rows}


TWO-YEAR ENDPOINT DEFINITION
------------------------------------------------------------------------

Threshold:
{TWO_YEAR_DAYS} days

Class 0:
Death before 2 years
(deadstatus.event = 1 AND Survival.time < 730 days)

Class 1:
Survival for at least 2 years
(Survival.time >= 730 days)

Censored / undetermined:
Alive with less than 730 days follow-up.


FINAL TARGET DISTRIBUTION
------------------------------------------------------------------------

Class 0 - Death before 2 years:
{class_0_count}

Class 0 percentage:
{class_0_percent:.2f}%

Class 1 - Survival >= 2 years:
{class_1_count}

Class 1 percentage:
{class_1_percent:.2f}%

Censored / undetermined:
{censored_count}

Total classified patients:
{classified_count}


METHODOLOGICAL STATUS
------------------------------------------------------------------------

PASS - Patient-level samples.
PASS - One row per patient.
PASS - Binary two-year survival endpoint.
PASS - Censored patients not incorrectly assigned.
PASS - No classifier trained.
PASS - No feature selection performed.
PASS - No normalization performed.
PASS - No cross-validation performed.

NEXT STEP
------------------------------------------------------------------------

Create stratified patient-level 5-fold cross-validation.

Feature selection must occur inside each training fold only.

Normalization parameters must be fitted using training patients
only and then applied to validation patients.

DO NOT select features using the complete dataset before
cross-validation.


OUTPUT FILES
------------------------------------------------------------------------

Classification dataset:
{OUTPUT_DATASET}

Censored patients:
{CENSORED_FILE}

Class distribution:
{DISTRIBUTION_FILE}

Summary:
{SUMMARY_FILE}

===========================================================================
STEP 21 COMPLETE
===========================================================================
"""


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(summary_text)


print(
    f"\nSaved:\n{SUMMARY_FILE}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("STEP 21 COMPLETE")
print("=" * 75)

print(
    f"\nTotal original patients: "
    f"{total_count}"
)

print(
    f"Binary classification patients: "
    f"{classified_count}"
)

print(
    f"Class 0: "
    f"{class_0_count} "
    f"({class_0_percent:.2f}%)"
)

print(
    f"Class 1: "
    f"{class_1_count} "
    f"({class_1_percent:.2f}%)"
)

print(
    f"Censored / undetermined: "
    f"{censored_count}"
)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nOUTPUT FILES:")
print(OUTPUT_DATASET)
print(CENSORED_FILE)
print(DISTRIBUTION_FILE)
print(SUMMARY_FILE)

print("\n" + "=" * 75)
print("READY FOR STEP 22 - STRATIFIED PATIENT-LEVEL 5-FOLD CV")
print("=" * 75)


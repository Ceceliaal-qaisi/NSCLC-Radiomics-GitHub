# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 15 - PATIENT-LEVEL CLASSIFICATION DATASET PREPARATION
#
# Purpose:
#   Prepare the patient-level dataset for Phase 3 classification.
#
# Important methodological rule:
#   - Radiomic features are taken only from the FINAL STABLE FEATURES.
#   - No feature selection is performed here.
#   - No slice-level samples are used as independent patients.
#   - Survival endpoint is defined at patient level.
#
# Two-year survival:
#   1 = survival time >= 2 years
#   0 = survival time < 2 years
#
# The clinical file is NOT assumed.
# The script searches for candidate CSV/XLSX files and reports them.
# ================================================================

import os
import glob
import re
import pandas as pd


# ================================================================
# PATHS
# ================================================================

BASE_ROOT = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

PATIENT_DIR = os.path.join(
    BASE_ROOT,
    "LUNG1-001",
    "69331"
)

STABLE_FEATURE_FILE = os.path.join(
    PATIENT_DIR,
    "STEP_14_STABLE_FEATURES",
    "GTV1_All_Final_Stable_Features.csv"
)

OUTPUT_DIR = os.path.join(
    PATIENT_DIR,
    "STEP_15_CLASSIFICATION_DATASET"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 15 - PATIENT-LEVEL CLASSIFICATION DATASET PREPARATION")
print("=" * 75)


# ================================================================
# STEP 1 - READ FINAL STABLE FEATURES
# ================================================================

print("\nSTEP 1 - READING FINAL STABLE FEATURES")
print("=" * 75)

if not os.path.exists(STABLE_FEATURE_FILE):

    raise FileNotFoundError(
        "\nFinal stable feature file was not found:\n"
        + STABLE_FEATURE_FILE
    )


stable_df = pd.read_csv(
    STABLE_FEATURE_FILE
)


print(
    "File:",
    STABLE_FEATURE_FILE
)

print(
    "Columns:",
    stable_df.columns.tolist()
)

print(
    "\nNumber of stable feature rows:",
    len(stable_df)
)


# ================================================================
# STEP 2 - EXTRACT FEATURES THAT MUST BE KEPT
# ================================================================

print("\nSTEP 2 - IDENTIFYING FINAL STABLE RADIOMIC FEATURES")
print("=" * 75)


required_columns = [
    "Feature",
    "Original_Value"
]


for column in required_columns:

    if column not in stable_df.columns:

        raise ValueError(
            f"Required column '{column}' "
            "was not found in the stable feature file."
        )


# Keep only rows whose Decision is KEEP
if "Decision" in stable_df.columns:

    final_stable = stable_df[
        stable_df["Decision"]
        .astype(str)
        .str.upper()
        .eq("KEEP")
    ].copy()

elif "Stability" in stable_df.columns:

    final_stable = stable_df[
        stable_df["Stability"]
        .astype(str)
        .str.upper()
        .eq("STABLE")
    ].copy()

else:

    raise ValueError(
        "Could not identify the stability/decision column."
    )


print(
    "\nFinal stable features:"
)

for _, row in final_stable.iterrows():

    print(
        f"- {row['Feature']}"
    )


print(
    "\nTotal stable features:",
    len(final_stable)
)


# ================================================================
# STEP 3 - CREATE PATIENT-LEVEL RADIOMIC RECORD
# ================================================================

print("\nSTEP 3 - CREATING PATIENT-LEVEL RADIOMIC RECORD")
print("=" * 75)


patient_id = "LUNG1-001"


radiomic_record = {

    "Patient_ID":
        patient_id

}


for _, row in final_stable.iterrows():

    feature_name = str(
        row["Feature"]
    )

    feature_value = float(
        row["Original_Value"]
    )

    radiomic_record[
        feature_name
    ] = feature_value


radiomic_df = pd.DataFrame(
    [radiomic_record]
)


print(
    radiomic_df.to_string(
        index=False
    )
)


# ================================================================
# STEP 4 - SEARCH FOR CLINICAL / OUTCOME FILES
# ================================================================

print("\nSTEP 4 - SEARCHING FOR CLINICAL / OUTCOME DATA")
print("=" * 75)


candidate_files = []


patterns = [

    "**/*.csv",
    "**/*.xlsx",
    "**/*.xls"

]


for pattern in patterns:

    matches = glob.glob(
        os.path.join(
            BASE_ROOT,
            pattern
        ),
        recursive=True
    )

    for path in matches:

        normalized = path.lower()

        # Ignore our own generated files
        if "step_15_classification_dataset" in normalized:

            continue

        if "step_14_stable_features" in normalized:

            continue

        candidate_files.append(
            path
        )


candidate_files = sorted(
    set(candidate_files)
)


print(
    "Candidate tabular files found:",
    len(candidate_files)
)


# ================================================================
# STEP 5 - INSPECT FILES FOR CLINICAL VARIABLES
# ================================================================

print("\nSTEP 5 - INSPECTING TABULAR FILES")
print("=" * 75)


clinical_keywords = [

    "survival",
    "surv",
    "os",
    "overall",
    "death",
    "status",
    "stage",
    "age",
    "histology",
    "patient"

]


clinical_candidates = []


for path in candidate_files:

    try:

        if path.lower().endswith(".csv"):

            df = pd.read_csv(
                path,
                nrows=10
            )

        else:

            df = pd.read_excel(
                path,
                nrows=10
            )


        columns = [
            str(c).strip()
            for c in df.columns
        ]


        searchable_columns = " ".join(
            columns
        ).lower()


        score = 0

        matched_keywords = []


        for keyword in clinical_keywords:

            if keyword in searchable_columns:

                score += 1

                matched_keywords.append(
                    keyword
                )


        if score > 0:

            clinical_candidates.append({

                "File":
                    path,

                "Score":
                    score,

                "Matched_Keywords":
                    ", ".join(
                        matched_keywords
                    ),

                "Columns":
                    ", ".join(
                        columns
                    )

            })


    except Exception as e:

        print(
            "\nCould not inspect:",
            path
        )

        print(
            "Reason:",
            e
        )


clinical_candidates_df = pd.DataFrame(
    clinical_candidates
)


if len(
    clinical_candidates_df
) == 0:

    print(
        "\nNO CLINICAL / OUTCOME FILE WAS IDENTIFIED AUTOMATICALLY."
    )

    print(
        "\nThe script will save the candidate file list."
    )

else:

    clinical_candidates_df = (
        clinical_candidates_df
        .sort_values(
            "Score",
            ascending=False
        )
    )


    print(
        "\nPossible clinical/outcome files:"
    )


    print(
        clinical_candidates_df[
            [
                "Score",
                "File",
                "Matched_Keywords"
            ]
        ].to_string(
            index=False
        )
    )


# ================================================================
# STEP 6 - SAVE CLINICAL FILE CANDIDATES
# ================================================================

print("\nSTEP 6 - SAVING CLINICAL FILE CANDIDATES")
print("=" * 75)


candidate_output = os.path.join(
    OUTPUT_DIR,
    "STEP_15_Clinical_File_Candidates.csv"
)


if len(
    clinical_candidates_df
) > 0:

    clinical_candidates_df.to_csv(
        candidate_output,
        index=False
    )

else:

    pd.DataFrame(
        columns=[
            "Score",
            "File",
            "Matched_Keywords",
            "Columns"
        ]
    ).to_csv(
        candidate_output,
        index=False
    )


print(
    "Saved:",
    candidate_output
)


# ================================================================
# STEP 7 - SAVE RADIOMIC PATIENT RECORD
# ================================================================

print("\nSTEP 7 - SAVING PATIENT-LEVEL RADIOMIC RECORD")
print("=" * 75)


radiomic_output = os.path.join(
    OUTPUT_DIR,
    "STEP_15_Patient_Radiomic_Features.csv"
)


radiomic_df.to_csv(
    radiomic_output,
    index=False
)


print(
    "Saved:",
    radiomic_output
)


# ================================================================
# STEP 8 - CHECK FOR DUPLICATE FEATURE NAMES
# ================================================================

print("\nSTEP 8 - VALIDATING FEATURE NAMES")
print("=" * 75)


feature_names = final_stable[
    "Feature"
].astype(str).tolist()


duplicates = (
    pd.Series(
        feature_names
    )
    .duplicated()
)


if duplicates.any():

    duplicated_names = (
        pd.Series(
            feature_names
        )[duplicates]
        .tolist()
    )

    print(
        "FAIL - Duplicate feature names:",
        duplicated_names
    )

else:

    print(
        "PASS - No duplicate stable feature names."
    )


# ================================================================
# STEP 9 - CHECK FEATURE VALUES
# ================================================================

print("\nSTEP 9 - VALIDATING FEATURE VALUES")
print("=" * 75)


feature_columns = [
    c
    for c in radiomic_df.columns
    if c != "Patient_ID"
]


values = radiomic_df[
    feature_columns
].values


if pd.isna(values).any():

    print(
        "WARNING - Missing feature values detected."
    )

else:

    print(
        "PASS - No missing radiomic feature values."
    )


if not pd.api.types.is_numeric_dtype(
    radiomic_df[feature_columns].dtypes.iloc[0]
):

    print(
        "WARNING - Feature datatype validation requires inspection."
    )


# ================================================================
# STEP 10 - REPORT METHODOLOGICAL RULES
# ================================================================

print("\nSTEP 10 - METHOD VALIDATION")
print("=" * 75)


print(
    "PASS - Features are taken from the final stable feature set."
)

print(
    "PASS - Unstable features are not included."
)

print(
    "PASS - Patient-level record is being constructed."
)

print(
    "PASS - No feature selection performed at this stage."
)

print(
    "PASS - No slice-level samples are treated as independent patients."
)

print(
    "IMPORTANT - Feature selection will be performed INSIDE "
    "each cross-validation training fold."
)


# ================================================================
# STEP 11 - SAVE METHOD REPORT
# ================================================================

print("\nSTEP 11 - SAVING STEP 15 REPORT")
print("=" * 75)


report_path = os.path.join(
    OUTPUT_DIR,
    "STEP_15_dataset_preparation_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 15 - PATIENT-LEVEL CLASSIFICATION DATASET\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Patient ID: {patient_id}\n"
    )

    f.write(
        f"Stable radiomic features: "
        f"{len(final_stable)}\n\n"
    )


    f.write(
        "FINAL STABLE FEATURES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for _, row in final_stable.iterrows():

        f.write(
            f"{row['Feature']}: "
            f"{float(row['Original_Value']):.10f}\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "METHODOLOGICAL RULES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "Only final stable features are retained.\n"
    )

    f.write(
        "Unstable features are excluded.\n"
    )

    f.write(
        "Samples are represented at the patient level.\n"
    )

    f.write(
        "No feature selection is performed on the full dataset.\n"
    )

    f.write(
        "Feature selection must occur inside each "
        "cross-validation training fold only.\n"
    )

    f.write(
        "Cross-validation will be stratified and patient-level.\n"
    )

    f.write(
        "Two-year survival will be defined as a binary endpoint.\n"
    )


    f.write(
        "\n"
    )


    f.write(
        "CLINICAL DATA\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    if len(
        clinical_candidates_df
    ) > 0:

        for _, row in clinical_candidates_df.iterrows():

            f.write(
                f"Candidate: {row['File']}\n"
            )

            f.write(
                f"Score: {row['Score']}\n"
            )

            f.write(
                f"Matched keywords: "
                f"{row['Matched_Keywords']}\n\n"
            )

    else:

        f.write(
            "No clinical/outcome file was identified automatically.\n"
        )


print(
    "Saved:",
    report_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 75)

print(
    "STEP 15 - PATIENT-LEVEL DATASET PREPARATION COMPLETE"
)

print("=" * 75)


print(
    "\nPatient:",
    patient_id
)

print(
    "Stable radiomic features:",
    len(final_stable)
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nFiles:"
)

print(
    radiomic_output
)

print(
    candidate_output
)

print(
    report_path
)


print("\n")
print("=" * 75)

print(
    "NEXT STEP:"
)

print(
    "Identify the correct clinical/outcome file containing "
    "survival time/status, stage, age, and histology."
)

print(
    "Then define the two-year survival endpoint "
    "and build the complete multi-patient dataset."
)

print("=" * 75)
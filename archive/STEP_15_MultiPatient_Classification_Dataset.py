# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 15 - MULTI-PATIENT CLASSIFICATION DATASET
#
# Purpose:
#   Collect the final stable radiomic features for ALL patients
#   under the NSCLC radiomics root directory.
#
# Root:
#   C:\Users\CeCe\Downloads\nsclc_radiomics
#
# Important:
#   - One row = one patient
#   - No slice-level samples
#   - No feature selection is performed here
#   - Unstable features are excluded
#   - Statistical Entropy and GLCM Entropy receive unique names
# ================================================================

import os
import glob
import pandas as pd


# ================================================================
# ROOT DIRECTORY
# ================================================================

ROOT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
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
print("STEP 15 - MULTI-PATIENT CLASSIFICATION DATASET")
print("=" * 75)

print("\nRoot directory:")
print(ROOT_DIR)


# ================================================================
# REQUIRED FINAL STABLE FEATURES
# ================================================================

EXPECTED_FEATURES = [

    "Statistical_Mean",
    "Statistical_Variance",
    "Statistical_Smoothness",
    "Statistical_Third_Moment",
    "Statistical_Uniformity",
    "Statistical_Entropy",

    "GLCM_Contrast",
    "GLCM_Correlation",
    "GLCM_Homogeneity",
    "GLCM_Entropy",

    "LBP_Mean",
    "LBP_Variance",
    "LBP_Uniformity",
    "LBP_Entropy",

    "Spectral_Fundamental_Period",
    "Spectral_Angular_Mean",
    "Spectral_Angular_Variance"

]


# ================================================================
# STEP 1 - FIND ALL PATIENT DIRECTORIES
# ================================================================

print("\nSTEP 1 - FINDING ALL PATIENTS")
print("=" * 75)


patient_dirs = []


for path in glob.glob(
    os.path.join(
        ROOT_DIR,
        "LUNG1-*"
    )
):

    if os.path.isdir(path):

        patient_dirs.append(path)


patient_dirs = sorted(
    patient_dirs
)


print(
    "Patient directories found:",
    len(patient_dirs)
)


if len(patient_dirs) == 0:

    raise FileNotFoundError(
        "\nNo LUNG1-* patient directories were found."
    )


print("\nFirst patients:")

for path in patient_dirs[:10]:

    print(
        "-",
        os.path.basename(path)
    )


if len(patient_dirs) > 10:

    print(
        f"... and {len(patient_dirs) - 10} more patients."
    )


# ================================================================
# STEP 2 - SEARCH FOR FINAL STABLE FEATURE FILES
# ================================================================

print("\nSTEP 2 - SEARCHING FOR FINAL STABLE FEATURE FILES")
print("=" * 75)


patient_records = []

patients_with_features = []

patients_without_features = []


for patient_dir in patient_dirs:

    patient_id = os.path.basename(
        patient_dir
    )


    pattern = os.path.join(
        patient_dir,
        "**",
        "GTV1_All_Final_Stable_Features.csv"
    )


    matches = glob.glob(
        pattern,
        recursive=True
    )


    if len(matches) == 0:

        patients_without_features.append(
            patient_id
        )

        print(
            f"[NO FEATURES] {patient_id}"
        )

        continue


    # Use first matching file
    stable_file = matches[0]


    print(
        f"\n[PROCESSING] {patient_id}"
    )

    print(
        "Feature file:",
        stable_file
    )


    try:

        df = pd.read_csv(
            stable_file
        )

    except Exception as e:

        print(
            "ERROR reading file:",
            e
        )

        patients_without_features.append(
            patient_id
        )

        continue


    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------

    required_columns = [

        "Feature",
        "Original_Value",
        "Decision"

    ]


    missing = [

        c
        for c in required_columns
        if c not in df.columns

    ]


    if len(missing) > 0:

        print(
            "ERROR - Missing columns:",
            missing
        )

        patients_without_features.append(
            patient_id
        )

        continue


    # ------------------------------------------------------------
    # Keep only stable features
    # ------------------------------------------------------------

    stable = df[
        df["Decision"]
        .astype(str)
        .str.upper()
        .eq("KEEP")
    ].copy()


    print(
        "Stable feature rows:",
        len(stable)
    )


    # ------------------------------------------------------------
    # Create patient record
    # ------------------------------------------------------------

    record = {

        "Patient_ID":
            patient_id

    }


    for _, row in stable.iterrows():

        feature_name = str(
            row["Feature"]
        ).strip()


        value = float(
            row["Original_Value"]
        )


        # --------------------------------------------------------
        # Resolve duplicate Entropy names
        # --------------------------------------------------------

        if feature_name == "Entropy":

            feature_group = ""


            if "Feature_Group" in stable.columns:

                feature_group = str(
                    row["Feature_Group"]
                ).strip().lower()


            if feature_group == "statistical":

                feature_name = (
                    "Statistical_Entropy"
                )

            elif feature_group == "glcm":

                feature_name = (
                    "GLCM_Entropy"
                )

            else:

                print(
                    "WARNING - Ambiguous Entropy "
                    f"for patient {patient_id}"
                )

                continue


        # --------------------------------------------------------
        # Add group prefix to make names unique
        # --------------------------------------------------------

        elif feature_name in [
            "Mean",
            "Variance",
            "Smoothness",
            "Third_Moment",
            "Uniformity"
        ]:

            feature_name = (
                "Statistical_"
                + feature_name
            )


        elif feature_name in [
            "Contrast",
            "Correlation",
            "Homogeneity",
            "Maximum_Probability"
        ]:

            feature_name = (
                "GLCM_"
                + feature_name
            )


        elif feature_name in [
            "LBP_Mean",
            "LBP_Variance",
            "LBP_Uniformity",
            "LBP_Entropy"
        ]:

            pass


        elif feature_name in [
            "Fundamental_Period",
            "Angular_Mean",
            "Angular_Variance",
            "Dominant_Frequency",
            "Dominant_Orientation",
            "Radial_Mean",
            "Radial_Variance",
            "Radial_Peak_Frequency",
            "Angular_Peak_Orientation"
        ]:

            feature_name = (
                "Spectral_"
                + feature_name
            )


        # --------------------------------------------------------
        # Store feature
        # --------------------------------------------------------

        record[
            feature_name
        ] = value


    patient_records.append(
        record
    )

    patients_with_features.append(
        patient_id
    )


# ================================================================
# STEP 3 - CREATE MULTI-PATIENT DATAFRAME
# ================================================================

print("\nSTEP 3 - CREATING MULTI-PATIENT TABLE")
print("=" * 75)


if len(patient_records) == 0:

    raise ValueError(
        "No patient feature records were created."
    )


all_features_df = pd.DataFrame(
    patient_records
)


# ================================================================
# STEP 4 - ENSURE ALL 17 FEATURES EXIST
# ================================================================

print("\nSTEP 4 - VALIDATING THE 17 FINAL FEATURES")
print("=" * 75)


for feature in EXPECTED_FEATURES:

    if feature not in all_features_df.columns:

        all_features_df[
            feature
        ] = pd.NA


# Put Patient_ID first
ordered_columns = [

    "Patient_ID"

] + EXPECTED_FEATURES


all_features_df = (
    all_features_df[
        ordered_columns
    ]
)


print(
    "Expected features:",
    len(EXPECTED_FEATURES)
)

print(
    "Actual feature columns:",
    len(
        all_features_df.columns
    ) - 1
)


# ================================================================
# STEP 5 - CHECK DUPLICATE COLUMN NAMES
# ================================================================

print("\nSTEP 5 - CHECKING COLUMN NAMES")
print("=" * 75)


duplicate_columns = (
    all_features_df.columns[
        all_features_df.columns.duplicated()
    ]
    .tolist()
)


if len(duplicate_columns) > 0:

    print(
        "FAIL - Duplicate columns:",
        duplicate_columns
    )

else:

    print(
        "PASS - All column names are unique."
    )


# ================================================================
# STEP 6 - CHECK FEATURE COVERAGE
# ================================================================

print("\nSTEP 6 - FEATURE COVERAGE")
print("=" * 75)


coverage_records = []


for feature in EXPECTED_FEATURES:

    available = int(
        all_features_df[
            feature
        ].notna().sum()
    )


    total = len(
        all_features_df
    )


    percentage = (
        100.0
        *
        available
        /
        total
    )


    coverage_records.append({

        "Feature":
            feature,

        "Patients_Available":
            available,

        "Total_Patients":
            total,

        "Coverage_Percent":
            percentage

    })


    print(
        f"{feature:32s} "
        f"{available}/{total} "
        f"({percentage:.2f}%)"
    )


coverage_df = pd.DataFrame(
    coverage_records
)


# ================================================================
# STEP 7 - SAVE MULTI-PATIENT FEATURES
# ================================================================

print("\nSTEP 7 - SAVING MULTI-PATIENT FEATURE TABLE")
print("=" * 75)


features_output = os.path.join(
    OUTPUT_DIR,
    "STEP_15_All_Patients_Radiomic_Features.csv"
)


all_features_df.to_csv(
    features_output,
    index=False
)


print(
    "Saved:",
    features_output
)


# ================================================================
# STEP 8 - SAVE FEATURE COVERAGE
# ================================================================

print("\nSTEP 8 - SAVING FEATURE COVERAGE")
print("=" * 75)


coverage_output = os.path.join(
    OUTPUT_DIR,
    "STEP_15_Feature_Coverage.csv"
)


coverage_df.to_csv(
    coverage_output,
    index=False
)


print(
    "Saved:",
    coverage_output
)


# ================================================================
# STEP 9 - SAVE PATIENT STATUS
# ================================================================

print("\nSTEP 9 - SAVING PATIENT PROCESSING STATUS")
print("=" * 75)


status_records = []


for patient_id in sorted(
    set(
        patients_with_features
        +
        patients_without_features
    )
):

    status_records.append({

        "Patient_ID":
            patient_id,

        "Radiomic_Features_Found":
            (
                "YES"
                if patient_id
                in patients_with_features
                else "NO"
            )

    })


status_df = pd.DataFrame(
    status_records
)


status_output = os.path.join(
    OUTPUT_DIR,
    "STEP_15_Patient_Processing_Status.csv"
)


status_df.to_csv(
    status_output,
    index=False
)


print(
    "Saved:",
    status_output
)


# ================================================================
# STEP 10 - SAVE REPORT
# ================================================================

print("\nSTEP 10 - SAVING REPORT")
print("=" * 75)


report_output = os.path.join(
    OUTPUT_DIR,
    "STEP_15_MultiPatient_Dataset_Report.txt"
)


with open(
    report_output,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 15 - MULTI-PATIENT CLASSIFICATION DATASET\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )


    f.write(
        f"Root directory:\n{ROOT_DIR}\n\n"
    )


    f.write(
        f"Patient directories found: "
        f"{len(patient_dirs)}\n"
    )


    f.write(
        f"Patients with feature files: "
        f"{len(patients_with_features)}\n"
    )


    f.write(
        f"Patients without feature files: "
        f"{len(patients_without_features)}\n\n"
    )


    f.write(
        "FINAL STABLE FEATURES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for feature in EXPECTED_FEATURES:

        f.write(
            feature
            +
            "\n"
        )


    f.write(
        "\nMETHODOLOGICAL RULES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "One row represents one patient.\n"
    )

    f.write(
        "Slice-level observations are not treated as independent patients.\n"
    )

    f.write(
        "Only final stable features are retained.\n"
    )

    f.write(
        "Unstable features are excluded.\n"
    )

    f.write(
        "Statistical Entropy and GLCM Entropy have unique names.\n"
    )

    f.write(
        "No feature selection is performed at this stage.\n"
    )

    f.write(
        "Feature selection must occur inside each cross-validation "
        "training fold only.\n"
    )

    f.write(
        "Clinical outcome variables are not yet assigned in this step.\n"
    )


    f.write(
        "\nPATIENTS WITHOUT FEATURE FILES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for patient_id in patients_without_features:

        f.write(
            patient_id
            +
            "\n"
        )


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 75)

print(
    "STEP 15 - MULTI-PATIENT DATASET COMPLETE"
)

print("=" * 75)


print(
    "\nTotal patient directories:",
    len(patient_dirs)
)

print(
    "Patients with radiomic features:",
    len(patients_with_features)
)

print(
    "Patients without radiomic features:",
    len(patients_without_features)
)

print(
    "\nFinal radiomic feature columns:",
    len(EXPECTED_FEATURES)
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSaved files:"
)

print(
    features_output
)

print(
    coverage_output
)

print(
    status_output
)

print(
    report_output
)


print("\n")
print("=" * 75)

print(
    "NEXT STEP:"
)

print(
    "Verify feature coverage and connect the patient-level "
    "clinical/outcome data."
)

print(
    "Then define the two-year survival endpoint and class balance."
)

print("=" * 75)
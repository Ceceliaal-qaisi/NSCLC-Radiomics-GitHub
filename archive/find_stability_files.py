```python
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 14 - EVIDENCE-BASED STABLE FEATURE INTEGRATION
#
# Purpose:
#   Integrate only features that survived segmentation stability
#   analysis from:
#       1. Statistical texture
#       2. GLCM texture
#       3. LBP texture
#       4. Spectral texture
#
# Stability criterion:
#   Mean Absolute Change <= 10%  -> STABLE -> KEEP
#   Mean Absolute Change > 10%   -> UNSTABLE -> EXCLUDE
#
# IMPORTANT:
#   This script does NOT manually assume which features are stable.
#   It reads the actual stability CSV files produced by the previous
#   steps and uses their Status and Mean_Absolute_Change_% values.
# ================================================================

import os
import pandas as pd


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_14_STABLE_FEATURES"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# STABILITY INPUT FILES
# ================================================================

stability_files = {

    "Statistical":
        os.path.join(
            BASE_DIR,
            "STEP_10_STATISTICAL_TEXTURE",
            "GTV1_Statistical_Feature_Stability.csv"
        ),

    "GLCM":
        os.path.join(
            BASE_DIR,
            "STEP_11_GLCM_TEXTURE",
            "GTV1_GLCM_Feature_Stability.csv"
        ),

    "LBP":
        os.path.join(
            BASE_DIR,
            "STEP_12_LBP_TEXTURE",
            "GTV1_LBP_Feature_Stability.csv"
        ),

    "Spectral":
        os.path.join(
            BASE_DIR,
            "STEP_13_SPECTRAL_TEXTURE",
            "GTV1_Spectral_Feature_Stability.csv"
        )
}


# ================================================================
# PARAMETERS
# ================================================================

STABILITY_THRESHOLD = 10.0


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("PROJECT 7 - RADIOMICS")
print("STEP 14 - EVIDENCE-BASED STABLE FEATURE INTEGRATION")
print("=" * 80)

print(
    "\nStability criterion:"
)

print(
    f"Mean Absolute Change <= {STABILITY_THRESHOLD}% -> STABLE -> KEEP"
)

print(
    f"Mean Absolute Change > {STABILITY_THRESHOLD}% -> UNSTABLE -> EXCLUDE"
)


# ================================================================
# FUNCTION: READ STABILITY FILE
# ================================================================

def read_stability_file(
    feature_group,
    file_path
):

    print("\n")
    print("=" * 80)
    print(f"READING {feature_group.upper()} STABILITY RESULTS")
    print("=" * 80)

    print(
        "File:",
        file_path
    )

    # ------------------------------------------------------------
    # Check file exists
    # ------------------------------------------------------------

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nStability file not found:\n{file_path}"
        )

    # ------------------------------------------------------------
    # Read CSV
    # ------------------------------------------------------------

    df = pd.read_csv(
        file_path
    )

    print(
        "\nColumns found:"
    )

    print(
        list(df.columns)
    )

    # ------------------------------------------------------------
    # Required column
    # ------------------------------------------------------------

    if "Feature" not in df.columns:

        raise ValueError(
            f"{feature_group}: "
            f"'Feature' column not found."
        )

    # ------------------------------------------------------------
    # Identify stability percentage column
    # ------------------------------------------------------------

    change_column = None

    possible_change_columns = [

        "Mean_Absolute_Change_%",
        "Mean_Change_%",
        "Mean_Absolute_Change"

    ]

    for column in possible_change_columns:

        if column in df.columns:

            change_column = column
            break

    if change_column is None:

        raise ValueError(
            f"{feature_group}: "
            "Could not find a mean percentage change column."
        )

    # ------------------------------------------------------------
    # Identify status
    # ------------------------------------------------------------

    status_column = None

    possible_status_columns = [

        "Status",
        "Stability",
        "Result"

    ]

    for column in possible_status_columns:

        if column in df.columns:

            status_column = column
            break

    if status_column is None:

        raise ValueError(
            f"{feature_group}: "
            "Could not find a stability status column."
        )

    # ------------------------------------------------------------
    # Normalize status
    # ------------------------------------------------------------

    df["Status_Normalized"] = (

        df[status_column]
        .astype(str)
        .str.strip()
        .str.upper()

    )

    # ------------------------------------------------------------
    # Normalize percentage change
    # ------------------------------------------------------------

    df["Mean_Absolute_Change_%_Used"] = pd.to_numeric(

        df[change_column],
        errors="coerce"

    )

    # ------------------------------------------------------------
    # Validate values
    # ------------------------------------------------------------

    invalid_change = df[
        df["Mean_Absolute_Change_%_Used"].isna()
    ]

    if len(invalid_change) > 0:

        raise ValueError(
            f"{feature_group}: "
            "Invalid or missing percentage-change values found."
        )

    # ------------------------------------------------------------
    # Independently verify Status using threshold
    # ------------------------------------------------------------

    df["Status_Verified"] = df[
        "Mean_Absolute_Change_%_Used"
    ].apply(

        lambda x:
            "STABLE"
            if abs(float(x)) <= STABILITY_THRESHOLD
            else "UNSTABLE"

    )

    # ------------------------------------------------------------
    # Check reported status against calculated status
    # ------------------------------------------------------------

    df["Status_Match"] = (

        df["Status_Normalized"]
        ==
        df["Status_Verified"]

    )

    mismatches = df[
        ~df["Status_Match"]
    ]

    if len(mismatches) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Reported status does not match "
            "the 10% stability criterion."
        )

        print(
            mismatches[
                [
                    "Feature",
                    status_column,
                    "Mean_Absolute_Change_%_Used",
                    "Status_Verified"
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            f"{feature_group}: "
            "Stability status validation failed."
        )

    # ------------------------------------------------------------
    # Add feature group
    # ------------------------------------------------------------

    df["Feature_Group"] = feature_group

    return df


# ================================================================
# READ ALL STABILITY RESULTS
# ================================================================

all_stability_results = []

for feature_group, file_path in stability_files.items():

    df = read_stability_file(
        feature_group,
        file_path
    )

    all_stability_results.append(
        df
    )


# ================================================================
# COMBINE ALL RESULTS
# ================================================================

combined_df = pd.concat(
    all_stability_results,
    ignore_index=True
)


# ================================================================
# DISPLAY ALL VERIFIED RESULTS
# ================================================================

print("\n")
print("=" * 80)
print("ALL VERIFIED FEATURE STABILITY RESULTS")
print("=" * 80)

display_columns = [

    "Feature_Group",
    "Feature",
    "Mean_Absolute_Change_%_Used",
    "Status_Verified"

]

print(
    combined_df[
        display_columns
    ].to_string(index=False)
)


# ================================================================
# IDENTIFY STABLE FEATURES
# ================================================================

stable_df = combined_df[
    combined_df[
        "Status_Verified"
    ]
    ==
    "STABLE"
].copy()


unstable_df = combined_df[
    combined_df[
        "Status_Verified"
    ]
    ==
    "UNSTABLE"
].copy()


# ================================================================
# ADD DECISION
# ================================================================

combined_df["Decision"] = combined_df[
    "Status_Verified"
].apply(

    lambda x:
        "KEEP"
        if x == "STABLE"
        else "EXCLUDE"

)


# ================================================================
# FINAL STABLE FEATURE TABLE
# ================================================================

final_stable_df = stable_df[
    [
        "Feature_Group",
        "Feature",
        "Mean_Absolute_Change_%_Used",
        "Status_Verified"
    ]
].copy()


final_stable_df.rename(

    columns={
        "Mean_Absolute_Change_%_Used":
            "Mean_Absolute_Change_%"
    },

    inplace=True

)


# ================================================================
# FINAL UNSTABLE FEATURE TABLE
# ================================================================

final_unstable_df = unstable_df[
    [
        "Feature_Group",
        "Feature",
        "Mean_Absolute_Change_%_Used",
        "Status_Verified"
    ]
].copy()


final_unstable_df.rename(

    columns={
        "Mean_Absolute_Change_%_Used":
            "Mean_Absolute_Change_%"
    },

    inplace=True

)


# ================================================================
# SORT
# ================================================================

group_order = {

    "Statistical": 1,
    "GLCM": 2,
    "LBP": 3,
    "Spectral": 4

}


final_stable_df["Group_Order"] = (
    final_stable_df["Feature_Group"]
    .map(group_order)
)


final_stable_df = (

    final_stable_df
    .sort_values(
        [
            "Group_Order",
            "Feature"
        ]
    )
    .drop(
        columns=["Group_Order"]
    )
    .reset_index(drop=True)

)


final_unstable_df["Group_Order"] = (
    final_unstable_df["Feature_Group"]
    .map(group_order)
)


final_unstable_df = (

    final_unstable_df
    .sort_values(
        [
            "Group_Order",
            "Feature"
        ]
    )
    .drop(
        columns=["Group_Order"]
    )
    .reset_index(drop=True)

)


# ================================================================
# ADD FEATURE IDs
# ================================================================

final_stable_df.insert(

    0,
    "Feature_ID",
    range(
        1,
        len(final_stable_df) + 1
    )

)


# ================================================================
# SAVE FINAL STABLE FEATURES
# ================================================================

final_stable_path = os.path.join(

    OUTPUT_DIR,
    "GTV1_All_Final_Stable_Features.csv"

)


final_stable_df.to_csv(

    final_stable_path,
    index=False

)


print("\n")
print("=" * 80)
print("FINAL STABLE FEATURES")
print("=" * 80)

print(
    final_stable_df.to_string(index=False)
)

print(
    "\nSaved:"
)

print(
    final_stable_path
)


# ================================================================
# SAVE EXCLUDED FEATURES
# ================================================================

excluded_path = os.path.join(

    OUTPUT_DIR,
    "GTV1_Excluded_Unstable_Features.csv"

)


final_unstable_df.to_csv(

    excluded_path,
    index=False

)


print("\n")
print("=" * 80)
print("EXCLUDED UNSTABLE FEATURES")
print("=" * 80)

if len(final_unstable_df) == 0:

    print(
        "NONE"
    )

else:

    print(
        final_unstable_df.to_string(index=False)
    )

print(
    "\nSaved:"
)

print(
    excluded_path
)


# ================================================================
# SUMMARY BY GROUP
# ================================================================

summary_rows = []

for feature_group in stability_files.keys():

    group_all = combined_df[
        combined_df[
            "Feature_Group"
        ]
        ==
        feature_group
    ]

    group_stable = group_all[
        group_all[
            "Status_Verified"
        ]
        ==
        "STABLE"
    ]

    group_unstable = group_all[
        group_all[
            "Status_Verified"
        ]
        ==
        "UNSTABLE"
    ]

    summary_rows.append({

        "Feature_Group":
            feature_group,

        "Total_Features":
            len(group_all),

        "Stable_Features":
            len(group_stable),

        "Unstable_Features":
            len(group_unstable)

    })


summary_df = pd.DataFrame(
    summary_rows
)


# ================================================================
# TOTAL
# ================================================================

total_features = len(
    combined_df
)

total_stable = len(
    final_stable_df
)

total_unstable = len(
    final_unstable_df
)


# ================================================================
# SAVE SUMMARY
# ================================================================

summary_path = os.path.join(

    OUTPUT_DIR,
    "GTV1_Stable_Features_Summary.csv"

)


summary_df.to_csv(

    summary_path,
    index=False

)


print("\n")
print("=" * 80)
print("STABILITY SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(index=False)
)

print("\nTOTAL FEATURES:", total_features)
print("TOTAL STABLE:", total_stable)
print("TOTAL UNSTABLE:", total_unstable)

print(
    "\nSaved:"
)

print(
    summary_path
)


# ================================================================
# SAVE COMPLETE EVIDENCE TABLE
# ================================================================

evidence_df = combined_df[
    [
        "Feature_Group",
        "Feature",
        "Mean_Absolute_Change_%_Used",
        "Status_Verified",
        "Decision"
    ]
].copy()


evidence_df.rename(

    columns={
        "Mean_Absolute_Change_%_Used":
            "Mean_Absolute_Change_%"
    },

    inplace=True

)


evidence_path = os.path.join(

    OUTPUT_DIR,
    "GTV1_Complete_Stability_Evidence.csv"

)


evidence_df.to_csv(

    evidence_path,
    index=False

)


print("\n")
print("=" * 80)
print("COMPLETE STABILITY EVIDENCE")
print("=" * 80)

print(
    evidence_df.to_string(index=False)
)

print(
    "\nSaved:"
)

print(
    evidence_path
)


# ================================================================
# SAVE TEXT REPORT
# ================================================================

report_path = os.path.join(

    OUTPUT_DIR,
    "stable_feature_integration_report.txt"

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
        "STEP 14 - EVIDENCE-BASED STABLE FEATURE INTEGRATION\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "PATIENT: LUNG1-001\n"
    )

    f.write(
        "SERIES: 69331\n\n"
    )

    f.write(
        "STABILITY CRITERION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Mean Absolute Change <= 10% -> STABLE -> KEEP\n"
    )

    f.write(
        "Mean Absolute Change > 10% -> UNSTABLE -> EXCLUDE\n\n"
    )

    f.write(
        "FEATURE STABILITY EVIDENCE\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for _, row in evidence_df.iterrows():

        f.write(

            f"{row['Feature_Group']} | "
            f"{row['Feature']} | "
            f"Mean Absolute Change = "
            f"{row['Mean_Absolute_Change_%']:.6f}% | "
            f"{row['Status_Verified']} | "
            f"{row['Decision']}\n"

        )

    f.write("\n")

    f.write(
        "STABLE FEATURES - RETAINED\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for _, row in final_stable_df.iterrows():

        f.write(

            f"{row['Feature_ID']}. "
            f"{row['Feature_Group']} - "
            f"{row['Feature']} | "
            f"Mean Absolute Change = "
            f"{row['Mean_Absolute_Change_%']:.6f}%\n"

        )

    f.write("\n")

    f.write(
        "UNSTABLE FEATURES - EXCLUDED\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    if len(final_unstable_df) == 0:

        f.write(
            "NONE\n"
        )

    else:

        for _, row in final_unstable_df.iterrows():

            f.write(

                f"{row['Feature_Group']} - "
                f"{row['Feature']} | "
                f"Mean Absolute Change = "
                f"{row['Mean_Absolute_Change_%']:.6f}% | "
                f"EXCLUDED\n"

            )

    f.write("\n")

    f.write(
        "SUMMARY BY FEATURE GROUP\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for _, row in summary_df.iterrows():

        f.write(

            f"{row['Feature_Group']}: "
            f"Total = {row['Total_Features']}, "
            f"Stable = {row['Stable_Features']}, "
            f"Unstable = {row['Unstable_Features']}\n"

        )

    f.write("\n")

    f.write(
        f"TOTAL FEATURES ANALYZED: {total_features}\n"
    )

    f.write(
        f"TOTAL STABLE FEATURES: {total_stable}\n"
    )

    f.write(
        f"TOTAL EXCLUDED FEATURES: {total_unstable}\n"
    )

    f.write("\n")

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "Only features satisfying the predefined stability criterion "
        "were retained for the final radiomics feature set.\n"
    )

    f.write(
        "Features exceeding the 10% mean absolute percentage change "
        "were excluded and documented as unstable.\n"
    )


print("\n")
print("=" * 80)
print("REPORT SAVED")
print("=" * 80)

print(
    report_path
)


# ================================================================
# FINAL VALIDATION
# ================================================================

print("\n")
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print(
    f"Total features analyzed : {total_features}"
)

print(
    f"Stable features retained: {total_stable}"
)

print(
    f"Unstable features excluded: {total_unstable}"
)


# Check that every retained feature satisfies criterion

stable_check = all(

    final_stable_df[
        "Mean_Absolute_Change_%"
    ]
    <=
    STABILITY_THRESHOLD

)


# Check that every excluded feature violates criterion

unstable_check = all(

    final_unstable_df[
        "Mean_Absolute_Change_%"
    ]
    >
    STABILITY_THRESHOLD

)


if stable_check:

    print(
        "PASS - All retained features satisfy the <=10% criterion."
    )

else:

    print(
        "FAIL - A retained feature violates the stability criterion."
    )


if unstable_check:

    print(
        "PASS - All excluded features satisfy the >10% criterion."
    )

else:

    print(
        "FAIL - An excluded feature does not satisfy the >10% criterion."
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print("STEP 14 - STABLE FEATURE INTEGRATION COMPLETE")
print("=" * 80)

print(
    "\nFinal stable feature count:",
    total_stable
)

print(
    "Excluded unstable feature count:",
    total_unstable
)

print(
    "\nFinal stable feature file:"
)

print(
    final_stable_path
)

print(
    "\nExcluded feature file:"
)

print(
    excluded_path
)

print(
    "\nComplete evidence file:"
)

print(
    evidence_path
)

print(
    "\nReport:"
)

print(
    report_path
)

print("\n")

if stable_check and unstable_check:

    print(
        "SUCCESS - Evidence-based stable feature integration completed."
    )

else:

    print(
        "WARNING - Review validation results."
    )

print("=" * 80)


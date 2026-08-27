
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 14 - EVIDENCE-BASED STABLE FEATURE INTEGRATION
#
# Purpose:
#   Read the ACTUAL stability-analysis CSV files from:
#       STEP 10 - Statistical
#       STEP 11 - GLCM
#       STEP 12 - LBP
#       STEP 13 - Spectral
#
#   Automatically identify:
#       STABLE  -> KEEP
#       UNSTABLE -> EXCLUDE
#
#   No feature names are hard-coded.
#   The feature names are taken directly from the CSV files.
#
# Stability criterion:
#   Mean Absolute Change <= 10% -> STABLE -> KEEP
#   Mean Absolute Change >  10% -> UNSTABLE -> EXCLUDE
# ================================================================

import os
import pandas as pd
import numpy as np


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
# PARAMETERS
# ================================================================

STABILITY_THRESHOLD = 10.0


# ================================================================
# STABILITY FILES
# ================================================================

STABILITY_FILES = {

    "Statistical": os.path.join(
        BASE_DIR,
        "STEP_10_STATISTICAL_TEXTURE",
        "GTV1_Statistical_Feature_Stability.csv"
    ),

    "GLCM": os.path.join(
        BASE_DIR,
        "STEP_11_GLCM_TEXTURE",
        "GTV1_GLCM_Feature_Stability.csv"
    ),

    "LBP": os.path.join(
        BASE_DIR,
        "STEP_12_LBP_TEXTURE",
        "GTV1_LBP_Feature_Stability.csv"
    ),

    "Spectral": os.path.join(
        BASE_DIR,
        "STEP_13_SPECTRAL_TEXTURE",
        "GTV1_Spectral_Feature_Stability.csv"
    )
}


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("PROJECT 7 - RADIOMICS")
print("STEP 14 - EVIDENCE-BASED STABLE FEATURE INTEGRATION")
print("=" * 80)

print("\nStability criterion:")
print(
    f"Mean Absolute Change <= {STABILITY_THRESHOLD}% "
    "-> STABLE -> KEEP"
)

print(
    f"Mean Absolute Change > {STABILITY_THRESHOLD}% "
    "-> UNSTABLE -> EXCLUDE"
)


# ================================================================
# FUNCTION: FIND CHANGE COLUMN
# ================================================================

def find_change_column(df, group):

    possible_columns = [

        "Mean_Absolute_Change_%",

        "Mean_Absolute_Change_Percent",

        "Mean_Absolute_Change",

        "Mean Absolute Change (%)",

        "Mean_Absolute_Percentage_Change",

        "Mean_Absolute_Percentage_Change_%"

    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    # ------------------------------------------------------------
    # Automatic fallback:
    # find a column containing:
    # mean + absolute + change
    # ------------------------------------------------------------

    normalized = {
        str(column).lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("%", "")
        : column

        for column in df.columns
    }

    for normalized_name, original_name in normalized.items():

        if (
            "mean" in normalized_name
            and "absolute" in normalized_name
            and "change" in normalized_name
        ):

            return original_name

    raise ValueError(
        f"{group}: Could not find a mean absolute "
        f"percentage change column.\n"
        f"Available columns:\n{list(df.columns)}"
    )


# ================================================================
# FUNCTION: FIND STABILITY COLUMN
# ================================================================

def find_stability_column(df, group):

    possible_columns = [

        "Stability",

        "Status",

        "Decision"

    ]

    for column in possible_columns:

        if column in df.columns:
            return column

    return None


# ================================================================
# FUNCTION: READ STABILITY FILE
# ================================================================

def read_stability_file(
    group,
    file_path
):

    print("\n")
    print("=" * 80)
    print(f"READING {group.upper()} STABILITY RESULTS")
    print("=" * 80)

    print(
        "File:",
        file_path
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"{group}: Stability file not found:\n"
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    print("\nColumns found:")
    print(
        list(df.columns)
    )

    # ------------------------------------------------------------
    # Feature column
    # ------------------------------------------------------------

    if "Feature" not in df.columns:

        raise ValueError(
            f"{group}: CSV does not contain "
            f"'Feature' column."
        )

    # ------------------------------------------------------------
    # Find mean change column
    # ------------------------------------------------------------

    change_column = find_change_column(
        df,
        group
    )

    print(
        "\nMean change column:",
        change_column
    )

    # ------------------------------------------------------------
    # Find stability column if available
    # ------------------------------------------------------------

    stability_column = find_stability_column(
        df,
        group
    )

    if stability_column is not None:

        print(
            "Stability column:",
            stability_column
        )

    else:

        print(
            "No stability/status column found."
        )

    # ------------------------------------------------------------
    # Convert change values to numeric
    # ------------------------------------------------------------

    df[change_column] = pd.to_numeric(
        df[change_column],
        errors="coerce"
    )

    if df[change_column].isna().any():

        bad_rows = df[
            df[change_column].isna()
        ]

        raise ValueError(
            f"{group}: Invalid values found in "
            f"{change_column}.\n"
            f"{bad_rows.to_string(index=False)}"
        )

    # ------------------------------------------------------------
    # Determine stability from numerical evidence
    #
    # IMPORTANT:
    # The decision is based on the numerical
    # mean absolute change, not merely a text label.
    # ------------------------------------------------------------

    df["Evidence_Based_Decision"] = np.where(

        df[change_column]
        <=
        STABILITY_THRESHOLD,

        "KEEP",

        "EXCLUDE"

    )

    df["Evidence_Based_Stability"] = np.where(

        df[change_column]
        <=
        STABILITY_THRESHOLD,

        "STABLE",

        "UNSTABLE"

    )

    # ------------------------------------------------------------
    # Create standardized output
    # ------------------------------------------------------------

    standardized = pd.DataFrame({

        "Feature_Group":
            group,

        "Feature":
            df["Feature"].astype(str),

        "Mean_Absolute_Change_Percent":
            df[change_column].astype(float),

        "Stability":
            df["Evidence_Based_Stability"],

        "Decision":
            df["Evidence_Based_Decision"]

    })

    # ------------------------------------------------------------
    # Preserve original values when available
    # ------------------------------------------------------------

    if "Original_Value" in df.columns:

        standardized[
            "Original_Value"
        ] = pd.to_numeric(
            df["Original_Value"],
            errors="coerce"
        )

    elif "Original" in df.columns:

        standardized[
            "Original_Value"
        ] = pd.to_numeric(
            df["Original"],
            errors="coerce"
        )

    else:

        standardized[
            "Original_Value"
        ] = np.nan

    # ------------------------------------------------------------
    # Preserve maximum change when available
    # ------------------------------------------------------------

    maximum_change_column = None

    for column in [

        "Maximum_Change_%",

        "Maximum_Absolute_Change_Percent",

        "Maximum_Absolute_Change_%"

    ]:

        if column in df.columns:

            maximum_change_column = column
            break

    if maximum_change_column is not None:

        standardized[
            "Maximum_Absolute_Change_Percent"
        ] = pd.to_numeric(
            df[maximum_change_column],
            errors="coerce"
        )

    else:

        standardized[
            "Maximum_Absolute_Change_Percent"
        ] = np.nan

    # ------------------------------------------------------------
    # Print evidence
    # ------------------------------------------------------------

    print("\nFEATURE EVIDENCE")
    print("-" * 80)

    for _, row in standardized.iterrows():

        print(

            f"{row['Feature']:<30} "
            f"Mean Change = "
            f"{row['Mean_Absolute_Change_Percent']:.4f}% "
            f"-> "
            f"{row['Decision']}"

        )

    return standardized


# ================================================================
# READ ALL FOUR GROUPS
# ================================================================

all_standardized_results = []


for group, file_path in STABILITY_FILES.items():

    standardized_df = read_stability_file(
        group,
        file_path
    )

    all_standardized_results.append(
        standardized_df
    )


# ================================================================
# COMBINE RESULTS
# ================================================================

print("\n")
print("=" * 80)
print("COMBINING ALL STABILITY RESULTS")
print("=" * 80)


combined_df = pd.concat(
    all_standardized_results,
    ignore_index=True
)


# ================================================================
# ORDER COLUMNS
# ================================================================

combined_df = combined_df[
    [

        "Feature_Group",

        "Feature",

        "Original_Value",

        "Mean_Absolute_Change_Percent",

        "Maximum_Absolute_Change_Percent",

        "Stability",

        "Decision"

    ]
]


# ================================================================
# REMOVE EXACT DUPLICATE ROWS ONLY
# ================================================================

combined_df = combined_df.drop_duplicates(
    subset=[
        "Feature_Group",
        "Feature"
    ]
).reset_index(
    drop=True
)


# ================================================================
# FEATURE ID
# ================================================================

combined_df.insert(
    0,
    "Feature_ID",
    range(
        1,
        len(combined_df) + 1
    )
)


# ================================================================
# FINAL STABLE FEATURES
# ================================================================

final_stable_df = combined_df[
    combined_df["Decision"]
    ==
    "KEEP"
].copy()


# ================================================================
# FINAL EXCLUDED FEATURES
# ================================================================

excluded_df = combined_df[
    combined_df["Decision"]
    ==
    "EXCLUDE"
].copy()


# ================================================================
# DISPLAY FINAL RESULTS
# ================================================================

print("\n")
print("=" * 80)
print("FINAL EVIDENCE-BASED FEATURE RESULTS")
print("=" * 80)

print(
    combined_df.to_string(
        index=False
    )
)


# ================================================================
# DISPLAY STABLE FEATURES
# ================================================================

print("\n")
print("=" * 80)
print("FINAL STABLE FEATURES - KEEP")
print("=" * 80)


if len(final_stable_df) == 0:

    print("NONE")

else:

    for _, row in final_stable_df.iterrows():

        print(

            f"{row['Feature_ID']}. "
            f"{row['Feature_Group']} - "
            f"{row['Feature']} "
            f"(Mean Change = "
            f"{row['Mean_Absolute_Change_Percent']:.4f}%)"

        )


# ================================================================
# DISPLAY EXCLUDED FEATURES
# ================================================================

print("\n")
print("=" * 80)
print("FINAL UNSTABLE FEATURES - EXCLUDE")
print("=" * 80)


if len(excluded_df) == 0:

    print("NONE")

else:

    for _, row in excluded_df.iterrows():

        print(

            f"EXCLUDE: "
            f"{row['Feature_Group']} - "
            f"{row['Feature']} "
            f"(Mean Change = "
            f"{row['Mean_Absolute_Change_Percent']:.4f}%)"

        )


# ================================================================
# SAVE COMPLETE EVIDENCE TABLE
# ================================================================

complete_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_All_Features_Stability_Evidence.csv"

)


combined_df.to_csv(
    complete_csv,
    index=False
)


print("\nSaved:")
print(
    complete_csv
)


# ================================================================
# SAVE FINAL STABLE FEATURES
# ================================================================

stable_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_All_Final_Stable_Features.csv"

)


final_stable_df.to_csv(
    stable_csv,
    index=False
)


print("\nSaved:")
print(
    stable_csv
)


# ================================================================
# SAVE EXCLUDED FEATURES
# ================================================================

excluded_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_Excluded_Unstable_Features.csv"

)


excluded_df.to_csv(
    excluded_csv,
    index=False
)


print("\nSaved:")
print(
    excluded_csv
)


# ================================================================
# SUMMARY BY GROUP
# ================================================================

summary_rows = []


for group in STABILITY_FILES.keys():

    group_df = combined_df[
        combined_df[
            "Feature_Group"
        ]
        ==
        group
    ]

    stable_count = int(
        np.sum(
            group_df["Decision"]
            ==
            "KEEP"
        )
    )

    excluded_count = int(
        np.sum(
            group_df["Decision"]
            ==
            "EXCLUDE"
        )
    )

    total_count = len(
        group_df
    )

    summary_rows.append({

        "Feature_Group":
            group,

        "Total_Features":
            total_count,

        "Stable_Features":
            stable_count,

        "Excluded_Unstable_Features":
            excluded_count

    })


summary_df = pd.DataFrame(
    summary_rows
)


# ================================================================
# SAVE SUMMARY
# ================================================================

summary_csv = os.path.join(

    OUTPUT_DIR,

    "GTV1_Stable_Features_Summary.csv"

)


summary_df.to_csv(
    summary_csv,
    index=False
)


print("\n")
print("=" * 80)
print("STABILITY SUMMARY BY FEATURE GROUP")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(
    summary_csv
)


# ================================================================
# SAVE REPORT
# ================================================================

report_path = os.path.join(

    OUTPUT_DIR,

    "stable_feature_integration_evidence_report.txt"

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
        "Stability criterion:\n"
    )

    f.write(
        f"Mean Absolute Change <= "
        f"{STABILITY_THRESHOLD}% -> "
        f"STABLE -> KEEP\n"
    )

    f.write(
        f"Mean Absolute Change > "
        f"{STABILITY_THRESHOLD}% -> "
        f"UNSTABLE -> EXCLUDE\n\n"
    )

    # ------------------------------------------------------------
    # All features
    # ------------------------------------------------------------

    f.write(
        "ALL FEATURE STABILITY EVIDENCE\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for _, row in combined_df.iterrows():

        f.write(

            f"{row['Feature_ID']}. "
            f"{row['Feature_Group']} - "
            f"{row['Feature']}\n"

        )

        f.write(

            f"Mean Absolute Change: "
            f"{row['Mean_Absolute_Change_Percent']:.6f}%\n"

        )

        if pd.notna(
            row[
                "Maximum_Absolute_Change_Percent"
            ]
        ):

            f.write(

                f"Maximum Absolute Change: "
                f"{row['Maximum_Absolute_Change_Percent']:.6f}%\n"

            )

        f.write(

            f"Stability: "
            f"{row['Stability']}\n"

        )

        f.write(

            f"Decision: "
            f"{row['Decision']}\n\n"

        )

    # ------------------------------------------------------------
    # Final stable
    # ------------------------------------------------------------

    f.write(
        "FINAL STABLE FEATURES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    if len(final_stable_df) == 0:

        f.write(
            "NONE\n"
        )

    else:

        for _, row in final_stable_df.iterrows():

            f.write(

                f"KEEP: "
                f"{row['Feature_Group']} - "
                f"{row['Feature']} "
                f"(Mean Change = "
                f"{row['Mean_Absolute_Change_Percent']:.6f}%)\n"

            )

    f.write("\n")

    # ------------------------------------------------------------
    # Excluded
    # ------------------------------------------------------------

    f.write(
        "EXCLUDED UNSTABLE FEATURES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    if len(excluded_df) == 0:

        f.write(
            "NONE\n"
        )

    else:

        for _, row in excluded_df.iterrows():

            f.write(

                f"EXCLUDE: "
                f"{row['Feature_Group']} - "
                f"{row['Feature']} "
                f"(Mean Change = "
                f"{row['Mean_Absolute_Change_Percent']:.6f}%)\n"

            )

    f.write("\n")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    f.write(
        "SUMMARY\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"Total features analyzed: "
        f"{len(combined_df)}\n"
    )

    f.write(
        f"Stable features kept: "
        f"{len(final_stable_df)}\n"
    )

    f.write(
        f"Unstable features excluded: "
        f"{len(excluded_df)}\n\n"
    )

    for _, row in summary_df.iterrows():

        f.write(

            f"{row['Feature_Group']}: "
            f"Total={row['Total_Features']}, "
            f"Stable={row['Stable_Features']}, "
            f"Excluded={row['Excluded_Unstable_Features']}\n"

        )


print("\nSaved:")
print(
    report_path
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print("STEP 14 - EVIDENCE-BASED STABLE FEATURE INTEGRATION COMPLETE")
print("=" * 80)

print(
    f"\nTotal features analyzed: "
    f"{len(combined_df)}"
)

print(
    f"Stable features kept: "
    f"{len(final_stable_df)}"
)

print(
    f"Unstable features excluded: "
    f"{len(excluded_df)}"
)

print("\nOutput directory:")
print(
    OUTPUT_DIR
)

print("\n")
print("=" * 80)
print("SUCCESS")
print("=" * 80)


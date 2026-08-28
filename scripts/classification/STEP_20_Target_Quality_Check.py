import os
import pandas as pd

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "STEP_19_MERGED_RADIOMICS_CLINICAL",
    "STEP_19_Final_Radiomics_Clinical_Dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_20_TARGET_QUALITY_CHECK"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 20 - TARGET AND DATA QUALITY CHECK")
print("=" * 75)

if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(
        "Input dataset not found: " + INPUT_FILE
    )

df = pd.read_csv(INPUT_FILE)

print("")
print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Columns:", df.columns.tolist())

stable_features = [
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

clinical_columns = [
    "PatientID",
    "age",
    "clinical.T.Stage",
    "Clinical.N.Stage",
    "Clinical.M.Stage",
    "Overall.Stage",
    "Histology",
    "gender",
    "Survival.time",
    "deadstatus.event"
]

print("")
print("=" * 75)
print("COLUMN CHECK")
print("=" * 75)

missing_columns = [
    c for c in stable_features + clinical_columns
    if c not in df.columns
]

if missing_columns:
    print("MISSING COLUMNS:")
    for c in missing_columns:
        print("-", c)
else:
    print("All expected columns are present.")

print("")
print("=" * 75)
print("MISSING VALUE ANALYSIS")
print("=" * 75)

missing_table = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": [
        df[c].isna().sum()
        for c in df.columns
    ]
})

missing_table["Missing_Percent"] = (
    missing_table["Missing_Count"] / len(df) * 100
)

print(missing_table.to_string(index=False))

missing_file = os.path.join(
    OUTPUT_DIR,
    "STEP_20_Missing_Value_Report.csv"
)

missing_table.to_csv(
    missing_file,
    index=False
)
print("")
print("=" * 75)
print("RADIOMIC FEATURE QUALITY")
print("=" * 75)

feature_quality = []

for feature in stable_features:
    numeric = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    feature_quality.append({
        "Feature": feature,
        "Total": len(df),
        "Missing": numeric.isna().sum(),
        "Missing_Percent": (
            numeric.isna().sum() / len(df) * 100
        ),
        "Unique_Values": numeric.nunique(),
        "Mean": numeric.mean(),
        "Std": numeric.std(),
        "Min": numeric.min(),
        "Max": numeric.max()
    })

feature_quality_df = pd.DataFrame(
    feature_quality
)

print(
    feature_quality_df.to_string(index=False)
)

feature_quality_file = os.path.join(
    OUTPUT_DIR,
    "STEP_20_Radiomic_Feature_Quality.csv"
)

feature_quality_df.to_csv(
    feature_quality_file,
    index=False
)


print("")
print("=" * 75)
print("TARGET CANDIDATE: deadstatus.event")
print("=" * 75)

death = pd.to_numeric(
    df["deadstatus.event"],
    errors="coerce"
)

death_counts = death.value_counts(
    dropna=False
).sort_index()

print("Distribution:")
print(death_counts)

death_percent = (
    death_counts / len(df) * 100
)

print("Percent:")
print(death_percent)

death_table = pd.DataFrame({
    "Class": death_counts.index.astype(str),
    "Count": death_counts.values,
    "Percent": death_percent.values
})

death_file = os.path.join(
    OUTPUT_DIR,
    "STEP_20_Death_Status_Distribution.csv"
)

death_table.to_csv(
    death_file,
    index=False
)


print("")
print("=" * 75)
print("TARGET CANDIDATE: Overall.Stage")
print("=" * 75)

stage_counts = (
    df["Overall.Stage"]
    .astype(str)
    .value_counts(dropna=False)
)

print("Distribution:")
print(stage_counts)

stage_table = pd.DataFrame({
    "Stage": stage_counts.index,
    "Count": stage_counts.values,
    "Percent": (
        stage_counts.values / len(df) * 100
    )
})

stage_file = os.path.join(
    OUTPUT_DIR,
    "STEP_20_Overall_Stage_Distribution.csv"
)

stage_table.to_csv(
    stage_file,
    index=False
)


print("")
print("=" * 75)
print("TARGET CANDIDATE: Histology")
print("=" * 75)

histology_counts = (
    df["Histology"]
    .astype(str)
    .value_counts(dropna=False)
)

print("Distribution:")
print(histology_counts)

histology_table = pd.DataFrame({
    "Histology": histology_counts.index,
    "Count": histology_counts.values,
    "Percent": (
        histology_counts.values / len(df) * 100
    )
})

histology_file = os.path.join(
    OUTPUT_DIR,
    "STEP_20_Histology_Distribution.csv"
)

histology_table.to_csv(
    histology_file,
    index=False
)
print("")
print("=" * 75)
print("SURVIVAL TIME QUALITY")
print("=" * 75)

survival = pd.to_numeric(
    df["Survival.time"],
    errors="coerce"
)

print("Missing:", survival.isna().sum())
print("Minimum:", survival.min())
print("Maximum:", survival.max())
print("Mean:", survival.mean())
print("Median:", survival.median())


print("")
print("=" * 75)
print("PATIENT ID CHECK")
print("=" * 75)

patient_id_column = "PatientID"

unique_ids = df[patient_id_column].nunique()
duplicate_ids = df[patient_id_column].duplicated(
    keep=False
).sum()
missing_ids = df[patient_id_column].isna().sum()

print("Patient ID column:", patient_id_column)
print("Unique Patient IDs:", unique_ids)
print("Total rows:", len(df))
print("Duplicate Patient IDs:", duplicate_ids)
print("Missing Patient IDs:", missing_ids)


print("")
print("=" * 75)
print("LUNG1 PATIENT IDENTIFIER CHECK")
print("=" * 75)

patient_ids = df[patient_id_column].astype(str)

lung1_ids = patient_ids[
    patient_ids.str.startswith("LUNG1-")
]

print("LUNG1 IDs:", len(lung1_ids))
print("Non-LUNG1 IDs:", len(df) - len(lung1_ids))


print("")
print("=" * 75)
print("RADIO-METRIC VS DEATH STATUS CHECK")
print("=" * 75)

correlation_rows = []

for feature in stable_features:

    x = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    temp = pd.DataFrame({
        "feature": x,
        "target": death
    }).dropna()

    if len(temp) > 2:
        corr = temp["feature"].corr(
            temp["target"]
        )
    else:
        corr = None

    correlation_rows.append({
        "Feature": feature,
        "Pearson_Correlation_With_Death": corr,
        "Valid_Pairs": len(temp)
    })

correlation_df = pd.DataFrame(
    correlation_rows
)

print(
    correlation_df.to_string(index=False)
)

correlation_file = os.path.join(
    OUTPUT_DIR,
    "STEP_20_Radiomic_Death_Correlation.csv"
)

correlation_df.to_csv(
    correlation_file,
    index=False
)


print("")
print("=" * 75)
print("DATA QUALITY STATUS")
print("=" * 75)

all_columns_present = len(missing_columns) == 0
no_duplicate_ids = duplicate_ids == 0
no_missing_ids = missing_ids == 0
all_ids_lung1 = len(lung1_ids) == len(df)

if (
    all_columns_present
    and no_duplicate_ids
    and no_missing_ids
    and all_ids_lung1
):
    quality_status = "VALID"
else:
    quality_status = "CHECK_REQUIRED"

print("Quality status:", quality_status)


summary_file = os.path.join(
    OUTPUT_DIR,
    "STEP_20_Target_Quality_Summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("PROJECT 7 - RADIOMICS\n")
    f.write(
        "STEP 20 - TARGET AND DATA QUALITY CHECK\n"
    )
    f.write("=" * 75 + "\n\n")

    f.write("SOURCE DATASET\n")
    f.write(INPUT_FILE + "\n\n")

    f.write(
        f"Total patients: {len(df)}\n"
    )
    f.write(
        f"Total columns: {len(df.columns)}\n\n"
    )

    f.write("Stable radiomic features:\n")

    for feature in stable_features:
        f.write(f"- {feature}\n")

    f.write("\nClinical and outcome variables:\n")

    for column in clinical_columns:
        f.write(f"- {column}\n")

    f.write("\nTarget candidates:\n")
    f.write(
        "- deadstatus.event: binary outcome\n"
    )
    f.write(
        "- Overall.Stage: multiclass clinical stage\n"
    )
    f.write(
        "- Histology: multiclass tumor type\n"
    )
    f.write(
        "- Survival.time: survival outcome, "
        "not ordinary binary classification\n"
    )

    f.write("\nDeath status distribution:\n")

    for cls, count in death_counts.items():
        f.write(
            f"{cls}: {count} "
            f"({count / len(df) * 100:.2f}%)\n"
        )

    f.write("\nOverall Stage distribution:\n")

    for stage, count in stage_counts.items():
        f.write(
            f"{stage}: {count} "
            f"({count / len(df) * 100:.2f}%)\n"
        )

    f.write("\nHistology distribution:\n")

    for hist, count in histology_counts.items():
        f.write(
            f"{hist}: {count} "
            f"({count / len(df) * 100:.2f}%)\n"
        )

    f.write("\nSurvival Time Quality:\n")
    f.write(
        f"Missing: {survival.isna().sum()}\n"
    )
    f.write(
        f"Minimum: {survival.min()}\n"
    )
    f.write(
        f"Maximum: {survival.max()}\n"
    )
    f.write(
        f"Mean: {survival.mean()}\n"
    )
    f.write(
        f"Median: {survival.median()}\n"
    )

    f.write("\nPatient ID Check:\n")
    f.write(
        f"Patient ID column: {patient_id_column}\n"
    )
    f.write(
        f"Unique Patient IDs: {unique_ids}\n"
    )
    f.write(
        f"Total rows: {len(df)}\n"
    )
    f.write(
        f"Duplicate Patient IDs: {duplicate_ids}\n"
    )
    f.write(
        f"Missing Patient IDs: {missing_ids}\n"
    )
    f.write(
        f"LUNG1 IDs: {len(lung1_ids)}\n"
    )
    f.write(
        f"Non-LUNG1 IDs: "
        f"{len(df) - len(lung1_ids)}\n"
    )

    f.write(
        f"\nData Quality Status: {quality_status}\n"
    )

    f.write("\nIMPORTANT:\n")
    f.write(
        "One row represents one patient.\n"
    )
    f.write(
        "No slice-level observations are treated "
        "as independent patients.\n"
    )
    f.write(
        "Target selection is not performed in STEP 20.\n"
    )
    f.write(
        "deadstatus.event is a binary outcome candidate.\n"
    )
    f.write(
        "Overall.Stage and Histology are multiclass "
        "outcome candidates.\n"
    )
    f.write(
        "Survival.time is a survival outcome and should "
        "not be treated as an ordinary binary target.\n"
    )


print("")
print("=" * 75)
print("STEP 20 COMPLETE")
print("=" * 75)

print("")
print("Quality status:", quality_status)

print("")
print("Output directory:")
print(OUTPUT_DIR)

print("")
print("Generated files:")
print(missing_file)
print(feature_quality_file)
print(death_file)
print(stage_file)
print(histology_file)
print(correlation_file)
print(summary_file)

print("")
print("=" * 75)
print("READY FOR TARGET DECISION")
print("=" * 75)

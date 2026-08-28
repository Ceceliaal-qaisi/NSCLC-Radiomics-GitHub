import os
import pandas as pd

ROOT = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

RADIOMICS = os.path.join(
    ROOT,
    "STEP_15_FINAL_FEATURE_MATRIX_FIXED",
    "STEP_15_FIXED_Final_Stable_Feature_Matrix.csv"
)

CLINICAL = os.path.join(
    ROOT,
    "STEP_18_CORRECT_CLINICAL_DATASET",
    "STEP_18_Correct_Clinical_Dataset.csv"
)

OUT = os.path.join(
    ROOT,
    "STEP_19_MERGED_RADIOMICS_CLINICAL"
)

os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 19 - MERGE RADIOMICS + CLINICAL DATA")
print("=" * 70)

if not os.path.isfile(RADIOMICS):
    raise FileNotFoundError(RADIOMICS)

if not os.path.isfile(CLINICAL):
    raise FileNotFoundError(CLINICAL)

rad = pd.read_csv(RADIOMICS)
cli = pd.read_csv(CLINICAL)

rad["PatientID"] = rad["Patient_ID"].astype(str)
cli["PatientID"] = cli["PatientID"].astype(str)

rad = rad.drop(columns=["Patient_ID"])

common = sorted(
    set(rad["PatientID"]) &
    set(cli["PatientID"])
)

merged = pd.merge(
    rad,
    cli,
    on="PatientID",
    how="inner"
)

merged = merged.drop_duplicates(
    subset=["PatientID"]
)

output_file = os.path.join(
    OUT,
    "STEP_19_Final_Radiomics_Clinical_Dataset.csv"
)

merged.to_csv(
    output_file,
    index=False
)

radiomic_only = sorted(
    set(rad["PatientID"]) -
    set(cli["PatientID"])
)

clinical_only = sorted(
    set(cli["PatientID"]) -
    set(rad["PatientID"])
)

report = f"""PROJECT 7 - STEP 19 MERGE REPORT
======================================================================

Radiomic patients: {len(rad)}
Clinical patients: {len(cli)}
Common patients: {len(common)}
Radiomic-only patients: {len(radiomic_only)}
Clinical-only patients: {len(clinical_only)}
Merged patients: {len(merged)}
Final columns: {len(merged.columns)}

Radiomics source:
{RADIOMICS}

Clinical source:
{CLINICAL}

Output:
{output_file}
"""

with open(
    os.path.join(
        OUT,
        "STEP_19_Merge_Report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print()
print("Radiomic patients:", len(rad))
print("Clinical patients:", len(cli))
print("Common patients:", len(common))
print("Radiomic-only patients:", len(radiomic_only))
print("Clinical-only patients:", len(clinical_only))
print("Merged patients:", len(merged))
print("Final columns:", len(merged.columns))
print()
print("Output:")
print(output_file)
print()
print("SUCCESS - STEP 19 COMPLETED")

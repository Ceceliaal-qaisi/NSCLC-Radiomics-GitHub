import os
import shutil
import pandas as pd

ROOT = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

SOURCE = (
    r"C:\Users\CeCe\Downloads"
    r"\NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
)

OUT = os.path.join(
    ROOT,
    "STEP_18_CORRECT_CLINICAL_DATASET"
)

os.makedirs(OUT, exist_ok=True)

EXPECTED = [
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

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 18 - CORRECT CLINICAL DATASET")
print("=" * 70)

if not os.path.isfile(SOURCE):
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE)

missing = [c for c in EXPECTED if c not in df.columns]
duplicates = int(df["PatientID"].duplicated().sum())
missing_ids = int(df["PatientID"].isna().sum())

valid = (
    len(df) == 422
    and len(df.columns) == 10
    and not missing
    and duplicates == 0
    and missing_ids == 0
)

status = "VALID_CORRECT_CLINICAL_DATASET" if valid else "CHECK_REQUIRED"

FINAL_FILE = os.path.join(
    OUT,
    "STEP_18_Correct_Clinical_Dataset.csv"
)

shutil.copy2(SOURCE, FINAL_FILE)

report = f"""PROJECT 7 - RADIOMICS
STEP 18 - CORRECT CLINICAL DATASET
======================================================================

SOURCE FILE:
{SOURCE}

FINAL CLINICAL DATASET:
{FINAL_FILE}

Rows: {len(df)}
Columns: {len(df.columns)}
Expected patients: 422
Duplicate Patient IDs: {duplicates}
Missing Patient IDs: {missing_ids}

Missing columns:
{", ".join(missing) if missing else "NONE"}

STATUS:
{status}
"""

with open(
    os.path.join(
        OUT,
        "STEP_18_Clinical_Dataset_Report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate IDs:", duplicates)
print("Missing IDs:", missing_ids)
print("Status:", status)
print()
print("Output:")
print(FINAL_FILE)
print()
print("SUCCESS - STEP 18 COMPLETED")

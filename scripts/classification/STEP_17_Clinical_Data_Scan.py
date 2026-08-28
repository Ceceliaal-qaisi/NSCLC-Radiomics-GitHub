import os
import pandas as pd

ROOT = r"C:\Users\CeCe\Downloads\nsclc_radiomics"
CLINICAL = r"C:\Users\CeCe\Downloads\NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
OUT = os.path.join(ROOT, "STEP_17_CLINICAL_DATA_SCAN")

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CLINICAL)

expected = [
    "PatientID", "age", "clinical.T.Stage",
    "Clinical.N.Stage", "Clinical.M.Stage",
    "Overall.Stage", "Histology", "gender",
    "Survival.time", "deadstatus.event"
]

missing = [x for x in expected if x not in df.columns]
duplicates = int(df["PatientID"].duplicated().sum())
missing_ids = int(df["PatientID"].isna().sum())

status = "VALID_CLINICAL_DATASET"

if len(df) != 422 or missing or duplicates or missing_ids:
    status = "CHECK_REQUIRED"

results = pd.DataFrame([{
    "File": CLINICAL,
    "Rows": len(df),
    "Columns": len(df.columns),
    "Expected_Patients": 422,
    "Missing_Columns": ", ".join(missing),
    "Duplicate_Patient_IDs": duplicates,
    "Missing_Patient_IDs": missing_ids,
    "Status": status
}])

results.to_csv(
    os.path.join(OUT, "STEP_17_Clinical_Data_Scan_Results.csv"),
    index=False
)

potential = pd.DataFrame([{
    "File": CLINICAL,
    "File_Name": os.path.basename(CLINICAL),
    "Rows_Read": len(df),
    "Number_of_Columns": len(df.columns),
    "Patient_ID_Columns": "PatientID",
    "Clinical_Outcome_Columns":
        "age | clinical.T.Stage | Clinical.N.Stage | "
        "Clinical.M.Stage | Overall.Stage | Histology | "
        "gender | Survival.time | deadstatus.event",
    "Potential_Clinical_File": "YES",
    "Validation_Status": status
}])

potential.to_csv(
    os.path.join(OUT, "STEP_17_Potential_Clinical_Files.csv"),
    index=False
)

report = f"""PROJECT 7 - RADIOMICS
STEP 17 - CLINICAL DATA SCAN
===============================================

Clinical file:
{CLINICAL}

Rows: {len(df)}
Columns: {len(df.columns)}
Expected patients: 422
Duplicate Patient IDs: {duplicates}
Missing Patient IDs: {missing_ids}

Missing columns:
{", ".join(missing) if missing else "NONE"}

FINAL STATUS:
{status}
"""

with open(
    os.path.join(OUT, "STEP_17_Clinical_Data_Scan_Report.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print("=" * 60)
print("STEP 17 - CLINICAL DATA SCAN")
print("=" * 60)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate IDs:", duplicates)
print("Missing IDs:", missing_ids)
print("Status:", status)
print()
print("SUCCESS - STEP 17 COMPLETED")

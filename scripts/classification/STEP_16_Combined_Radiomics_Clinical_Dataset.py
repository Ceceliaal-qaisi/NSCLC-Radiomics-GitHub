import pandas as pd
import os

ROOT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

FEATURES_FILE = os.path.join(
    ROOT_DIR,
    "STEP_15_FINAL_FEATURE_MATRIX_FIXED",
    "STEP_15_FIXED_Final_Stable_Feature_Matrix.csv"
)

CLINICAL_FILE = r"C:\Users\CeCe\Downloads\NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "STEP_16_CLASSIFICATION_DATASET"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "STEP_16_Combined_Radiomics_Clinical_Dataset.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 16 - COMBINING RADIOMICS + CLINICAL DATA")
print("=" * 75)

features = pd.read_csv(FEATURES_FILE)
clinical = pd.read_csv(CLINICAL_FILE)

print("\nRadiomics patients:", len(features))
print("Clinical patients:", len(clinical))

features = features.rename(
    columns={"Patient_ID": "PatientID"}
)

merged = pd.merge(
    features,
    clinical,
    on="PatientID",
    how="inner"
)

print("Merged patients:", len(merged))
print("Merged columns:", len(merged.columns))

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nFirst 5 patients:")
print(merged[["PatientID"]].head().to_string(index=False))

print("\nSUCCESS - STEP 16 CREATED")

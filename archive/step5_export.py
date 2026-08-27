import os
import numpy as np
import pandas as pd

# ============================================================
# STEP 5 - EXPORT REGIONAL DESCRIPTORS TO EXCEL
# ============================================================

PATIENT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

RESULTS_DIR = os.path.join(
    PATIENT_DIR,
    "REGIONAL_DESCRIPTORS"
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "regional_descriptors_results.npy"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "STEP5_REGIONAL_DESCRIPTORS_VALUES.xlsx"
)

print("=" * 70)
print("STEP 5 - EXPORTING REGIONAL DESCRIPTOR VALUES")
print("=" * 70)

# ------------------------------------------------------------
# Check input file
# ------------------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print("ERROR: Results file not found!")
    print(INPUT_FILE)
    input("Press ENTER to close...")
    raise SystemExit

print("\nLoading results...")
print(INPUT_FILE)

data = np.load(
    INPUT_FILE,
    allow_pickle=True
).item()

print("Results loaded successfully.")
print("Number of slices:", len(data))

# ------------------------------------------------------------
# Extract values
# ------------------------------------------------------------

rows = []

for slice_number in sorted(data.keys(), key=lambda x: int(x)):

    result = data[slice_number]

    print(f"Processing slice: {slice_number}")

    # Hu moments
    hu = np.asarray(
        result["hu_moments"],
        dtype=float
    ).flatten()

    row = {
        "Slice": int(slice_number),

        "Area":
            float(result["area"]),

        "Perimeter":
            float(result["perimeter"]),

        "Compactness":
            float(result["compactness"]),

        "Circularity":
            float(result["circularity"]),

        "Eccentricity":
            float(result["eccentricity"]),

        "Solidity":
            float(result["solidity"]),
    }

    # --------------------------------------------------------
    # Seven Hu invariant moments
    # --------------------------------------------------------

    for i in range(7):

        if i < len(hu):
            row[f"Hu{i+1}"] = hu[i]
        else:
            row[f"Hu{i+1}"] = np.nan

    rows.append(row)

# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary = pd.DataFrame({
    "Item": [
        "Number of tumor slices",
        "Regional descriptors per slice",
        "Total descriptor values",
        "Descriptors",
        "Hu moments"
    ],

    "Value": [
        len(df),
        13,
        len(df) * 13,
        "Area, Perimeter, Compactness, Circularity, "
        "Eccentricity, Solidity",
        "Hu1, Hu2, Hu3, Hu4, Hu5, Hu6, Hu7"
    ]
})

# ------------------------------------------------------------
# Export to Excel
# ------------------------------------------------------------

print("\nSaving Excel file...")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    df.to_excel(
        writer,
        sheet_name="Regional Values",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

# ------------------------------------------------------------
# Format Excel
# ------------------------------------------------------------

from openpyxl import load_workbook

workbook = load_workbook(OUTPUT_FILE)

for worksheet in workbook.worksheets:

    worksheet.freeze_panes = "A2"

    worksheet.auto_filter.ref = worksheet.dimensions

    for column in worksheet.columns:

        max_length = 0

        for cell in column:

            if cell.value is not None:

                length = len(str(cell.value))

                if length > max_length:
                    max_length = length

        column_letter = column[0].column_letter

        worksheet.column_dimensions[
            column_letter
        ].width = min(max(max_length + 2, 12), 25)

workbook.save(OUTPUT_FILE)

# ------------------------------------------------------------
# Final message
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 EXPORT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nTumor slices:")
print(len(df))

print("\nDescriptors per slice:")
print("13")

print("\nTotal descriptor values:")
print(len(df) * 13)

print("\nExcel file:")
print(OUTPUT_FILE)

print("\nSheets:")
print("1. Regional Values")
print("2. Summary")

print("\n" + "=" * 70)

input("Press ENTER to close...")
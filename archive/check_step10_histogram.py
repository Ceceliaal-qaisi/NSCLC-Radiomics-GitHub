import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt

# ============================================================
# CHECK STEP 10 - STATISTICAL TEXTURE HISTOGRAM
# ============================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_DIR = os.path.join(BASE_DIR, "82046")
MASK_FILE = os.path.join(BASE_DIR, "GTV1_MASK.nrrd")

RESULT_DIR = os.path.join(BASE_DIR, "STEP_10_STATISTICAL_TEXTURE")

CSV_FILE = os.path.join(
    RESULT_DIR,
    "GTV1_statistical_texture.csv"
)

print("=" * 70)
print("CHECKING STEP 10 - HISTOGRAM")
print("=" * 70)

# ============================================================
# STEP 1 - LOAD CT
# ============================================================

print("\nSTEP 1 - LOADING CT")

ct_files = []

for file in os.listdir(CT_DIR):
    path = os.path.join(CT_DIR, file)

    if file.lower().endswith(".dcm"):
        try:
            ds = pydicom.dcmread(path)

            if hasattr(ds, "PixelData"):
                ct_files.append(ds)

        except:
            pass

ct_files.sort(
    key=lambda ds: float(
        getattr(ds, "ImagePositionPatient", [0, 0, float(getattr(ds, "InstanceNumber", 0))])[2]
    )
    if hasattr(ds, "ImagePositionPatient")
    else float(getattr(ds, "InstanceNumber", 0))
)

print("CT files found:", len(ct_files))

if len(ct_files) == 0:
    raise FileNotFoundError("No CT DICOM files found.")

# Convert DICOM to HU
ct_slices = []

for ds in ct_files:

    image = ds.pixel_array.astype(np.float64)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    image = image * slope + intercept

    ct_slices.append(image)

ct = np.stack(ct_slices, axis=0)

print("CT shape:", ct.shape)

# ============================================================
# STEP 2 - LOAD MASK
# ============================================================

print("\nSTEP 2 - LOADING GTV-1 MASK")

try:
    import nrrd
except ImportError:
    print("\nERROR: pynrrd is not installed.")
    print("Run:")
    print("py -m pip install pynrrd")
    raise

mask, header = nrrd.read(MASK_FILE)

print("Original mask shape:", mask.shape)

# Original mask is (Y,X,Z)
if mask.shape == (512, 512, 134):
    mask = np.transpose(mask, (2, 0, 1))

print("Converted mask shape:", mask.shape)

if mask.shape != ct.shape:
    raise ValueError(
        f"CT and mask shapes do not match!\n"
        f"CT: {ct.shape}\n"
        f"Mask: {mask.shape}"
    )

binary_mask = mask > 0

print("Tumor voxels:", np.sum(binary_mask))

# ============================================================
# STEP 3 - EXTRACT TUMOR HU
# ============================================================

print("\nSTEP 3 - EXTRACTING TUMOR INTENSITIES")

tumor_values = ct[binary_mask]

print("Number of tumor voxels:", len(tumor_values))
print("Minimum HU:", tumor_values.min())
print("Maximum HU:", tumor_values.max())
print("Mean HU:", tumor_values.mean())

# ============================================================
# STEP 4 - REBUILD HISTOGRAM
# ============================================================

print("\nSTEP 4 - REBUILDING HISTOGRAM")

hist, bin_edges = np.histogram(
    tumor_values,
    bins=256,
    range=(tumor_values.min(), tumor_values.max())
)

probability = hist / hist.sum()

print("Number of bins:", len(hist))
print("Histogram total:", hist.sum())
print("Probability total:", probability.sum())

# ============================================================
# STEP 5 - CHECK HISTOGRAM MEAN
# ============================================================

print("\nSTEP 5 - CHECKING HISTOGRAM STATISTICS")

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

hist_mean = np.sum(bin_centers * probability)

hist_variance = np.sum(
    ((bin_centers - hist_mean) ** 2) * probability
)

hist_third_moment = np.sum(
    ((bin_centers - hist_mean) ** 3) * probability
)

direct_mean = np.mean(tumor_values)

direct_variance = np.var(tumor_values)

direct_third = np.mean(
    (tumor_values - direct_mean) ** 3
)

print("\nHistogram mean :", hist_mean)
print("Direct mean    :", direct_mean)

print("\nHistogram variance :", hist_variance)
print("Direct variance    :", direct_variance)

print("\nHistogram third moment :", hist_third_moment)
print("Direct third moment    :", direct_third)

# ============================================================
# STEP 6 - CHECK EXISTING CSV
# ============================================================

print("\nSTEP 6 - CHECKING SAVED CSV")

if os.path.exists(CSV_FILE):

    print("CSV found:")
    print(CSV_FILE)

    df = pd.read_csv(CSV_FILE)

    print("\nCSV contents:")
    print(df.to_string(index=False))

else:

    print("WARNING: CSV file not found.")
    print(CSV_FILE)

# ============================================================
# STEP 7 - SAVE INDEPENDENT CHECK HISTOGRAM
# ============================================================

print("\nSTEP 7 - CREATING INDEPENDENT HISTOGRAM")

check_histogram = os.path.join(
    RESULT_DIR,
    "CHECK_01_Independent_Intensity_Histogram.png"
)

plt.figure(figsize=(10, 6))

plt.hist(
    tumor_values,
    bins=256,
    range=(tumor_values.min(), tumor_values.max())
)

plt.xlabel("CT Intensity (HU)")
plt.ylabel("Number of Tumor Voxels")
plt.title("Independent GTV-1 Intensity Histogram")

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(check_histogram, dpi=300)

plt.close()

print("Saved:")
print(check_histogram)

# ============================================================
# STEP 8 - SAVE PROBABILITY HISTOGRAM
# ============================================================

check_probability = os.path.join(
    RESULT_DIR,
    "CHECK_02_Independent_Normalized_Histogram.png"
)

plt.figure(figsize=(10, 6))

plt.bar(
    bin_centers,
    probability,
    width=np.diff(bin_edges),
    align="center"
)

plt.xlabel("CT Intensity (HU)")
plt.ylabel("Probability")
plt.title("Independent GTV-1 Normalized Intensity Histogram")

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(check_probability, dpi=300)

plt.close()

print("Saved:")
print(check_probability)

# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL HISTOGRAM CHECK")
print("=" * 70)

print("\nTumor voxels:", len(tumor_values))
print("HU range:", tumor_values.min(), "to", tumor_values.max())
print("Bins:", len(hist))
print("Probability sum:", probability.sum())

mean_difference = abs(hist_mean - direct_mean)

print("\nMean difference:", mean_difference)

if mean_difference < 1.0:
    print("\nPASS: Histogram mean is consistent with direct voxel mean.")
else:
    print("\nWARNING: Histogram mean differs significantly.")

print("\nIndependent histogram files created successfully.")

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)
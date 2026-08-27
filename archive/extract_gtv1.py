
import os
import glob
import numpy as np
import pydicom
import SimpleITK as sitk

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

SEG_FILE = os.path.join(
    BASE,
    "9.554",
    "553521b9-f9e8-4103-b04d-5f032b974b68.dcm"
)

CT_FOLDER = os.path.join(BASE, "82046")

OUTPUT_MASK = os.path.join(BASE, "GTV1_MASK.nrrd")

# ============================================================
# 1. READ DICOM SEG
# ============================================================

print("=" * 70)
print("READING DICOM SEG")
print("=" * 70)

seg = pydicom.dcmread(SEG_FILE)

print("Modality:", seg.Modality)
print("Rows:", seg.Rows)
print("Columns:", seg.Columns)
print("Number of segments:", len(seg.SegmentSequence))

# ============================================================
# 2. FIND GTV-1 SEGMENT
# ============================================================

gtv_segment_number = None

for s in seg.SegmentSequence:
    label = getattr(s, "SegmentLabel", "")
    description = getattr(s, "SegmentDescription", "")

    print(
        f"Segment {s.SegmentNumber}: "
        f"{label} | {description}"
    )

    if description == "GTV-1" or label == "Neoplasm, Primary":
        gtv_segment_number = int(s.SegmentNumber)

if gtv_segment_number is None:
    raise RuntimeError("GTV-1 segment was NOT found.")

print()
print("GTV-1 Segment Number:", gtv_segment_number)

# ============================================================
# 3. READ SEG PIXEL DATA
# ============================================================

pixel_array = seg.pixel_array

print()
print("SEG pixel array shape:", pixel_array.shape)

# ============================================================
# 4. EXTRACT ONLY GTV-1
# ============================================================

frames = []

for i, frame in enumerate(seg.PerFrameFunctionalGroupsSequence):

    segment_id = int(
        frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
    )

    if segment_id == gtv_segment_number:
        frames.append(i)

print()
print("GTV-1 frames:", len(frames))

if len(frames) == 0:
    raise RuntimeError("No GTV-1 frames found.")

# ============================================================
# 5. READ CT FILES
# ============================================================

ct_files = glob.glob(os.path.join(CT_FOLDER, "*.dcm"))

if len(ct_files) == 0:
    raise RuntimeError("No CT DICOM files found.")

print("CT files found:", len(ct_files))

# Read CT datasets
ct_datasets = []

for f in ct_files:
    d = pydicom.dcmread(f, stop_before_pixels=True)

    if getattr(d, "Modality", "") == "CT":
        ct_datasets.append(d)

# ============================================================
# 6. SORT CT BY Z POSITION
# ============================================================

ct_datasets.sort(
    key=lambda x: float(x.ImagePositionPatient[2])
)

print("CT slices:", len(ct_datasets))

# ============================================================
# 7. CREATE EMPTY 3D MASK
# ============================================================

rows = int(ct_datasets[0].Rows)
cols = int(ct_datasets[0].Columns)
num_slices = len(ct_datasets)

mask = np.zeros(
    (num_slices, rows, cols),
    dtype=np.uint8
)

# ============================================================
# 8. MATCH SEG FRAMES TO CT SLICES
# ============================================================

print()
print("=" * 70)
print("MATCHING GTV-1 FRAMES TO CT SLICES")
print("=" * 70)

matched = 0
unmatched = 0

for frame_index in frames:

    frame = seg.PerFrameFunctionalGroupsSequence[frame_index]

    # --------------------------------------------------------
    # Get SEG image position
    # --------------------------------------------------------

    try:
        seg_position = frame.PlanePositionSequence[0].ImagePositionPatient
        seg_z = float(seg_position[2])
    except Exception:
        unmatched += 1
        continue

    # --------------------------------------------------------
    # Find closest CT slice
    # --------------------------------------------------------

    distances = [
        abs(float(ct.ImagePositionPatient[2]) - seg_z)
        for ct in ct_datasets
    ]

    closest_index = int(np.argmin(distances))
    min_distance = distances[closest_index]

    # Allow small numerical differences
    if min_distance > 1.5:
        unmatched += 1
        continue

    # --------------------------------------------------------
    # Add SEG frame to corresponding CT slice
    # --------------------------------------------------------

    binary_frame = pixel_array[frame_index]

    binary_frame = (binary_frame > 0).astype(np.uint8)

    mask[closest_index] = np.maximum(
        mask[closest_index],
        binary_frame
    )

    matched += 1

# ============================================================
# 9. RESULTS
# ============================================================

print()
print("Matched GTV-1 frames:", matched)
print("Unmatched frames:", unmatched)
print("Mask voxels:", int(np.sum(mask)))

if np.sum(mask) == 0:
    raise RuntimeError(
        "ERROR: GTV-1 mask is empty after CT matching."
    )

# ============================================================
# 10. CONVERT NUMPY MASK TO SIMPLEITK
# ============================================================

mask_image = sitk.GetImageFromArray(mask)

# ------------------------------------------------------------
# CT spacing
# ------------------------------------------------------------

pixel_spacing = [
    float(ct_datasets[0].PixelSpacing[0]),
    float(ct_datasets[0].PixelSpacing[1])
]

slice_thickness = float(
    getattr(ct_datasets[0], "SliceThickness", 3.0)
)

mask_image.SetSpacing(
    (
        pixel_spacing[0],
        pixel_spacing[1],
        slice_thickness
    )
)

# ------------------------------------------------------------
# CT origin
# ------------------------------------------------------------

origin = [
    float(x)
    for x in ct_datasets[0].ImagePositionPatient
]

mask_image.SetOrigin(origin)

# ------------------------------------------------------------
# CT direction
# ------------------------------------------------------------

orientation = [
    float(x)
    for x in ct_datasets[0].ImageOrientationPatient
]

row_cosines = orientation[:3]
column_cosines = orientation[3:]

normal = np.cross(
    np.array(row_cosines),
    np.array(column_cosines)
)

direction = (
    row_cosines
    + column_cosines
    + normal.tolist()
)

mask_image.SetDirection(direction)

# ============================================================
# 11. SAVE NRRD
# ============================================================

sitk.WriteImage(
    mask_image,
    OUTPUT_MASK
)

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)

print("Saved:")
print(OUTPUT_MASK)

print()
print("Mask size:", mask_image.GetSize())
print("Mask spacing:", mask_image.GetSpacing())
print("Mask origin:", mask_image.GetOrigin())
print("Mask voxels:", int(np.sum(mask)))


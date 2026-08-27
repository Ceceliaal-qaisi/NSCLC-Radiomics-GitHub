# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 14 - GTV-1 / CT ALIGNMENT QUALITY CONTROL
#
# Purpose:
#   Verify that GTV-1 segmentation is correctly aligned with CT.
#
# Checks:
#   1. CT geometry
#   2. SEG geometry
#   3. FrameOfReferenceUID
#   4. Referenced CT Series
#   5. CT slice references
#   6. Reconstruction of GTV-1 from original DICOM SEG
#   7. Comparison between DICOM SEG GTV-1 and NRRD mask
#   8. HU statistics inside GTV-1
#   9. Percentage of very low HU voxels
#  10. Save QC report and visualization
#
# No radiomics features are calculated here.
# ================================================================

import os
import glob
import numpy as np
import pandas as pd
import pydicom
import nrrd
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_DIR = os.path.join(
    BASE_DIR,
    "82046"
)

SEG_DIR = os.path.join(
    BASE_DIR,
    "9.554"
)

MASK_PATH = os.path.join(
    BASE_DIR,
    "GTV1_MASK.nrrd"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_14_GTV1_CT_ALIGNMENT_QC"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PROJECT 7 - RADIOMICS")
print("STEP 14 - GTV-1 / CT ALIGNMENT QUALITY CONTROL")
print("=" * 70)


# ================================================================
# STEP 1 - FIND CT FILES
# ================================================================

print("\nSTEP 1 - READING CT")
print("=" * 70)

ct_files = glob.glob(
    os.path.join(
        CT_DIR,
        "*.dcm"
    )
)

if len(ct_files) == 0:
    raise FileNotFoundError(
        f"No CT DICOM files found:\n{CT_DIR}"
    )

print(
    "CT files found:",
    len(ct_files)
)


def read_ct_header(path):

    return pydicom.dcmread(
        path,
        stop_before_pixels=True
    )


ct_datasets = [
    read_ct_header(path)
    for path in ct_files
]


# ================================================================
# SORT CT BY IMAGE POSITION
# ================================================================

def ct_sort_key(ds):

    ipp = getattr(
        ds,
        "ImagePositionPatient",
        None
    )

    if ipp is not None:

        return float(ipp[2])

    return float(
        getattr(
            ds,
            "InstanceNumber",
            0
        )
    )


ct_pairs = list(
    zip(
        ct_files,
        ct_datasets
    )
)

ct_pairs.sort(
    key=lambda x: ct_sort_key(x[1])
)

ct_files = [
    x[0]
    for x in ct_pairs
]

ct_datasets = [
    x[1]
    for x in ct_pairs
]


# ================================================================
# CT INFORMATION
# ================================================================

ct_reference_ds = ct_datasets[0]

ct_shape = (
    len(ct_datasets),
    int(ct_reference_ds.Rows),
    int(ct_reference_ds.Columns)
)

ct_frame_uid = str(
    getattr(
        ct_reference_ds,
        "FrameOfReferenceUID",
        ""
    )
)

ct_series_uid = str(
    getattr(
        ct_reference_ds,
        "SeriesInstanceUID",
        ""
    )
)

ct_study_uid = str(
    getattr(
        ct_reference_ds,
        "StudyInstanceUID",
        ""
    )
)

ct_pixel_spacing = list(
    map(
        float,
        ct_reference_ds.PixelSpacing
    )
)

ct_orientation = list(
    map(
        float,
        ct_reference_ds.ImageOrientationPatient
    )
)


print(
    "CT shape:",
    ct_shape
)

print(
    "CT SeriesInstanceUID:",
    ct_series_uid
)

print(
    "CT FrameOfReferenceUID:",
    ct_frame_uid
)

print(
    "CT PixelSpacing:",
    ct_pixel_spacing
)

print(
    "CT ImageOrientationPatient:",
    ct_orientation
)


# ================================================================
# STEP 2 - READ CT PIXELS AND HU
# ================================================================

print("\nSTEP 2 - READING CT PIXELS / HU")
print("=" * 70)

ct_volume = []

for path in ct_files:

    ds = pydicom.dcmread(path)

    pixel_array = ds.pixel_array.astype(
        np.float64
    )

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0.0
        )
    )

    hu = (
        pixel_array * slope
        + intercept
    )

    ct_volume.append(
        hu
    )

ct_volume = np.stack(
    ct_volume,
    axis=0
)

print(
    "CT volume shape:",
    ct_volume.shape
)


# ================================================================
# CREATE CT SOP UID -> SLICE INDEX MAP
# ================================================================

ct_sop_to_index = {}

for index, ds in enumerate(ct_datasets):

    sop_uid = str(
        ds.SOPInstanceUID
    )

    ct_sop_to_index[
        sop_uid
    ] = index


print(
    "CT SOP Instance UIDs mapped:",
    len(ct_sop_to_index)
)


# ================================================================
# STEP 3 - FIND DICOM SEG
# ================================================================

print("\nSTEP 3 - FINDING DICOM SEG")
print("=" * 70)

seg_files = glob.glob(
    os.path.join(
        SEG_DIR,
        "*.dcm"
    )
)

if len(seg_files) == 0:

    raise FileNotFoundError(
        f"No DICOM SEG found in:\n{SEG_DIR}"
    )

print(
    "SEG files found:",
    len(seg_files)
)


seg_path = None
seg_ds = None

for path in seg_files:

    ds = pydicom.dcmread(
        path,
        stop_before_pixels=True
    )

    if getattr(
        ds,
        "Modality",
        ""
    ) == "SEG":

        seg_path = path

        seg_ds = pydicom.dcmread(
            path
        )

        break


if seg_ds is None:

    raise ValueError(
        "No DICOM SEG object was found."
    )


print(
    "SEG file:",
    seg_path
)

print(
    "SEG Modality:",
    seg_ds.Modality
)

print(
    "SEG NumberOfFrames:",
    getattr(
        seg_ds,
        "NumberOfFrames",
        "N/A"
    )
)


# ================================================================
# STEP 4 - SEG GEOMETRY
# ================================================================

print("\nSTEP 4 - CHECKING SEG GEOMETRY")
print("=" * 70)

seg_frame_uid = str(
    getattr(
        seg_ds,
        "FrameOfReferenceUID",
        ""
    )
)

seg_rows = int(
    seg_ds.Rows
)

seg_cols = int(
    seg_ds.Columns
)

seg_pixel_spacing = list(
    map(
        float,
        seg_ds.SharedFunctionalGroupsSequence[0]
        .PixelMeasuresSequence[0]
        .PixelSpacing
    )
)


print(
    "SEG FrameOfReferenceUID:",
    seg_frame_uid
)

print(
    "SEG rows:",
    seg_rows
)

print(
    "SEG columns:",
    seg_cols
)

print(
    "SEG PixelSpacing:",
    seg_pixel_spacing
)


# ================================================================
# FRAME OF REFERENCE CHECK
# ================================================================

print("\nFRAME OF REFERENCE CHECK")
print("-" * 70)

frame_uid_match = (
    ct_frame_uid == seg_frame_uid
)

print(
    "CT FrameOfReferenceUID:",
    ct_frame_uid
)

print(
    "SEG FrameOfReferenceUID:",
    seg_frame_uid
)

print(
    "MATCH:",
    frame_uid_match
)


# ================================================================
# PIXEL GEOMETRY CHECK
# ================================================================

print("\nPIXEL GEOMETRY CHECK")
print("-" * 70)

rows_match = (
    ct_shape[1] == seg_rows
)

cols_match = (
    ct_shape[2] == seg_cols
)

spacing_match = np.allclose(
    ct_pixel_spacing,
    seg_pixel_spacing,
    atol=1e-6
)

print(
    "Rows match:",
    rows_match
)

print(
    "Columns match:",
    cols_match
)

print(
    "Pixel spacing match:",
    spacing_match
)


# ================================================================
# STEP 5 - FIND GTV-1 SEGMENT
# ================================================================

print("\nSTEP 5 - FINDING GTV-1 SEGMENT")
print("=" * 70)

gtv1_segment_number = None

for segment in seg_ds.SegmentSequence:

    segment_number = int(
        segment.SegmentNumber
    )

    label = str(
        getattr(
            segment,
            "SegmentLabel",
            ""
        )
    )

    description = str(
        getattr(
            segment,
            "SegmentDescription",
            ""
        )
    )

    print(
        f"Segment {segment_number}: "
        f"Label='{label}' | "
        f"Description='{description}'"
    )

    combined = (
        label + " " + description
    ).lower()

    if (
        "gtv-1" in combined
        or "gtv1" in combined
        or "neoplasm" in combined
    ):

        gtv1_segment_number = (
            segment_number
        )


if gtv1_segment_number is None:

    raise ValueError(
        "Could not identify GTV-1 segment."
    )


print(
    "\nSelected GTV-1 Segment Number:",
    gtv1_segment_number
)


# ================================================================
# STEP 6 - RECONSTRUCT GTV-1 FROM DICOM SEG
# ================================================================

print("\nSTEP 6 - RECONSTRUCTING GTV-1 FROM DICOM SEG")
print("=" * 70)

seg_pixel_array = seg_ds.pixel_array

print(
    "SEG pixel array shape:",
    seg_pixel_array.shape
)

if seg_pixel_array.ndim != 3:

    raise ValueError(
        "Unexpected SEG pixel array dimensions."
    )


gtv1_seg_mask = np.zeros(
    ct_shape,
    dtype=bool
)


frames_used = 0
frames_skipped = 0
referenced_ct_uids = set()


# ================================================================
# PER-FRAME INFORMATION
# ================================================================

for frame_index in range(
    int(seg_ds.NumberOfFrames)
):

    frame_fg = (
        seg_ds
        .PerFrameFunctionalGroupsSequence[
            frame_index
        ]
    )

    # ------------------------------------------------------------
    # Segment number
    # ------------------------------------------------------------

    frame_segment_number = int(
        frame_fg
        .SegmentIdentificationSequence[0]
        .ReferencedSegmentNumber
    )

    if (
        frame_segment_number
        != gtv1_segment_number
    ):

        continue


    # ------------------------------------------------------------
    # Referenced CT SOP UID
    # ------------------------------------------------------------

    referenced_uid = None

    try:

        source_sequence = (
            frame_fg
            .DerivationImageSequence[0]
            .SourceImageSequence
        )

        if len(source_sequence) > 0:

            referenced_uid = str(
                source_sequence[0]
                .ReferencedSOPInstanceUID
            )

    except Exception:

        referenced_uid = None


    if referenced_uid is None:

        frames_skipped += 1

        continue


    referenced_ct_uids.add(
        referenced_uid
    )


    # ------------------------------------------------------------
    # Find CT slice
    # ------------------------------------------------------------

    if (
        referenced_uid
        not in ct_sop_to_index
    ):

        frames_skipped += 1

        continue


    slice_index = (
        ct_sop_to_index[
            referenced_uid
        ]
    )


    # ------------------------------------------------------------
    # Add segmentation frame
    # ------------------------------------------------------------

    frame_mask = (
        seg_pixel_array[
            frame_index
        ] > 0
    )

    gtv1_seg_mask[
        slice_index
    ] |= frame_mask

    frames_used += 1


print(
    "GTV-1 SEG frames used:",
    frames_used
)

print(
    "Frames skipped:",
    frames_skipped
)

print(
    "CT slices referenced by GTV-1:",
    len(referenced_ct_uids)
)

print(
    "Reconstructed GTV-1 voxels:",
    int(
        np.sum(
            gtv1_seg_mask
        )
    )
)


# ================================================================
# STEP 7 - READ NRRD MASK
# ================================================================

print("\nSTEP 7 - READING NRRD GTV-1 MASK")
print("=" * 70)

nrrd_data, nrrd_header = nrrd.read(
    MASK_PATH
)

print(
    "Original NRRD shape:",
    nrrd_data.shape
)

print(
    "NRRD dtype:",
    nrrd_data.dtype
)


nrrd_mask = np.asarray(
    nrrd_data
)


# ================================================================
# CONVERT NRRD TO CT ORDER
# ================================================================

if nrrd_mask.shape == (
    ct_shape[1],
    ct_shape[2],
    ct_shape[0]
):

    nrrd_mask = np.transpose(
        nrrd_mask,
        (2, 1, 0)
    )

elif nrrd_mask.shape == ct_shape:

    pass

else:

    raise ValueError(
        "NRRD dimensions do not match CT."
    )


nrrd_mask = (
    nrrd_mask > 0
)


print(
    "NRRD mask converted shape:",
    nrrd_mask.shape
)

print(
    "NRRD tumor voxels:",
    int(
        np.sum(
            nrrd_mask
        )
    )
)


# ================================================================
# STEP 8 - COMPARE DICOM SEG WITH NRRD
# ================================================================

print("\nSTEP 8 - COMPARING DICOM SEG GTV-1 WITH NRRD")
print("=" * 70)


intersection = (
    gtv1_seg_mask
    & nrrd_mask
)

union = (
    gtv1_seg_mask
    | nrrd_mask
)

seg_only = (
    gtv1_seg_mask
    & ~nrrd_mask
)

nrrd_only = (
    nrrd_mask
    & ~gtv1_seg_mask
)


seg_voxels = int(
    np.sum(
        gtv1_seg_mask
    )
)

nrrd_voxels = int(
    np.sum(
        nrrd_mask
    )
)

intersection_voxels = int(
    np.sum(
        intersection
    )
)

union_voxels = int(
    np.sum(
        union
    )
)

seg_only_voxels = int(
    np.sum(
        seg_only
    )
)

nrrd_only_voxels = int(
    np.sum(
        nrrd_only
    )
)


if union_voxels > 0:

    dice = (
        2.0
        * intersection_voxels
        / (
            seg_voxels
            + nrrd_voxels
        )
    )

    iou = (
        intersection_voxels
        / union_voxels
    )

else:

    dice = 1.0
    iou = 1.0


print(
    "DICOM SEG voxels:",
    seg_voxels
)

print(
    "NRRD voxels:",
    nrrd_voxels
)

print(
    "Intersection:",
    intersection_voxels
)

print(
    "SEG-only voxels:",
    seg_only_voxels
)

print(
    "NRRD-only voxels:",
    nrrd_only_voxels
)

print(
    f"Dice similarity: {dice:.6f}"
)

print(
    f"IoU similarity:  {iou:.6f}"
)


# ================================================================
# STEP 9 - GTV-1 HU STATISTICS
# ================================================================

print("\nSTEP 9 - GTV-1 HU STATISTICS")
print("=" * 70)

gtv1_hu = ct_volume[
    nrrd_mask
]

if len(gtv1_hu) == 0:

    raise ValueError(
        "NRRD GTV-1 mask is empty."
    )


hu_min = float(
    np.min(
        gtv1_hu
    )
)

hu_max = float(
    np.max(
        gtv1_hu
    )
)

hu_mean = float(
    np.mean(
        gtv1_hu
    )
)

hu_median = float(
    np.median(
        gtv1_hu
    )
)

hu_p01 = float(
    np.percentile(
        gtv1_hu,
        1
    )
)

hu_p05 = float(
    np.percentile(
        gtv1_hu,
        5
    )
)

hu_p25 = float(
    np.percentile(
        gtv1_hu,
        25
    )
)

hu_p75 = float(
    np.percentile(
        gtv1_hu,
        75
    )
)

hu_p95 = float(
    np.percentile(
        gtv1_hu,
        95
    )
)

hu_p99 = float(
    np.percentile(
        gtv1_hu,
        99
    )
)


print(
    f"Minimum HU : {hu_min:.3f}"
)

print(
    f"Maximum HU : {hu_max:.3f}"
)

print(
    f"Mean HU    : {hu_mean:.3f}"
)

print(
    f"Median HU  : {hu_median:.3f}"
)

print(
    f"1st percentile  : {hu_p01:.3f}"
)

print(
    f"5th percentile  : {hu_p05:.3f}"
)

print(
    f"25th percentile : {hu_p25:.3f}"
)

print(
    f"75th percentile : {hu_p75:.3f}"
)

print(
    f"95th percentile : {hu_p95:.3f}"
)

print(
    f"99th percentile : {hu_p99:.3f}"
)


# ================================================================
# LOW HU ANALYSIS
# ================================================================

below_minus_900 = np.sum(
    gtv1_hu < -900
)

below_minus_800 = np.sum(
    gtv1_hu < -800
)

below_minus_500 = np.sum(
    gtv1_hu < -500
)

total_hu_voxels = len(
    gtv1_hu
)


pct_below_900 = (
    100.0
    * below_minus_900
    / total_hu_voxels
)

pct_below_800 = (
    100.0
    * below_minus_800
    / total_hu_voxels
)

pct_below_500 = (
    100.0
    * below_minus_500
    / total_hu_voxels
)


print(
    f"\nHU < -900: "
    f"{int(below_minus_900)} voxels "
    f"({pct_below_900:.2f}%)"
)

print(
    f"HU < -800: "
    f"{int(below_minus_800)} voxels "
    f"({pct_below_800:.2f}%)"
)

print(
    f"HU < -500: "
    f"{int(below_minus_500)} voxels "
    f"({pct_below_500:.2f}%)"
)


# ================================================================
# STEP 10 - DETERMINE ALIGNMENT STATUS
# ================================================================

print("\nSTEP 10 - FINAL ALIGNMENT ASSESSMENT")
print("=" * 70)


geometry_pass = (
    frame_uid_match
    and rows_match
    and cols_match
    and spacing_match
)

mask_similarity_pass = (
    dice >= 0.99
)


if (
    geometry_pass
    and mask_similarity_pass
):

    alignment_status = "PASS"

else:

    alignment_status = "REVIEW"


print(
    "Geometry check:",
    "PASS" if geometry_pass else "FAIL"
)

print(
    f"Mask Dice check: "
    f"{'PASS' if mask_similarity_pass else 'REVIEW'} "
    f"(Dice = {dice:.6f})"
)

print(
    "\nFINAL STATUS:",
    alignment_status
)


# ================================================================
# STEP 11 - SAVE COMPARISON SUMMARY
# ================================================================

print("\nSTEP 11 - SAVING QC SUMMARY")
print("=" * 70)


summary_df = pd.DataFrame({

    "Check": [

        "CT_SEG_FrameOfReferenceUID",
        "Rows",
        "Columns",
        "PixelSpacing",
        "DICOM_SEG_GTV1_Voxels",
        "NRRD_GTV1_Voxels",
        "Intersection_Voxels",
        "SEG_Only_Voxels",
        "NRRD_Only_Voxels",
        "Dice",
        "IoU",
        "Minimum_HU",
        "Maximum_HU",
        "Mean_HU",
        "Median_HU",
        "Percent_HU_Below_-900",
        "Percent_HU_Below_-800",
        "Percent_HU_Below_-500",
        "Final_Alignment_Status"

    ],

    "Value": [

        "PASS"
        if frame_uid_match
        else "FAIL",

        "PASS"
        if rows_match
        else "FAIL",

        "PASS"
        if cols_match
        else "FAIL",

        "PASS"
        if spacing_match
        else "FAIL",

        seg_voxels,

        nrrd_voxels,

        intersection_voxels,

        seg_only_voxels,

        nrrd_only_voxels,

        dice,

        iou,

        hu_min,

        hu_max,

        hu_mean,

        hu_median,

        pct_below_900,

        pct_below_800,

        pct_below_500,

        alignment_status

    ]

})


summary_path = os.path.join(
    OUTPUT_DIR,
    "GTV1_CT_Alignment_QC_Summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print(
    "Saved:",
    summary_path
)


# ================================================================
# STEP 12 - SAVE HU STATISTICS
# ================================================================

hu_summary_df = pd.DataFrame({

    "Statistic": [

        "Minimum",
        "Maximum",
        "Mean",
        "Median",
        "P01",
        "P05",
        "P25",
        "P75",
        "P95",
        "P99"

    ],

    "HU": [

        hu_min,
        hu_max,
        hu_mean,
        hu_median,
        hu_p01,
        hu_p05,
        hu_p25,
        hu_p75,
        hu_p95,
        hu_p99

    ]

})


hu_summary_path = os.path.join(
    OUTPUT_DIR,
    "GTV1_HU_Statistics_QC.csv"
)

hu_summary_df.to_csv(
    hu_summary_path,
    index=False
)

print(
    "Saved:",
    hu_summary_path
)


# ================================================================
# STEP 13 - VISUALIZATION
# ================================================================

print("\nSTEP 13 - SAVING VISUAL QC")
print("=" * 70)


tumor_slice_indices = np.where(
    np.any(
        nrrd_mask,
        axis=(1, 2)
    )
)[0]


middle_slice = int(
    tumor_slice_indices[
        len(tumor_slice_indices) // 2
    ]
)


ct_slice = ct_volume[
    middle_slice
]

mask_slice = nrrd_mask[
    middle_slice
]

seg_slice = gtv1_seg_mask[
    middle_slice
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


axes[0].imshow(
    ct_slice,
    cmap="gray"
)

axes[0].set_title(
    f"CT | Slice {middle_slice}"
)

axes[0].axis("off")


axes[1].imshow(
    ct_slice,
    cmap="gray"
)

axes[1].imshow(
    np.ma.masked_where(
        ~seg_slice,
        seg_slice
    ),
    alpha=0.45
)

axes[1].set_title(
    "Original DICOM SEG GTV-1"
)

axes[1].axis("off")


axes[2].imshow(
    ct_slice,
    cmap="gray"
)

axes[2].imshow(
    np.ma.masked_where(
        ~mask_slice,
        mask_slice
    ),
    alpha=0.45
)

axes[2].set_title(
    "NRRD GTV-1 Mask"
)

axes[2].axis("off")


plt.tight_layout()


visual_path = os.path.join(
    OUTPUT_DIR,
    "01_GTV1_CT_SEG_NRRD_Alignment.png"
)

plt.savefig(
    visual_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    visual_path
)


# ================================================================
# STEP 14 - SAVE HU HISTOGRAM
# ================================================================

print("\nSTEP 14 - SAVING HU HISTOGRAM")
print("=" * 70)


plt.figure(
    figsize=(10, 6)
)

plt.hist(
    gtv1_hu,
    bins=100
)

plt.xlabel(
    "HU"
)

plt.ylabel(
    "Voxel count"
)

plt.title(
    "GTV-1 HU Distribution"
)

plt.tight_layout()


histogram_path = os.path.join(
    OUTPUT_DIR,
    "02_GTV1_HU_Distribution.png"
)

plt.savefig(
    histogram_path,
    dpi=300
)

plt.close()


print(
    "Saved:",
    histogram_path
)


# ================================================================
# STEP 15 - SAVE REPORT
# ================================================================

print("\nSTEP 15 - SAVING QC REPORT")
print("=" * 70)


report_path = os.path.join(
    OUTPUT_DIR,
    "GTV1_CT_Alignment_QC_Report.txt"
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
        "STEP 14 - GTV-1 / CT ALIGNMENT QUALITY CONTROL\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "PATIENT: LUNG1-001\n"
    )

    f.write(
        "SERIES: 69331\n\n"
    )

    f.write(
        "GEOMETRY CHECKS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"CT shape: {ct_shape}\n"
    )

    f.write(
        f"CT SeriesInstanceUID: {ct_series_uid}\n"
    )

    f.write(
        f"CT FrameOfReferenceUID: "
        f"{ct_frame_uid}\n"
    )

    f.write(
        f"SEG FrameOfReferenceUID: "
        f"{seg_frame_uid}\n"
    )

    f.write(
        f"FrameOfReference match: "
        f"{frame_uid_match}\n"
    )

    f.write(
        f"Rows match: {rows_match}\n"
    )

    f.write(
        f"Columns match: {cols_match}\n"
    )

    f.write(
        f"Pixel spacing match: "
        f"{spacing_match}\n\n"
    )

    f.write(
        "GTV-1 MASK COMPARISON\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"DICOM SEG GTV-1 voxels: "
        f"{seg_voxels}\n"
    )

    f.write(
        f"NRRD GTV-1 voxels: "
        f"{nrrd_voxels}\n"
    )

    f.write(
        f"Intersection: "
        f"{intersection_voxels}\n"
    )

    f.write(
        f"SEG-only voxels: "
        f"{seg_only_voxels}\n"
    )

    f.write(
        f"NRRD-only voxels: "
        f"{nrrd_only_voxels}\n"
    )

    f.write(
        f"Dice: {dice:.10f}\n"
    )

    f.write(
        f"IoU: {iou:.10f}\n\n"
    )

    f.write(
        "GTV-1 HU STATISTICS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Minimum HU: {hu_min:.6f}\n"
    )

    f.write(
        f"Maximum HU: {hu_max:.6f}\n"
    )

    f.write(
        f"Mean HU: {hu_mean:.6f}\n"
    )

    f.write(
        f"Median HU: {hu_median:.6f}\n"
    )

    f.write(
        f"P01: {hu_p01:.6f}\n"
    )

    f.write(
        f"P05: {hu_p05:.6f}\n"
    )

    f.write(
        f"P25: {hu_p25:.6f}\n"
    )

    f.write(
        f"P75: {hu_p75:.6f}\n"
    )

    f.write(
        f"P95: {hu_p95:.6f}\n"
    )

    f.write(
        f"P99: {hu_p99:.6f}\n\n"
    )

    f.write(
        "LOW HU ANALYSIS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"HU < -900: "
        f"{pct_below_900:.4f}%\n"
    )

    f.write(
        f"HU < -800: "
        f"{pct_below_800:.4f}%\n"
    )

    f.write(
        f"HU < -500: "
        f"{pct_below_500:.4f}%\n\n"
    )

    f.write(
        "FINAL ASSESSMENT\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    f.write(
        f"Geometry status: "
        f"{'PASS' if geometry_pass else 'FAIL'}\n"
    )

    f.write(
        f"Mask similarity status: "
        f"{'PASS' if mask_similarity_pass else 'REVIEW'}\n"
    )

    f.write(
        f"FINAL ALIGNMENT STATUS: "
        f"{alignment_status}\n"
    )


print(
    "Saved:",
    report_path
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 70)
print("STEP 14 - GTV-1 / CT ALIGNMENT QC COMPLETE")
print("=" * 70)

print(
    "\nFrame of Reference match:",
    frame_uid_match
)

print(
    "Pixel geometry match:",
    geometry_pass
)

print(
    f"DICOM SEG GTV-1 voxels: "
    f"{seg_voxels}"
)

print(
    f"NRRD GTV-1 voxels: "
    f"{nrrd_voxels}"
)

print(
    f"Dice: {dice:.6f}"
)

print(
    f"IoU: {iou:.6f}"
)

print(
    f"Mean GTV-1 HU: "
    f"{hu_mean:.3f}"
)

print(
    f"HU < -900: "
    f"{pct_below_900:.2f}%"
)

print(
    "\nFINAL ALIGNMENT STATUS:",
    alignment_status
)

print(
    "\nQC files saved in:"
)

print(
    OUTPUT_DIR
)

print("\n")
print("=" * 70)
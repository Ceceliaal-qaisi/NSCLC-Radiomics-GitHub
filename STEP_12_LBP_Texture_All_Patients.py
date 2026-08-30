# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 12 - LOCAL BINARY PATTERNS (LBP)
# MULTI-PATIENT DICOM SEG VERSION
# ================================================================
#
# FROM SCRATCH:
#   - LBP calculation
#   - LBP histogram
#   - Mean LBP
#   - Variance of LBP
#   - Uniformity
#   - Entropy
#
# Does NOT use skimage LBP.
# Does NOT require GTV1_MASK.nrrd.
# Finds CT and DICOM SEG automatically.
# Finds GTV-1 / GTV1 / Neoplasm automatically.
# If one patient fails, processing continues.
#
# ================================================================

import os
import glob
import csv
import numpy as np
import pydicom
import matplotlib.pyplot as plt


# ================================================================
# PATHS
# ================================================================

ROOT_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "STEP_12_LBP_TEXTURE_ALL_PATIENTS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# PARAMETERS
# ================================================================

P = 8
R = 1
NUMBER_OF_BINS = 256

GTV_KEYWORDS = [
    "gtv-1",
    "gtv1",
    "gtv 1",
    "neoplasm"
]


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 12 - LOCAL BINARY PATTERNS")
print("MULTI-PATIENT DICOM SEG VERSION")
print("=" * 75)

print("\nROOT:")
print(ROOT_DIR)

print("\nOUTPUT:")
print(OUTPUT_DIR)

print("\nLBP PARAMETERS:")
print("Neighbors P:", P)
print("Radius R:", R)


# ================================================================
# FUNCTION 1 - FIND CT DIRECTORY
# ================================================================

def find_ct_directory(patient_dir):

    possible_ct_dirs = []

    for root, dirs, files in os.walk(patient_dir):

        dcm_files = glob.glob(
            os.path.join(root, "*.dcm")
        )

        if len(dcm_files) == 0:
            continue

        ct_count = 0

        for file in dcm_files:

            try:

                ds = pydicom.dcmread(
                    file,
                    stop_before_pixels=True
                )

                modality = str(
                    getattr(ds, "Modality", "")
                ).upper()

                if modality == "CT":
                    ct_count += 1

            except Exception:

                continue

        if ct_count > 0:

            possible_ct_dirs.append(
                (ct_count, root)
            )

    if len(possible_ct_dirs) == 0:
        return None

    possible_ct_dirs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return possible_ct_dirs[0][1]


# ================================================================
# FUNCTION 2 - READ CT VOLUME
# ================================================================

def read_ct_volume(ct_dir):

    ct_files = glob.glob(
        os.path.join(ct_dir, "*.dcm")
    )

    ct_slices = []

    for file in ct_files:

        try:

            ds = pydicom.dcmread(file)

            modality = str(
                getattr(ds, "Modality", "")
            ).upper()

            if modality != "CT":
                continue

            if not hasattr(
                ds,
                "pixel_array"
            ):
                continue

            if hasattr(
                ds,
                "ImagePositionPatient"
            ):

                z = float(
                    ds.ImagePositionPatient[2]
                )

            else:

                z = float(
                    getattr(
                        ds,
                        "InstanceNumber",
                        0
                    )
                )

            image = ds.pixel_array.astype(
                np.float32
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

            image_hu = (
                image * slope
                + intercept
            )

            ct_slices.append(
                (z, image_hu, ds)
            )

        except Exception:

            continue

    if len(ct_slices) == 0:

        return None, []

    ct_slices.sort(
        key=lambda x: x[0]
    )

    ct_volume = np.stack(
        [
            item[1]
            for item in ct_slices
        ],
        axis=0
    )

    ct_datasets = [
        (
            item[0],
            item[2]
        )
        for item in ct_slices
    ]

    return ct_volume, ct_datasets


# ================================================================
# FUNCTION 3 - FIND DICOM SEG
# ================================================================

def find_seg_file(patient_dir):

    seg_files = []

    for root, dirs, files in os.walk(patient_dir):

        for file in files:

            if not file.lower().endswith(".dcm"):
                continue

            full_path = os.path.join(
                root,
                file
            )

            try:

                ds = pydicom.dcmread(
                    full_path,
                    stop_before_pixels=True
                )

                modality = str(
                    getattr(
                        ds,
                        "Modality",
                        ""
                    )
                ).upper()

                if modality == "SEG":

                    seg_files.append(
                        full_path
                    )

            except Exception:

                continue

    if len(seg_files) == 0:
        return None

    return seg_files[0]


# ================================================================
# FUNCTION 4 - FIND GTV-1 SEGMENT
# ================================================================

def find_gtv1_segment(seg_ds):

    if not hasattr(
        seg_ds,
        "SegmentSequence"
    ):

        return None

    print("\nAvailable segments:")

    for segment in seg_ds.SegmentSequence:

        number = int(
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
            f"  Segment {number}: "
            f"Label='{label}' | "
            f"Description='{description}'"
        )

    for segment in seg_ds.SegmentSequence:

        label = str(
            getattr(
                segment,
                "SegmentLabel",
                ""
            )
        ).lower()

        description = str(
            getattr(
                segment,
                "SegmentDescription",
                ""
            )
        ).lower()

        combined = (
            label
            + " "
            + description
        )

        for keyword in GTV_KEYWORDS:

            if keyword in combined:

                number = int(
                    segment.SegmentNumber
                )

                print("\nGTV-1 FOUND")
                print(
                    "Segment Number:",
                    number
                )

                print(
                    "Label:",
                    label
                )

                print(
                    "Description:",
                    description
                )

                return number

    return None


# ================================================================
# FUNCTION 5 - EXTRACT GTV-1 FRAMES
# ================================================================

def extract_gtv1_frames(
    seg_ds,
    segment_number
):

    if not hasattr(
        seg_ds,
        "PerFrameFunctionalGroupsSequence"
    ):

        raise ValueError(
            "SEG does not contain "
            "PerFrameFunctionalGroupsSequence."
        )

    pixel_array = seg_ds.pixel_array

    frame_numbers = []

    for frame_index, frame_group in enumerate(
        seg_ds.PerFrameFunctionalGroupsSequence
    ):

        try:

            frame_segment_number = int(
                frame_group
                .SegmentIdentificationSequence[0]
                .ReferencedSegmentNumber
            )

        except Exception:

            continue

        if (
            frame_segment_number
            == segment_number
        ):

            frame_numbers.append(
                frame_index
            )

    print(
        "GTV-1 frames:",
        len(frame_numbers)
    )

    if len(frame_numbers) == 0:

        raise ValueError(
            "No GTV-1 frames found."
        )

    return pixel_array, frame_numbers


# ================================================================
# FUNCTION 6 - ALIGN SEG TO CT
# ================================================================

def align_seg_to_ct(
    seg_ds,
    pixel_array,
    frame_numbers,
    ct_volume,
    ct_datasets
):

    final_mask = np.zeros(
        ct_volume.shape,
        dtype=bool
    )

    ct_shape = ct_volume.shape

    matched_frames = 0

    for frame_index in frame_numbers:

        frame = pixel_array[
            frame_index
        ]

        if not np.any(frame):
            continue

        try:

            frame_group = (
                seg_ds
                .PerFrameFunctionalGroupsSequence[
                    frame_index
                ]
            )

            if hasattr(
                frame_group,
                "PlanePositionSequence"
            ):

                position = (
                    frame_group
                    .PlanePositionSequence[0]
                    .ImagePositionPatient
                )

                seg_z = float(
                    position[2]
                )

            else:

                continue

        except Exception:

            continue

        closest_index = None
        smallest_distance = float("inf")

        for ct_index, (
            ct_z,
            ct_ds
        ) in enumerate(ct_datasets):

            distance = abs(
                ct_z - seg_z
            )

            if distance < smallest_distance:

                smallest_distance = distance
                closest_index = ct_index

        if closest_index is None:
            continue

        if (
            frame.shape[0]
            != ct_shape[1]
            or
            frame.shape[1]
            != ct_shape[2]
        ):

            raise ValueError(
                "SEG and CT dimensions do not match."
            )

        final_mask[
            closest_index
        ] |= (
            frame > 0
        )

        matched_frames += 1

    print(
        "Matched GTV-1 frames:",
        matched_frames
    )

    tumor_voxels = int(
        np.sum(final_mask)
    )

    print(
        "Tumor voxels:",
        tumor_voxels
    )

    if tumor_voxels == 0:

        raise ValueError(
            "Final GTV-1 mask is empty."
        )

    return final_mask


# ================================================================
# FUNCTION 7 - CALCULATE LBP FROM SCRATCH
# ================================================================

def calculate_lbp(
    image,
    tumor_mask
):

    rows, cols = image.shape

    lbp_image = np.full(
        (rows, cols),
        -1,
        dtype=np.int16
    )

    # ------------------------------------------------------------
    # Only calculate inside valid one-pixel interior.
    # Vectorized implementation.
    # ------------------------------------------------------------

    center = image[
        1:-1,
        1:-1
    ]

    valid_mask = tumor_mask[
        1:-1,
        1:-1
    ]

    lbp = np.zeros(
        center.shape,
        dtype=np.uint16
    )

    neighbors = [

        image[
            0:-2,
            0:-2
        ],

        image[
            0:-2,
            1:-1
        ],

        image[
            0:-2,
            2:
        ],

        image[
            1:-1,
            2:
        ],

        image[
            2:,
            2:
        ],

        image[
            2:,
            1:-1
        ],

        image[
            2:,
            0:-2
        ],

        image[
            1:-1,
            0:-2
        ]
    ]

    for p in range(P):

        lbp += (
            neighbors[p] >= center
        ).astype(
            np.uint16
        ) * (
            2 ** p
        )

    lbp_image[
        1:-1,
        1:-1
    ][valid_mask] = (
        lbp[valid_mask]
    )

    return lbp_image


# ================================================================
# FUNCTION 8 - SAVE CSV
# ================================================================

def save_csv(
    filename,
    rows,
    fieldnames
):

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(rows)


# ================================================================
# FUNCTION 9 - PROCESS ONE PATIENT
# ================================================================

def process_patient(patient_dir):

    patient_id = os.path.basename(
        patient_dir
    )

    print("\n")
    print("-" * 75)
    print(
        "PROCESSING:",
        patient_id
    )
    print("-" * 75)

    # ------------------------------------------------------------
    # CT
    # ------------------------------------------------------------

    ct_dir = find_ct_directory(
        patient_dir
    )

    if ct_dir is None:

        raise FileNotFoundError(
            "CT directory not found."
        )

    print(
        "CT directory:",
        ct_dir
    )

    ct_volume, ct_datasets = (
        read_ct_volume(
            ct_dir
        )
    )

    if ct_volume is None:

        raise ValueError(
            "Could not read CT volume."
        )

    print(
        "CT shape:",
        ct_volume.shape
    )

    # ------------------------------------------------------------
    # SEG
    # ------------------------------------------------------------

    seg_file = find_seg_file(
        patient_dir
    )

    if seg_file is None:

        raise FileNotFoundError(
            "DICOM SEG not found."
        )

    print(
        "SEG:",
        seg_file
    )

    seg_ds = pydicom.dcmread(
        seg_file
    )

    # ------------------------------------------------------------
    # GTV-1
    # ------------------------------------------------------------

    segment_number = (
        find_gtv1_segment(
            seg_ds
        )
    )

    if segment_number is None:

        raise ValueError(
            "GTV-1 / GTV1 / Neoplasm "
            "not found."
        )

    # ------------------------------------------------------------
    # FRAMES
    # ------------------------------------------------------------

    pixel_array, frame_numbers = (
        extract_gtv1_frames(
            seg_ds,
            segment_number
        )
    )

    # ------------------------------------------------------------
    # ALIGN
    # ------------------------------------------------------------

    binary_mask = (
        align_seg_to_ct(
            seg_ds,
            pixel_array,
            frame_numbers,
            ct_volume,
            ct_datasets
        )
    )

    # ------------------------------------------------------------
    # TUMOR SLICES
    # ------------------------------------------------------------

    tumor_slice_indices = np.where(
        np.any(
            binary_mask,
            axis=(1, 2)
        )
    )[0]

    print(
        "Tumor slices:",
        len(tumor_slice_indices)
    )

    if len(tumor_slice_indices) == 0:

        raise ValueError(
            "No tumor-containing slices."
        )

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------

    patient_output_dir = os.path.join(
        patient_dir,
        "STEP_12_LBP_TEXTURE"
    )

    os.makedirs(
        patient_output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # ALL LBP VALUES
    # ------------------------------------------------------------

    all_lbp_values = []

    slice_results = []

    representative_lbp = None
    representative_slice = None

    middle_position = (
        len(tumor_slice_indices)
        // 2
    )

    # ------------------------------------------------------------
    # PROCESS SLICES
    # ------------------------------------------------------------

    for position, slice_index in enumerate(
        tumor_slice_indices
    ):

        print(
            f"Processing slice: {int(slice_index)}"
        )

        image = ct_volume[
            slice_index
        ]

        slice_mask = binary_mask[
            slice_index
        ]

        lbp_image = calculate_lbp(
            image,
            slice_mask
        )

        valid_lbp = lbp_image[
            lbp_image >= 0
        ]

        if len(valid_lbp) == 0:
            continue

        all_lbp_values.extend(
            valid_lbp.tolist()
        )

        slice_results.append({

            "Patient_ID":
                patient_id,

            "Slice":
                int(slice_index),

            "Number_of_LBP_pixels":
                int(len(valid_lbp)),

            "Mean_LBP":
                float(
                    np.mean(valid_lbp)
                ),

            "Variance_LBP":
                float(
                    np.var(valid_lbp)
                ),

            "Minimum_LBP":
                int(
                    np.min(valid_lbp)
                ),

            "Maximum_LBP":
                int(
                    np.max(valid_lbp)
                )
        })

        if position == middle_position:

            representative_lbp = (
                lbp_image.copy()
            )

            representative_slice = (
                int(slice_index)
            )

    all_lbp_values = np.asarray(
        all_lbp_values,
        dtype=np.int32
    )

    if len(all_lbp_values) == 0:

        raise ValueError(
            "No valid LBP pixels."
        )

    # ------------------------------------------------------------
    # HISTOGRAM
    # ------------------------------------------------------------

    lbp_histogram = np.bincount(
        all_lbp_values,
        minlength=NUMBER_OF_BINS
    )

    lbp_probability = (
        lbp_histogram.astype(
            np.float64
        )
        /
        np.sum(lbp_histogram)
    )

    # ------------------------------------------------------------
    # LBP STATISTICS
    # ------------------------------------------------------------

    lbp_values = np.arange(
        NUMBER_OF_BINS,
        dtype=np.float64
    )

    lbp_mean = np.sum(
        lbp_values
        *
        lbp_probability
    )

    lbp_variance = np.sum(
        (
            lbp_values
            -
            lbp_mean
        ) ** 2
        *
        lbp_probability
    )

    lbp_uniformity = np.sum(
        lbp_probability ** 2
    )

    nonzero_probability = (
        lbp_probability[
            lbp_probability > 0
        ]
    )

    lbp_entropy = -np.sum(
        nonzero_probability
        *
        np.log2(
            nonzero_probability
        )
    )

    # ------------------------------------------------------------
    # SAVE HISTOGRAM CSV
    # ------------------------------------------------------------

    histogram_rows = []

    for code in range(
        NUMBER_OF_BINS
    ):

        histogram_rows.append({

            "Patient_ID":
                patient_id,

            "LBP_Code":
                code,

            "Count":
                int(
                    lbp_histogram[code]
                ),

            "Probability":
                float(
                    lbp_probability[code]
                )
        })

    histogram_csv = os.path.join(
        patient_output_dir,
        "GTV1_LBP_histogram.csv"
    )

    save_csv(
        histogram_csv,
        histogram_rows,
        [
            "Patient_ID",
            "LBP_Code",
            "Count",
            "Probability"
        ]
    )

    # ------------------------------------------------------------
    # SAVE SLICE FEATURES
    # ------------------------------------------------------------

    slice_csv = os.path.join(
        patient_output_dir,
        "GTV1_LBP_slice_features.csv"
    )

    save_csv(
        slice_csv,
        slice_results,
        [
            "Patient_ID",
            "Slice",
            "Number_of_LBP_pixels",
            "Mean_LBP",
            "Variance_LBP",
            "Minimum_LBP",
            "Maximum_LBP"
        ]
    )

    # ------------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------------

    summary_rows = [{

        "Patient_ID":
            patient_id,

        "LBP_Mean":
            float(lbp_mean),

        "LBP_Variance":
            float(lbp_variance),

        "LBP_Uniformity":
            float(lbp_uniformity),

        "LBP_Entropy":
            float(lbp_entropy),

        "Tumor_Voxels":
            int(
                np.sum(binary_mask)
            ),

        "Tumor_Slices":
            int(
                len(tumor_slice_indices)
            ),

        "LBP_Pixels":
            int(
                len(all_lbp_values)
            ),

        "P":
            P,

        "R":
            R
    }]

    summary_csv = os.path.join(
        patient_output_dir,
        "GTV1_LBP_summary.csv"
    )

    save_csv(
        summary_csv,
        summary_rows,
        [
            "Patient_ID",
            "LBP_Mean",
            "LBP_Variance",
            "LBP_Uniformity",
            "LBP_Entropy",
            "Tumor_Voxels",
            "Tumor_Slices",
            "LBP_Pixels",
            "P",
            "R"
        ]
    )

    # ------------------------------------------------------------
    # HISTOGRAM PLOT
    # ------------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        np.arange(
            NUMBER_OF_BINS
        ),
        lbp_probability,
        width=1.0
    )

    plt.xlabel(
        "LBP Code"
    )

    plt.ylabel(
        "Probability"
    )

    plt.title(
        f"GTV-1 LBP Histogram - {patient_id}"
    )

    plt.xlim(
        0,
        255
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    histogram_plot = os.path.join(
        patient_output_dir,
        "01_GTV1_LBP_Histogram.png"
    )

    plt.savefig(
        histogram_plot,
        dpi=300
    )

    plt.close()

    # ------------------------------------------------------------
    # REPRESENTATIVE LBP IMAGE
    # ------------------------------------------------------------

    if representative_lbp is not None:

        plt.figure(
            figsize=(8, 7)
        )

        display_lbp = np.ma.masked_where(
            representative_lbp < 0,
            representative_lbp
        )

        plt.imshow(
            display_lbp,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        plt.colorbar(
            label="LBP Code"
        )

        plt.title(
            f"GTV-1 LBP | "
            f"{patient_id} | "
            f"Slice {representative_slice}"
        )

        plt.xlabel(
            "X"
        )

        plt.ylabel(
            "Y"
        )

        plt.tight_layout()

        lbp_image_path = os.path.join(
            patient_output_dir,
            "02_Representative_LBP.png"
        )

        plt.savefig(
            lbp_image_path,
            dpi=300
        )

        plt.close()

    else:

        lbp_image_path = ""

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    expected_pixels = len(
        all_lbp_values
    )

    histogram_total = int(
        np.sum(lbp_histogram)
    )

    probability_sum = float(
        np.sum(lbp_probability)
    )

    codes_valid = (
        np.min(all_lbp_values) >= 0
        and
        np.max(all_lbp_values) <= 255
    )

    validation_pass = (
        expected_pixels
        ==
        histogram_total
        and
        np.isclose(
            probability_sum,
            1.0
        )
        and
        codes_valid
    )

    # ------------------------------------------------------------
    # VALIDATION REPORT
    # ------------------------------------------------------------

    report_path = os.path.join(
        patient_output_dir,
        "STEP_12_LBP_VALIDATION_REPORT.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 12 - LBP VALIDATION REPORT\n"
        )

        f.write(
            "=" * 75 + "\n\n"
        )

        f.write(
            f"Patient: {patient_id}\n"
        )

        f.write(
            f"GTV-1 Segment Number: "
            f"{segment_number}\n\n"
        )

        f.write(
            "LBP PARAMETERS\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        f.write(
            f"P = {P}\n"
        )

        f.write(
            f"R = {R}\n\n"
        )

        f.write(
            "RESULTS\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        f.write(
            f"Tumor voxels: "
            f"{int(np.sum(binary_mask))}\n"
        )

        f.write(
            f"Tumor slices: "
            f"{len(tumor_slice_indices)}\n"
        )

        f.write(
            f"LBP pixels: "
            f"{expected_pixels}\n"
        )

        f.write(
            f"Histogram total: "
            f"{histogram_total}\n"
        )

        f.write(
            f"Probability sum: "
            f"{probability_sum:.12f}\n\n"
        )

        f.write(
            f"LBP Mean: "
            f"{lbp_mean:.10f}\n"
        )

        f.write(
            f"LBP Variance: "
            f"{lbp_variance:.10f}\n"
        )

        f.write(
            f"LBP Uniformity: "
            f"{lbp_uniformity:.10f}\n"
        )

        f.write(
            f"LBP Entropy: "
            f"{lbp_entropy:.10f} bits\n\n"
        )

        f.write(
            "VALIDATION\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        if expected_pixels == histogram_total:

            f.write(
                "PASS - Histogram contains "
                "all LBP pixels.\n"
            )

        else:

            f.write(
                "FAIL - Histogram pixel "
                "count mismatch.\n"
            )

        if np.isclose(
            probability_sum,
            1.0
        ):

            f.write(
                "PASS - Probability sum equals 1.\n"
            )

        else:

            f.write(
                "FAIL - Probability sum is incorrect.\n"
            )

        if codes_valid:

            f.write(
                "PASS - LBP codes are within [0,255].\n"
            )

        else:

            f.write(
                "FAIL - Invalid LBP code detected.\n"
            )

        f.write("\n")

        f.write(
            "FORMULA\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        f.write(
            "LBP = sum[s(gp - gc) * 2^p]\n"
        )

        f.write(
            "s(x) = 1 if x >= 0, otherwise 0.\n"
        )

    # ------------------------------------------------------------
    # RETURN RESULT
    # ------------------------------------------------------------

    return {

        "Patient_ID":
            patient_id,

        "Status":
            "SUCCESS",

        "CT_Directory":
            ct_dir,

        "SEG_File":
            seg_file,

        "GTV1_Segment_Number":
            segment_number,

        "Tumor_Voxels":
            int(
                np.sum(binary_mask)
            ),

        "Tumor_Slices":
            int(
                len(tumor_slice_indices)
            ),

        "LBP_Pixels":
            expected_pixels,

        "LBP_Mean":
            float(lbp_mean),

        "LBP_Variance":
            float(lbp_variance),

        "LBP_Uniformity":
            float(lbp_uniformity),

        "LBP_Entropy":
            float(lbp_entropy),

        "Validation":
            "PASS"
            if validation_pass
            else "FAIL",

        "Error":
            ""
    }


# ================================================================
# FIND ALL PATIENTS
# ================================================================

print("\n")
print("=" * 75)
print("FINDING ALL LUNG1 PATIENTS")
print("=" * 75)

patient_dirs = []

for item in os.listdir(
    ROOT_DIR
):

    full_path = os.path.join(
        ROOT_DIR,
        item
    )

    if (
        os.path.isdir(full_path)
        and
        item.upper().startswith("LUNG1-")
    ):

        patient_dirs.append(
            full_path
        )

patient_dirs.sort()

print(
    "Patients found:",
    len(patient_dirs)
)

if len(patient_dirs) == 0:

    raise FileNotFoundError(
        "No LUNG1 patient folders found."
    )


# ================================================================
# PROCESS ALL PATIENTS
# ================================================================

all_results = []

successful = 0
failed = 0

print("\n")
print("=" * 75)
print("PROCESSING ALL PATIENTS")
print("=" * 75)

for index, patient_dir in enumerate(
    patient_dirs,
    start=1
):

    patient_id = os.path.basename(
        patient_dir
    )

    print("\n")
    print(
        f"[{index}/{len(patient_dirs)}] "
        f"{patient_id}"
    )

    try:

        result = process_patient(
            patient_dir
        )

        all_results.append(
            result
        )

        successful += 1

        print(
            f"STATUS: {patient_id} SUCCESS"
        )

    except Exception as e:

        failed += 1

        error_message = str(e)

        print(
            f"STATUS: {patient_id} FAILED"
        )

        print(
            "Reason:",
            error_message
        )

        all_results.append({

            "Patient_ID":
                patient_id,

            "Status":
                "FAILED",

            "CT_Directory":
                "",

            "SEG_File":
                "",

            "GTV1_Segment_Number":
                "",

            "Tumor_Voxels":
                0,

            "Tumor_Slices":
                0,

            "LBP_Pixels":
                0,

            "LBP_Mean":
                "",

            "LBP_Variance":
                "",

            "LBP_Uniformity":
                "",

            "LBP_Entropy":
                "",

            "Validation":
                "FAIL",

            "Error":
                error_message
        })


# ================================================================
# SAVE ALL PATIENT RESULTS
# ================================================================

print("\n")
print("=" * 75)
print("SAVING MULTI-PATIENT RESULTS")
print("=" * 75)

all_patients_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_12_All_Patients_LBP_Features.csv"
)

all_fields = [

    "Patient_ID",
    "Status",
    "CT_Directory",
    "SEG_File",
    "GTV1_Segment_Number",
    "Tumor_Voxels",
    "Tumor_Slices",
    "LBP_Pixels",
    "LBP_Mean",
    "LBP_Variance",
    "LBP_Uniformity",
    "LBP_Entropy",
    "Validation",
    "Error"
]

save_csv(
    all_patients_csv,
    all_results,
    all_fields
)

print(
    "Saved:",
    all_patients_csv
)


# ================================================================
# SAVE SUCCESSFUL PATIENTS
# ================================================================

successful_results = [

    result
    for result in all_results

    if result["Status"]
    == "SUCCESS"
]

successful_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_12_Successful_Patients_LBP_Features.csv"
)

save_csv(
    successful_csv,
    successful_results,
    all_fields
)

print(
    "Saved:",
    successful_csv
)


# ================================================================
# SAVE PROCESSING STATUS
# ================================================================

status_rows = []

for result in all_results:

    status_rows.append({

        "Patient_ID":
            result["Patient_ID"],

        "Status":
            result["Status"],

        "Tumor_Voxels":
            result["Tumor_Voxels"],

        "Tumor_Slices":
            result["Tumor_Slices"],

        "LBP_Pixels":
            result["LBP_Pixels"],

        "LBP_Mean":
            result["LBP_Mean"],

        "LBP_Variance":
            result["LBP_Variance"],

        "LBP_Uniformity":
            result["LBP_Uniformity"],

        "LBP_Entropy":
            result["LBP_Entropy"],

        "Validation":
            result["Validation"],

        "Error":
            result["Error"]
    })


status_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_12_Patient_Processing_Status.csv"
)

save_csv(
    status_csv,
    status_rows,
    [
        "Patient_ID",
        "Status",
        "Tumor_Voxels",
        "Tumor_Slices",
        "LBP_Pixels",
        "LBP_Mean",
        "LBP_Variance",
        "LBP_Uniformity",
        "LBP_Entropy",
        "Validation",
        "Error"
    ]
)

print(
    "Saved:",
    status_csv
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 75)
print("STEP 12 - LBP MULTI-PATIENT COMPLETE")
print("=" * 75)

print(
    "\nTotal patients:",
    len(patient_dirs)
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Success rate:",
    f"{100 * successful / len(patient_dirs):.2f}%"
)

print("\nLBP DESIGN")
print("-" * 75)

print(
    "Method:",
    "LBP from scratch"
)

print(
    "Neighbors:",
    P
)

print(
    "Radius:",
    R
)

print(
    "Histogram bins:",
    NUMBER_OF_BINS
)

print("\nOUTPUT")
print("-" * 75)

print(
    OUTPUT_DIR
)

print("\n")
print("=" * 75)
print("SUCCESS - STEP 12 FINISHED")
print("=" * 75)

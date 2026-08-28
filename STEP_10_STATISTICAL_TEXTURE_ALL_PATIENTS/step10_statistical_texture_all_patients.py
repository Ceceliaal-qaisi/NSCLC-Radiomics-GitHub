# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 10 - STATISTICAL TEXTURE DESCRIPTORS
# Chapter 11 - Digital Image Processing
# MULTI-PATIENT VERSION
#
# Searches directly for DICOM SEG and extracts:
# GTV-1 / GTV1 / Neoplasm
#
# Features:
# 1. Mean
# 2. Variance
# 3. Smoothness
# 4. Third Moment
# 5. Uniformity
# 6. Entropy
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
    "STEP_10_STATISTICAL_TEXTURE_ALL_PATIENTS"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# SETTINGS
# ================================================================

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
print("STEP 10 - STATISTICAL TEXTURE DESCRIPTORS")
print("MULTI-PATIENT DICOM SEG VERSION")
print("=" * 75)

print("\nRoot directory:")
print(ROOT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


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

            if not hasattr(ds, "pixel_array"):
                continue

            if hasattr(ds, "ImagePositionPatient"):

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

            ct_slices.append(
                (z, ds)
            )

        except Exception:
            continue

    if len(ct_slices) == 0:
        return None

    ct_slices.sort(
        key=lambda x: x[0]
    )

    ct_volume = []

    for z, ds in ct_slices:

        image = ds.pixel_array.astype(
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

        image_hu = (
            image * slope
            + intercept
        )

        ct_volume.append(image_hu)

    ct_volume = np.stack(
        ct_volume,
        axis=0
    )

    return ct_volume


# ================================================================
# FUNCTION 3 - FIND DICOM SEG
# ================================================================

def find_seg_file(patient_dir):

    possible_seg_files = []

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

                    possible_seg_files.append(
                        full_path
                    )

            except Exception:
                continue

    if len(possible_seg_files) == 0:
        return None

    return possible_seg_files[0]


# ================================================================
# FUNCTION 4 - FIND GTV-1 SEGMENT
# ================================================================

def find_gtv1_segment(seg_ds):

    if not hasattr(
        seg_ds,
        "SegmentSequence"
    ):
        return None

    print("\nAvailable SEGMENTS:")

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
            f"  Segment {segment_number}: "
            f"Label='{label}' | "
            f"Description='{description}'"
        )

    # ------------------------------------------------------------
    # Search GTV-1
    # ------------------------------------------------------------

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

        combined_text = (
            label
            + " "
            + description
        )

        for keyword in GTV_KEYWORDS:

            if keyword in combined_text:

                print("\nGTV-1 FOUND:")

                print(
                    "Segment Number:",
                    segment.SegmentNumber
                )

                print(
                    "Segment Label:",
                    label
                )

                print(
                    "Segment Description:",
                    description
                )

                return int(
                    segment.SegmentNumber
                )

    return None


# ================================================================
# FUNCTION 5 - EXTRACT SEGMENT MASK
# ================================================================

def extract_segment_mask(
    seg_file,
    segment_number
):

    seg_ds = pydicom.dcmread(
        seg_file
    )

    pixel_array = seg_ds.pixel_array

    print(
        "\nSEG pixel array shape:",
        pixel_array.shape
    )

    number_of_frames = int(
        getattr(
            seg_ds,
            "NumberOfFrames",
            pixel_array.shape[0]
        )
    )

    frame_numbers = []

    if not hasattr(
        seg_ds,
        "PerFrameFunctionalGroupsSequence"
    ):

        raise ValueError(
            "SEG does not contain "
            "PerFrameFunctionalGroupsSequence."
        )

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
        "GTV-1 SEG frames:",
        len(frame_numbers)
    )

    if len(frame_numbers) == 0:

        raise ValueError(
            "No frames found for GTV-1."
        )

    rows = int(
        seg_ds.Rows
    )

    columns = int(
        seg_ds.Columns
    )

    seg_mask = np.zeros(
        (
            number_of_frames,
            rows,
            columns
        ),
        dtype=np.uint8
    )

    for frame_index in frame_numbers:

        frame = pixel_array[
            frame_index
        ]

        seg_mask[
            frame_index
        ] = (
            frame > 0
        ).astype(
            np.uint8
        )

    return (
        seg_mask,
        seg_ds,
        frame_numbers
    )


# ================================================================
# FUNCTION 6 - ALIGN SEG MASK TO CT
# ================================================================

def align_seg_mask_to_ct(
    seg_mask,
    seg_ds,
    ct_dir,
    ct_volume
):

    ct_shape = ct_volume.shape

    ct_files = glob.glob(
        os.path.join(
            ct_dir,
            "*.dcm"
        )
    )

    ct_datasets = []

    for file in ct_files:

        try:

            ds = pydicom.dcmread(
                file,
                stop_before_pixels=True
            )

            if str(
                getattr(
                    ds,
                    "Modality",
                    ""
                )
            ).upper() != "CT":

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

            ct_datasets.append(
                (
                    z,
                    ds
                )
            )

        except Exception:
            continue

    ct_datasets.sort(
        key=lambda x: x[0]
    )

    if len(ct_datasets) != ct_shape[0]:

        raise ValueError(
            "CT slice count mismatch."
        )

    if (
        seg_mask.shape[1] != ct_shape[1]
        or
        seg_mask.shape[2] != ct_shape[2]
    ):

        raise ValueError(
            "SEG and CT in-plane dimensions "
            "do not match."
        )

    final_mask = np.zeros(
        ct_shape,
        dtype=bool
    )

    if not hasattr(
        seg_ds,
        "PerFrameFunctionalGroupsSequence"
    ):

        raise ValueError(
            "Missing SEG frame information."
        )

    matched_frames = 0

    for frame_index in range(
        seg_mask.shape[0]
    ):

        if not np.any(
            seg_mask[frame_index]
        ):
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

        distances = []

        for ct_index, (
            ct_z,
            ct_ds
        ) in enumerate(
            ct_datasets
        ):

            distances.append(
                abs(
                    ct_z - seg_z
                )
            )

        closest_ct_index = int(
            np.argmin(
                distances
            )
        )

        final_mask[
            closest_ct_index
        ] |= (
            seg_mask[
                frame_index
            ].astype(bool)
        )

        matched_frames += 1

    print(
        "Matched SEG frames:",
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
# FUNCTION 7 - CALCULATE STATISTICAL FEATURES
# ================================================================

def calculate_statistical_features(
    ct_volume,
    binary_mask
):

    tumor_intensities = (
        ct_volume[binary_mask]
    )

    if len(
        tumor_intensities
    ) == 0:

        raise ValueError(
            "No tumor intensities found."
        )

    minimum_hu = float(
        np.min(
            tumor_intensities
        )
    )

    maximum_hu = float(
        np.max(
            tumor_intensities
        )
    )

    if maximum_hu == minimum_hu:

        maximum_hu = (
            minimum_hu + 1.0
        )

    # ------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------

    histogram, bin_edges = np.histogram(
        tumor_intensities,
        bins=NUMBER_OF_BINS,
        range=(
            minimum_hu,
            maximum_hu
        )
    )

    histogram = histogram.astype(
        np.float64
    )

    probability = (
        histogram
        /
        np.sum(histogram)
    )

    z = (
        bin_edges[:-1]
        +
        bin_edges[1:]
    ) / 2.0

    # ------------------------------------------------------------
    # Mean
    # ------------------------------------------------------------

    mean_value = np.sum(
        z * probability
    )

    # ------------------------------------------------------------
    # Variance
    # ------------------------------------------------------------

    variance = np.sum(
        (
            z - mean_value
        ) ** 2
        *
        probability
    )

    # ------------------------------------------------------------
    # Normalized Variance
    # ------------------------------------------------------------

    intensity_range = (
        maximum_hu
        - minimum_hu
    )

    if intensity_range == 0:

        normalized_variance = 0.0

    else:

        normalized_variance = (
            variance
            /
            (
                intensity_range
                ** 2
            )
        )

    # ------------------------------------------------------------
    # Smoothness
    # ------------------------------------------------------------

    smoothness = (
        1.0
        -
        (
            1.0
            /
            (
                1.0
                +
                normalized_variance
            )
        )
    )

    # ------------------------------------------------------------
    # Third Moment
    # ------------------------------------------------------------

    third_moment = np.sum(
        (
            z - mean_value
        ) ** 3
        *
        probability
    )

    # ------------------------------------------------------------
    # Uniformity
    # ------------------------------------------------------------

    uniformity = np.sum(
        probability ** 2
    )

    # ------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------

    nonzero_probability = (
        probability[
            probability > 0
        ]
    )

    entropy = -np.sum(
        nonzero_probability
        *
        np.log2(
            nonzero_probability
        )
    )

    return {

        "Statistical_Mean":
            mean_value,

        "Statistical_Variance":
            variance,

        "Statistical_Smoothness":
            smoothness,

        "Statistical_Third_Moment":
            third_moment,

        "Statistical_Uniformity":
            uniformity,

        "Statistical_Entropy":
            entropy,

        "Tumor_Voxels":
            int(
                np.sum(
                    binary_mask
                )
            ),

        "Minimum_HU":
            minimum_hu,

        "Maximum_HU":
            maximum_hu,

        "Intensity_Range":
            intensity_range
    }


# ================================================================
# FUNCTION 8 - SAVE PATIENT CSV
# ================================================================

def save_patient_csv(
    features,
    patient_output_dir
):

    os.makedirs(
        patient_output_dir,
        exist_ok=True
    )

    csv_file = os.path.join(
        patient_output_dir,
        "GTV1_statistical_texture.csv"
    )

    rows = [

        [
            "Statistical_Mean",
            features[
                "Statistical_Mean"
            ],
            "HU"
        ],

        [
            "Statistical_Variance",
            features[
                "Statistical_Variance"
            ],
            "HU^2"
        ],

        [
            "Statistical_Smoothness",
            features[
                "Statistical_Smoothness"
            ],
            "dimensionless"
        ],

        [
            "Statistical_Third_Moment",
            features[
                "Statistical_Third_Moment"
            ],
            "HU^3"
        ],

        [
            "Statistical_Uniformity",
            features[
                "Statistical_Uniformity"
            ],
            "dimensionless"
        ],

        [
            "Statistical_Entropy",
            features[
                "Statistical_Entropy"
            ],
            "bits"
        ]
    ]

    with open(
        csv_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Feature",
                "Value",
                "Unit"
            ]
        )

        writer.writerows(
            rows
        )

    return csv_file


# ================================================================
# FUNCTION 9 - SAVE HISTOGRAM
# ================================================================

def save_patient_histogram(
    patient_id,
    tumor_intensities,
    patient_output_dir
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        tumor_intensities,
        bins=NUMBER_OF_BINS
    )

    plt.xlabel(
        "CT Intensity (HU)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"GTV-1 Intensity Histogram - {patient_id}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    histogram_file = os.path.join(
        patient_output_dir,
        "01_GTV1_Intensity_Histogram.png"
    )

    plt.tight_layout()

    plt.savefig(
        histogram_file,
        dpi=300
    )

    plt.close()

    return histogram_file


# ================================================================
# STEP 1 - FIND PATIENTS
# ================================================================

print("\n")
print("STEP 1 - FINDING ALL PATIENTS")
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
        and item.upper().startswith(
            "LUNG1-"
        )
    ):

        patient_dirs.append(
            full_path
        )

patient_dirs.sort()

print(
    "Patient directories found:",
    len(patient_dirs)
)

if len(patient_dirs) == 0:

    raise FileNotFoundError(
        "No LUNG1 patient directories found."
    )


# ================================================================
# MULTI-PATIENT PROCESSING
# ================================================================

all_patient_features = []
processing_status = []

successful_patients = 0
failed_patients = 0


# ================================================================
# STEP 2 - PROCESS ALL PATIENTS
# ================================================================

print("\n")
print("STEP 2 - PROCESSING ALL PATIENTS")
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
        f"PROCESSING {patient_id}"
    )

    ct_dir = ""

    try:

        # --------------------------------------------------------
        # Find CT
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Read CT
        # --------------------------------------------------------

        ct_volume = read_ct_volume(
            ct_dir
        )

        if ct_volume is None:

            raise ValueError(
                "Could not read CT volume."
            )

        print(
            "CT shape:",
            ct_volume.shape
        )

        # --------------------------------------------------------
        # Find SEG
        # --------------------------------------------------------

        seg_file = find_seg_file(
            patient_dir
        )

        if seg_file is None:

            raise FileNotFoundError(
                "DICOM SEG file not found."
            )

        print(
            "SEG file:",
            seg_file
        )

        # --------------------------------------------------------
        # Read SEG
        # --------------------------------------------------------

        seg_ds = pydicom.dcmread(
            seg_file,
            stop_before_pixels=False
        )

        print(
            "SEG modality:",
            getattr(
                seg_ds,
                "Modality",
                ""
            )
        )

        # --------------------------------------------------------
        # Find GTV-1
        # --------------------------------------------------------

        segment_number = (
            find_gtv1_segment(
                seg_ds
            )
        )

        if segment_number is None:

            raise ValueError(
                "GTV-1 / GTV1 / Neoplasm "
                "segment not found."
            )

        # --------------------------------------------------------
        # Extract GTV-1 frames
        # --------------------------------------------------------

        (
            seg_mask,
            seg_ds,
            frame_numbers
        ) = extract_segment_mask(
            seg_file,
            segment_number
        )

        # --------------------------------------------------------
        # Align mask to CT
        # --------------------------------------------------------

        binary_mask = (
            align_seg_mask_to_ct(
                seg_mask,
                seg_ds,
                ct_dir,
                ct_volume
            )
        )

        tumor_voxels = int(
            np.sum(binary_mask)
        )

        # --------------------------------------------------------
        # Calculate features
        # --------------------------------------------------------

        features = (
            calculate_statistical_features(
                ct_volume,
                binary_mask
            )
        )

        print(
            "\nStatistical features extracted:"
        )

        print(
            "  Mean:",
            features[
                "Statistical_Mean"
            ]
        )

        print(
            "  Variance:",
            features[
                "Statistical_Variance"
            ]
        )

        print(
            "  Smoothness:",
            features[
                "Statistical_Smoothness"
            ]
        )

        print(
            "  Third Moment:",
            features[
                "Statistical_Third_Moment"
            ]
        )

        print(
            "  Uniformity:",
            features[
                "Statistical_Uniformity"
            ]
        )

        print(
            "  Entropy:",
            features[
                "Statistical_Entropy"
            ]
        )

        # --------------------------------------------------------
        # Patient output
        # --------------------------------------------------------

        patient_output_dir = os.path.join(
            patient_dir,
            "STEP_10_STATISTICAL_TEXTURE"
        )

        os.makedirs(
            patient_output_dir,
            exist_ok=True
        )

        # --------------------------------------------------------
        # Save CSV
        # --------------------------------------------------------

        patient_csv = save_patient_csv(
            features,
            patient_output_dir
        )

        print(
            "Patient CSV saved:",
            patient_csv
        )

        # --------------------------------------------------------
        # Histogram
        # --------------------------------------------------------

        tumor_intensities = (
            ct_volume[
                binary_mask
            ]
        )

        histogram_file = (
            save_patient_histogram(
                patient_id,
                tumor_intensities,
                patient_output_dir
            )
        )

        print(
            "Histogram saved:",
            histogram_file
        )

        # --------------------------------------------------------
        # Combined record
        # --------------------------------------------------------

        patient_record = {

            "Patient_ID":
                patient_id,

            "Statistical_Mean":
                features[
                    "Statistical_Mean"
                ],

            "Statistical_Variance":
                features[
                    "Statistical_Variance"
                ],

            "Statistical_Smoothness":
                features[
                    "Statistical_Smoothness"
                ],

            "Statistical_Third_Moment":
                features[
                    "Statistical_Third_Moment"
                ],

            "Statistical_Uniformity":
                features[
                    "Statistical_Uniformity"
                ],

            "Statistical_Entropy":
                features[
                    "Statistical_Entropy"
                ]
        }

        all_patient_features.append(
            patient_record
        )

        processing_status.append({

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
                tumor_voxels,

            "Error":
                ""
        })

        successful_patients += 1

    except Exception as e:

        print(
            "FAILED:",
            str(e)
        )

        processing_status.append({

            "Patient_ID":
                patient_id,

            "Status":
                "FAILED",

            "CT_Directory":
                ct_dir,

            "SEG_File":
                "",

            "GTV1_Segment_Number":
                "",

            "Tumor_Voxels":
                0,

            "Error":
                str(e)
        })

        failed_patients += 1


# ================================================================
# STEP 3 - SAVE COMBINED DATASET
# ================================================================

print("\n")
print(
    "STEP 3 - SAVING MULTI-PATIENT DATASET"
)
print("=" * 75)

combined_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_10_All_Patients_Statistical_Features.csv"
)

fieldnames = [

    "Patient_ID",

    "Statistical_Mean",

    "Statistical_Variance",

    "Statistical_Smoothness",

    "Statistical_Third_Moment",

    "Statistical_Uniformity",

    "Statistical_Entropy"
]

with open(
    combined_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for record in all_patient_features:

        writer.writerow(
            record
        )

print(
    "Saved:",
    combined_csv
)


# ================================================================
# STEP 4 - PROCESSING STATUS
# ================================================================

print("\n")
print(
    "STEP 4 - SAVING PROCESSING STATUS"
)
print("=" * 75)

status_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_10_Patient_Processing_Status.csv"
)

status_fields = [

    "Patient_ID",

    "Status",

    "CT_Directory",

    "SEG_File",

    "GTV1_Segment_Number",

    "Tumor_Voxels",

    "Error"
]

with open(
    status_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=status_fields
    )

    writer.writeheader()

    for record in processing_status:

        writer.writerow(
            record
        )

print(
    "Saved:",
    status_csv
)


# ================================================================
# STEP 5 - FEATURE COVERAGE
# ================================================================

print("\n")
print(
    "STEP 5 - FEATURE COVERAGE"
)
print("=" * 75)

coverage_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_10_Feature_Coverage.csv"
)

total_patients = len(
    patient_dirs
)

coverage_rows = []

for feature in fieldnames[1:]:

    count = 0

    for record in all_patient_features:

        value = record.get(
            feature
        )

        if value is not None:

            try:

                if np.isfinite(
                    float(value)
                ):

                    count += 1

            except Exception:
                pass

    percentage = (
        100.0
        * count
        /
        total_patients
        if total_patients > 0
        else 0.0
    )

    coverage_rows.append({

        "Feature":
            feature,

        "Patients_With_Feature":
            count,

        "Total_Patients":
            total_patients,

        "Coverage_Percent":
            percentage
    })

    print(
        f"{feature:35s} "
        f"{count}/{total_patients} "
        f"({percentage:.2f}%)"
    )

with open(
    coverage_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Feature",
            "Patients_With_Feature",
            "Total_Patients",
            "Coverage_Percent"
        ]
    )

    writer.writeheader()

    writer.writerows(
        coverage_rows
    )

print(
    "\nSaved:",
    coverage_csv
)


# ================================================================
# STEP 6 - VALIDATE FEATURE NAMES
# ================================================================

print("\n")
print(
    "STEP 6 - VALIDATING FEATURE NAMES"
)
print("=" * 75)

feature_names = fieldnames[1:]

if len(feature_names) == len(
    set(feature_names)
):

    print(
        "PASS - All feature names are unique."
    )

else:

    print(
        "FAIL - Duplicate feature names found."
    )


# ================================================================
# STEP 7 - VALIDATE FEATURE VALUES
# ================================================================

print("\n")
print(
    "STEP 7 - VALIDATING FEATURE VALUES"
)
print("=" * 75)

missing_values = 0
invalid_values = 0

for record in all_patient_features:

    for feature in feature_names:

        value = record.get(
            feature
        )

        if value is None:

            missing_values += 1

        else:

            try:

                if not np.isfinite(
                    float(value)
                ):

                    invalid_values += 1

            except Exception:

                invalid_values += 1


if missing_values == 0:

    print(
        "PASS - No missing feature values "
        "among successful patients."
    )

else:

    print(
        "WARNING - Missing values:",
        missing_values
    )


if invalid_values == 0:

    print(
        "PASS - No invalid numerical values."
    )

else:

    print(
        "WARNING - Invalid values:",
        invalid_values
    )


# ================================================================
# STEP 8 - SAVE REPORT
# ================================================================

print("\n")
print(
    "STEP 8 - SAVING REPORT"
)
print("=" * 75)

report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_10_MultiPatient_Statistical_Texture_Report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 10 - STATISTICAL TEXTURE DESCRIPTORS\n"
    )

    f.write(
        "MULTI-PATIENT DICOM SEG PROCESSING REPORT\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "Root directory:\n"
    )

    f.write(
        f"{ROOT_DIR}\n\n"
    )

    f.write(
        f"Total patient directories: "
        f"{total_patients}\n"
    )

    f.write(
        f"Successful patients: "
        f"{successful_patients}\n"
    )

    f.write(
        f"Failed patients: "
        f"{failed_patients}\n\n"
    )

    f.write(
        "GTV-1 SEGMENT SEARCH KEYWORDS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    for keyword in GTV_KEYWORDS:

        f.write(
            f"{keyword}\n"
        )

    f.write(
        "\nFEATURES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    for feature in feature_names:

        f.write(
            f"{feature}\n"
        )

    f.write(
        "\nFORMULAS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "Mean:\n"
        "m = sum[z_i * p(z_i)]\n\n"
    )

    f.write(
        "Variance:\n"
        "sigma^2 = "
        "sum[(z_i - m)^2 * p(z_i)]\n\n"
    )

    f.write(
        "Normalized variance:\n"
        "sigma_norm^2 = "
        "sigma^2 / (intensity_range)^2\n\n"
    )

    f.write(
        "Smoothness:\n"
        "R = 1 - 1/(1 + sigma_norm^2)\n\n"
    )

    f.write(
        "Third moment:\n"
        "mu_3 = "
        "sum[(z_i - m)^3 * p(z_i)]\n\n"
    )

    f.write(
        "Uniformity:\n"
        "U = sum[p(z_i)^2]\n\n"
    )

    f.write(
        "Entropy:\n"
        "e = "
        "-sum[p(z_i) * log2(p(z_i))]\n\n"
    )

    f.write(
        "OUTPUT FILES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        f"Combined dataset:\n"
        f"{combined_csv}\n\n"
    )

    f.write(
        f"Processing status:\n"
        f"{status_csv}\n\n"
    )

    f.write(
        f"Feature coverage:\n"
        f"{coverage_csv}\n\n"
    )

    f.write(
        f"Report:\n"
        f"{report_file}\n"
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 75)
print(
    "STEP 10 - MULTI-PATIENT "
    "STATISTICAL TEXTURE EXTRACTION COMPLETE"
)
print("=" * 75)

print("\nSUMMARY")
print("-" * 75)

print(
    "Total patients found:",
    total_patients
)

print(
    "Successful patients:",
    successful_patients
)

print(
    "Failed patients:",
    failed_patients
)

print(
    "Features per successful patient:",
    len(feature_names)
)

print("\nFINAL FEATURES")
print("-" * 75)

for feature in feature_names:

    print(
        "-",
        feature
    )


print("\nFILES")
print("-" * 75)

print(
    "\nCombined dataset:"
)

print(
    combined_csv
)

print(
    "\nProcessing status:"
)

print(
    status_csv
)

print(
    "\nFeature coverage:"
)

print(
    coverage_csv
)

print(
    "\nReport:"
)

print(
    report_file
)

print("\n")
print("=" * 75)
print("SUCCESS")
print("=" * 75)
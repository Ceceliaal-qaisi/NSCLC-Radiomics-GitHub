
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 11 - GLCM TEXTURE DESCRIPTORS
# ================================================================
#
# MULTI-PATIENT VERSION
#
# IMPORTANT:
# - Does NOT require GTV1_MASK.nrrd
# - Finds DICOM SEG automatically
# - Finds GTV-1 / GTV1 / Neoplasm automatically
# - Builds GLCM FROM SCRATCH
# - 2D slice-based GLCM
# - Distances: 1, 2, 3
# - Angles: 0, 45, 90, 135
# - Quantization: 32 gray levels
# - If one patient fails, processing continues
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
    "STEP_11_GLCM_TEXTURE_ALL_PATIENTS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ================================================================
# SETTINGS
# ================================================================

NUMBER_OF_LEVELS = 32

DISTANCES = [1, 2, 3]

ANGLES = [0, 45, 90, 135]

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
print("STEP 11 - GLCM TEXTURE DESCRIPTORS")
print("MULTI-PATIENT DICOM SEG VERSION")
print("=" * 75)

print("\nROOT:")
print(ROOT_DIR)

print("\nOUTPUT:")
print(OUTPUT_DIR)

# ================================================================
# FUNCTION 1
# FIND CT DIRECTORY
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
# FUNCTION 2
# READ CT VOLUME
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

            ct_slices.append(
                (z, ds)
            )

        except Exception:
            continue

    if len(ct_slices) == 0:
        return None, []

    ct_slices.sort(
        key=lambda x: x[0]
    )

    ct_volume = []

    ct_datasets = []

    for z, ds in ct_slices:

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

        ct_volume.append(
            image_hu
        )

        ct_datasets.append(
            (z, ds)
        )

    ct_volume = np.stack(
        ct_volume,
        axis=0
    )

    return ct_volume, ct_datasets


# ================================================================
# FUNCTION 3
# FIND DICOM SEG
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
# FUNCTION 4
# FIND GTV-1 SEGMENT
# ================================================================

def find_gtv1_segment(seg_ds):

    if not hasattr(
        seg_ds,
        "SegmentSequence"
    ):

        return None

    print("\nAvailable SEGMENTS:")

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

    # ------------------------------------------------------------
    # SEARCH GTV-1
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
# FUNCTION 5
# EXTRACT GTV-1 FRAMES
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
# FUNCTION 6
# ALIGN GTV-1 SEG TO CT
# ================================================================

def align_seg_to_ct(
    seg_ds,
    pixel_array,
    frame_numbers,
    ct_volume,
    ct_datasets
):

    ct_shape = ct_volume.shape

    final_mask = np.zeros(
        ct_shape,
        dtype=bool
    )

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

        # --------------------------------------------------------
        # Find closest CT slice
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Dimensions
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Add frame
        # --------------------------------------------------------

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
# FUNCTION 7
# QUANTIZE IMAGE
# ================================================================

def quantize_image(
    image,
    minimum,
    maximum
):

    if maximum <= minimum:

        return np.zeros(
            image.shape,
            dtype=np.uint8
        )

    scaled = (
        (image - minimum)
        /
        (maximum - minimum)
    )

    quantized = np.floor(
        scaled
        * NUMBER_OF_LEVELS
    ).astype(
        np.int32
    )

    quantized[
        quantized < 0
    ] = 0

    quantized[
        quantized >= NUMBER_OF_LEVELS
    ] = (
        NUMBER_OF_LEVELS - 1
    )

    return quantized.astype(
        np.uint8
    )


# ================================================================
# FUNCTION 8
# BUILD GLCM FROM SCRATCH
# ================================================================

def build_glcm(
    image,
    mask,
    distance,
    angle
):

    rows, cols = image.shape

    glcm = np.zeros(
        (
            NUMBER_OF_LEVELS,
            NUMBER_OF_LEVELS
        ),
        dtype=np.float64
    )

    # ------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------

    if angle == 0:

        dr = 0
        dc = distance

    elif angle == 45:

        dr = -distance
        dc = distance

    elif angle == 90:

        dr = -distance
        dc = 0

    elif angle == 135:

        dr = -distance
        dc = -distance

    else:

        raise ValueError(
            "Unsupported angle."
        )

    # ------------------------------------------------------------
    # LOOP THROUGH PIXELS
    # ------------------------------------------------------------

    for r in range(rows):

        nr = r + dr

        if nr < 0 or nr >= rows:
            continue

        for c in range(cols):

            if not mask[r, c]:
                continue

            nc = c + dc

            if nc < 0 or nc >= cols:
                continue

            if not mask[nr, nc]:
                continue

            i = int(
                image[r, c]
            )

            j = int(
                image[nr, nc]
            )

            glcm[i, j] += 1.0

    # ------------------------------------------------------------
    # MAKE SYMMETRIC
    # ------------------------------------------------------------

    glcm = glcm + glcm.T

    total = np.sum(glcm)

    if total > 0:

        glcm /= total

    return glcm


# ================================================================
# FUNCTION 9
# GLCM FEATURES
# ================================================================

def calculate_glcm_features(
    glcm
):

    levels = np.arange(
        NUMBER_OF_LEVELS,
        dtype=np.float64
    )

    i, j = np.meshgrid(
        levels,
        levels,
        indexing="ij"
    )

    # ------------------------------------------------------------
    # Contrast
    # ------------------------------------------------------------

    contrast = np.sum(
        (
            i - j
        ) ** 2
        * glcm
    )

    # ------------------------------------------------------------
    # Means
    # ------------------------------------------------------------

    mean_i = np.sum(
        i * glcm
    )

    mean_j = np.sum(
        j * glcm
    )

    # ------------------------------------------------------------
    # Standard deviations
    # ------------------------------------------------------------

    variance_i = np.sum(
        (
            i - mean_i
        ) ** 2
        * glcm
    )

    variance_j = np.sum(
        (
            j - mean_j
        ) ** 2
        * glcm
    )

    std_i = np.sqrt(
        variance_i
    )

    std_j = np.sqrt(
        variance_j
    )

    # ------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------

    if (
        std_i > 0
        and
        std_j > 0
    ):

        correlation = np.sum(
            (
                (i - mean_i)
                *
                (j - mean_j)
                *
                glcm
            )
        ) / (
            std_i * std_j
        )

    else:

        correlation = 1.0

    # ------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------

    energy = np.sum(
        glcm ** 2
    )

    # ------------------------------------------------------------
    # Homogeneity
    # ------------------------------------------------------------

    homogeneity = np.sum(
        glcm
        /
        (
            1.0
            +
            np.abs(i - j)
        )
    )

    # ------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------

    nonzero = (
        glcm[
            glcm > 0
        ]
    )

    if len(nonzero) > 0:

        entropy = -np.sum(
            nonzero
            *
            np.log2(nonzero)
        )

    else:

        entropy = 0.0

    # ------------------------------------------------------------
    # Maximum Probability
    # ------------------------------------------------------------

    maximum_probability = np.max(
        glcm
    )

    return {

        "Contrast":
            float(contrast),

        "Correlation":
            float(correlation),

        "Energy":
            float(energy),

        "Homogeneity":
            float(homogeneity),

        "Entropy":
            float(entropy),

        "Maximum_Probability":
            float(maximum_probability)
    }


# ================================================================
# FUNCTION 10
# SAVE DETAILED CSV
# ================================================================

# ================================================================
# FUNCTION 10
# SAVE CSV - FIXED
# ================================================================

def save_csv(filename, rows, fieldnames):

    if len(rows) == 0:
        return

    # Make sure ALL keys appearing in rows exist in fieldnames
    all_keys = set(fieldnames)

    for row in rows:
        all_keys.update(row.keys())

    # Keep requested fieldnames first
    final_fieldnames = list(fieldnames)

    # Add any missing fields automatically
    for key in all_keys:
        if key not in final_fieldnames:
            final_fieldnames.append(key)

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=final_fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)
# ================================================================
# FUNCTION 11
# PROCESS ONE PATIENT
# ================================================================

def process_patient(
    patient_dir
):

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
    # TUMOR INTENSITIES
    # ------------------------------------------------------------

    tumor_intensities = (
        ct_volume[
            binary_mask
        ]
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

    print(
        "HU range:",
        minimum_hu,
        "to",
        maximum_hu
    )

    print(
        "Quantization levels:",
        NUMBER_OF_LEVELS
    )

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------

    patient_output_dir = os.path.join(
        patient_dir,
        "STEP_11_GLCM_TEXTURE"
    )

    os.makedirs(
        patient_output_dir,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # DETAILED MEASUREMENTS
    # ------------------------------------------------------------

    detailed_rows = []

    tumor_slices = np.where(
        np.any(
            binary_mask,
            axis=(1, 2)
        )
    )[0]

    print(
        "Tumor slices:",
        len(tumor_slices)
    )

    # ------------------------------------------------------------
    # PROCESS SLICES
    # ------------------------------------------------------------

    for slice_index in tumor_slices:

        print(
            "Processing slice:",
            int(slice_index)
        )

        slice_ct = (
            ct_volume[
                slice_index
            ]
        )

        slice_mask = (
            binary_mask[
                slice_index
            ]
        )

        # --------------------------------------------------------
        # Quantize using GLOBAL tumor HU range
        # --------------------------------------------------------

        quantized = quantize_image(
            slice_ct,
            minimum_hu,
            maximum_hu
        )

        # --------------------------------------------------------
        # GLCM
        # --------------------------------------------------------

        for distance in DISTANCES:

            for angle in ANGLES:

                glcm = build_glcm(
                    quantized,
                    slice_mask,
                    distance,
                    angle
                )

                # ------------------------------------------------
                # Empty GLCM
                # ------------------------------------------------

                if np.sum(glcm) == 0:

                    continue

                features = (
                    calculate_glcm_features(
                        glcm
                    )
                )

                row = {

                    "Patient_ID":
                        patient_id,

                    "Slice":
                        int(slice_index),

                    "Distance":
                        int(distance),

                    "Angle":
                        int(angle),

                    "Contrast":
                        features[
                            "Contrast"
                        ],

                    "Correlation":
                        features[
                            "Correlation"
                        ],

                    "Energy":
                        features[
                            "Energy"
                        ],

                    "Homogeneity":
                        features[
                            "Homogeneity"
                        ],

                    "Entropy":
                        features[
                            "Entropy"
                        ],

                    "Maximum_Probability":
                        features[
                            "Maximum_Probability"
                        ]
                }

                detailed_rows.append(
                    row
                )

    # ------------------------------------------------------------
    # EXPECTED COUNT
    # ------------------------------------------------------------

    expected_measurements = (
        len(tumor_slices)
        *
        len(DISTANCES)
        *
        len(ANGLES)
    )

    actual_measurements = len(
        detailed_rows
    )

    print(
        "Expected GLCM measurements:",
        expected_measurements
    )

    print(
        "Actual GLCM measurements:",
        actual_measurements
    )

    if actual_measurements == 0:

        raise ValueError(
            "No valid GLCM measurements."
        )

    # ------------------------------------------------------------
    # FIELDNAMES
    # ------------------------------------------------------------

    detailed_fields = [

        "Patient_ID",
        "Slice",
        "Distance",
        "Angle",
        "Contrast",
        "Correlation",
        "Energy",
        "Homogeneity",
        "Entropy",
        "Maximum_Probability"
    ]

    # ------------------------------------------------------------
    # SAVE DETAILED CSV
    # ------------------------------------------------------------

    detailed_csv = os.path.join(
        patient_output_dir,
        "GTV1_GLCM_texture.csv"
    )

    save_csv(
        detailed_csv,
        detailed_rows,
        detailed_fields
    )

    # ------------------------------------------------------------
    # OVERALL FEATURES
    # ------------------------------------------------------------

    feature_names = [

        "Contrast",
        "Correlation",
        "Energy",
        "Homogeneity",
        "Entropy",
        "Maximum_Probability"
    ]

    overall_features = {}

    for feature in feature_names:

        values = np.array(
            [
                row[feature]
                for row in detailed_rows
            ],
            dtype=np.float64
        )

        overall_features[
            feature
        ] = float(
            np.mean(values)
        )

    # ------------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------------

    summary_csv = os.path.join(
        patient_output_dir,
        "GTV1_GLCM_summary.csv"
    )

    summary_rows = [

        {

            "Patient_ID":
                patient_id,

            "Contrast":
                overall_features[
                    "Contrast"
                ],

            "Correlation":
                overall_features[
                    "Correlation"
                ],

            "Energy":
                overall_features[
                    "Energy"
                ],

            "Homogeneity":
                overall_features[
                    "Homogeneity"
                ],

            "Entropy":
                overall_features[
                    "Entropy"
                ],

            "Maximum_Probability":
                overall_features[
                    "Maximum_Probability"
                ],

            "Tumor_Voxels":
                int(
                    np.sum(
                        binary_mask
                    )
                ),

            "Tumor_Slices":
                len(tumor_slices),

            "Quantization_Levels":
                NUMBER_OF_LEVELS
        }
    ]

    summary_fields = [

        "Patient_ID",
        "Contrast",
        "Correlation",
        "Energy",
        "Homogeneity",
        "Entropy",
        "Maximum_Probability",
        "Tumor_Voxels",
        "Tumor_Slices",
        "Quantization_Levels"
    ]

    save_csv(
        summary_csv,
        summary_rows,
        summary_fields
    )

    # ------------------------------------------------------------
    # VARIABILITY
    # ------------------------------------------------------------

    variability_rows = []

    for feature in feature_names:

        values = np.array(
            [
                row[feature]
                for row in detailed_rows
            ],
            dtype=np.float64
        )

        variability_rows.append({

            "Patient_ID":
                patient_id,

            "Feature":
                feature,

            "Mean":
                float(
                    np.mean(values)
                ),

            "SD":
                float(
                    np.std(
                        values
                    )
                ),

            "Minimum":
                float(
                    np.min(values)
                ),

            "Maximum":
                float(
                    np.max(values)
                )
        })

    variability_csv = os.path.join(
        patient_output_dir,
        "GTV1_GLCM_Feature_Variability.csv"
    )

    save_csv(
        variability_csv,
        variability_rows,
        [
            "Patient_ID",
            "Feature",
            "Mean",
            "SD",
            "Minimum",
            "Maximum"
        ]
    )

    # ------------------------------------------------------------
    # BY DISTANCE
    # ------------------------------------------------------------

    distance_rows = []

    for distance in DISTANCES:

        rows_distance = [

            row
            for row in detailed_rows
            if row["Distance"] == distance
        ]

        if len(rows_distance) == 0:
            continue

        record = {

            "Patient_ID":
                patient_id,

            "Distance":
                distance
        }

        for feature in feature_names:

            record[
                feature
            ] = float(
                np.mean(
                    [
                        row[feature]
                        for row in rows_distance
                    ]
                )
            )

        distance_rows.append(
            record
        )

    distance_csv = os.path.join(
        patient_output_dir,
        "GTV1_GLCM_Features_By_Distance.csv"
    )

    save_csv(
        distance_csv,
        distance_rows,
        [
            "Patient_ID",
            "Distance"
        ] + feature_names
    )

    # ------------------------------------------------------------
    # BY ORIENTATION
    # ------------------------------------------------------------

    orientation_rows = []

    for angle in ANGLES:

        rows_angle = [

            row
            for row in detailed_rows
            if row["Angle"] == angle
        ]

        if len(rows_angle) == 0:
            continue

        record = {

            "Patient_ID":
                patient_id,

            "Angle":
                angle
        }

        for feature in feature_names:

            record[
                feature
            ] = float(
                np.mean(
                    [
                        row[feature]
                        for row in rows_angle
                    ]
                )
            )

        orientation_rows.append(
            record
        )

    orientation_csv = os.path.join(
        patient_output_dir,
        "GTV1_GLCM_Features_By_Orientation.csv"
    )

    save_csv(
        orientation_csv,
        orientation_rows,
        [
            "Patient_ID",
            "Angle"
        ] + feature_names
    )

    # ------------------------------------------------------------
    # BY DISTANCE + ORIENTATION
    # ------------------------------------------------------------

    combination_rows = []

    for distance in DISTANCES:

        for angle in ANGLES:

            selected = [

                row
                for row in detailed_rows

                if (
                    row["Distance"]
                    == distance
                )

                and

                (
                    row["Angle"]
                    == angle
                )
            ]

            if len(selected) == 0:
                continue

            record = {

                "Patient_ID":
                    patient_id,

                "Distance":
                    distance,

                "Angle":
                    angle
            }

            for feature in feature_names:

                record[
                    feature
                ] = float(
                    np.mean(
                        [
                            row[feature]
                            for row in selected
                        ]
                    )
                )

            combination_rows.append(
                record
            )

    combination_csv = os.path.join(
        patient_output_dir,
        "GTV1_GLCM_Features_By_Distance_And_Orientation.csv"
    )

    save_csv(
        combination_csv,
        combination_rows,
        [
            "Patient_ID",
            "Distance",
            "Angle"
        ] + feature_names
    )

    # ------------------------------------------------------------
    # FEATURE PLOT
    # ------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    values = [

        overall_features[
            feature
        ]

        for feature in feature_names
    ]

    plt.bar(
        feature_names,
        values
    )

    plt.xlabel(
        "GLCM Feature"
    )

    plt.ylabel(
        "Mean Value"
    )

    plt.title(
        f"GLCM Texture Features - {patient_id}"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    feature_plot = os.path.join(
        patient_output_dir,
        "01_GLCM_Texture_Features.png"
    )

    plt.savefig(
        feature_plot,
        dpi=300
    )

    plt.close()

    # ------------------------------------------------------------
    # REPRESENTATIVE GLCM
    # ------------------------------------------------------------

    representative_glcm = build_glcm(

        quantized,

        slice_mask,

        distance=1,

        angle=0
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        representative_glcm,
        interpolation="nearest"
    )

    plt.xlabel(
        "Gray Level j"
    )

    plt.ylabel(
        "Gray Level i"
    )

    plt.title(
        f"Representative GLCM - {patient_id}"
    )

    plt.colorbar(
        label="Probability"
    )

    plt.tight_layout()

    glcm_plot = os.path.join(
        patient_output_dir,
        "02_Representative_GLCM.png"
    )

    plt.savefig(
        glcm_plot,
        dpi=300
    )

    plt.close()

    # ------------------------------------------------------------
    # RETURN PATIENT RESULT
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
                np.sum(
                    binary_mask
                )
            ),

        "Tumor_Slices":
            len(tumor_slices),

        "GLCM_Measurements":
            actual_measurements,

        "Contrast":
            overall_features[
                "Contrast"
            ],

        "Correlation":
            overall_features[
                "Correlation"
            ],

        "Energy":
            overall_features[
                "Energy"
            ],

        "Homogeneity":
            overall_features[
                "Homogeneity"
            ],

        "Entropy":
            overall_features[
                "Entropy"
            ],

        "Maximum_Probability":
            overall_features[
                "Maximum_Probability"
            ],

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
# MULTI-PATIENT PROCESSING
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
        f"[{index}/{len(patient_dirs)}]"
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

        print(
            f"STATUS: {patient_id} FAILED"
        )

        print(
            "Reason:",
            str(e)
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

            "GLCM_Measurements":
                0,

            "Contrast":
                "",

            "Correlation":
                "",

            "Energy":
                "",

            "Homogeneity":
                "",

            "Entropy":
                "",

            "Maximum_Probability":
                "",

            "Error":
                str(e)
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
    "STEP_11_All_Patients_GLCM_Features.csv"
)

all_fields = [

    "Patient_ID",
    "Status",
    "CT_Directory",
    "SEG_File",
    "GTV1_Segment_Number",
    "Tumor_Voxels",
    "Tumor_Slices",
    "GLCM_Measurements",
    "Contrast",
    "Correlation",
    "Energy",
    "Homogeneity",
    "Entropy",
    "Maximum_Probability",
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
# SAVE SUCCESSFUL PATIENT DATA ONLY
# ================================================================

successful_results = [

    result
    for result in all_results

    if result["Status"]
    == "SUCCESS"
]

successful_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_11_Successful_Patients_GLCM_Features.csv"
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

status_csv = os.path.join(
    OUTPUT_DIR,
    "STEP_11_Patient_Processing_Status.csv"
)

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

        "GLCM_Measurements":
            result["GLCM_Measurements"],

        "Error":
            result["Error"]
    })

save_csv(
    status_csv,
    status_rows,
    [
        "Patient_ID",
        "Status",
        "Tumor_Voxels",
        "Tumor_Slices",
        "GLCM_Measurements",
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
print("STEP 11 - GLCM MULTI-PATIENT COMPLETE")
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

print("\n")
print("GLCM DESIGN")
print("-" * 75)

print(
    "Quantization levels:",
    NUMBER_OF_LEVELS
)

print(
    "Distances:",
    DISTANCES
)

print(
    "Angles:",
    ANGLES
)

print(
    "GLCM:",
    "Symmetric"
)

print(
    "Method:",
    "2D slice-based"
)

print("\n")
print("FILES")
print("-" * 75)

print(
    all_patients_csv
)

print(
    successful_csv
)

print(
    status_csv
)

print("\n")
print("=" * 75)
print("SUCCESS - STEP 11 FINISHED")
print("=" * 75)
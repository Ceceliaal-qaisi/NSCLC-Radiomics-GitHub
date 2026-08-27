# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 13 - SPECTRAL TEXTURE - ALL PATIENTS
#
# Fourier-based spectral texture features
#
# Features:
#   - Spectral Energy
#   - Spectral Entropy
#   - Dominant Frequency
#   - Fundamental Period
#   - Dominant Orientation
#   - Radial Mean
#   - Radial Variance
#   - Radial Peak Frequency
#   - Angular Mean
#   - Angular Variance
#   - Angular Peak Orientation
#
# Implemented from scratch using NumPy FFT.
# No skimage texture functions.
# ================================================================

import os
import glob
import math

import numpy as np
import pandas as pd
import pydicom



# ================================================================
# PATHS
# ================================================================

BASE_ROOT = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

OUTPUT_DIR = os.path.join(
    BASE_ROOT,
    "STEP_13_SPECTRAL_TEXTURE_ALL_PATIENTS"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 13 - SPECTRAL TEXTURE - ALL PATIENTS")
print("=" * 75)


# ================================================================
# FEATURE LIST
# ================================================================

feature_columns = [

    "Spectral_Energy",
    "Spectral_Entropy",
    "Dominant_Frequency",
    "Fundamental_Period",
    "Dominant_Orientation",
    "Radial_Mean",
    "Radial_Variance",
    "Radial_Peak_Frequency",
    "Angular_Mean",
    "Angular_Variance",
    "Angular_Peak_Orientation"

]


# ================================================================
# FIND CT DICOM FILES
# ================================================================

def find_ct_files(patient_dir):

    candidate_files = glob.glob(
        os.path.join(
            patient_dir,
            "**",
            "*.dcm"
        ),
        recursive=True
    )

    ct_files = []

    for path in candidate_files:

        try:

            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True
            )

            modality = str(
                getattr(
                    ds,
                    "Modality",
                    ""
                )
            ).upper()

            if modality == "CT":

                ct_files.append(
                    path
                )

        except Exception:

            continue


    return ct_files


# ================================================================
# SORT CT FILES
# ================================================================

def get_instance_number(path):

    try:

        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True
        )

        return int(
            getattr(
                ds,
                "InstanceNumber",
                0
            )
        )

    except Exception:

        return 0


# ================================================================
# READ CT VOLUME
# ================================================================

def read_ct_volume(ct_files):

    ct_files = sorted(
        ct_files,
        key=get_instance_number
    )

    slices = []

    for path in ct_files:

        ds = pydicom.dcmread(
            path
        )

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

        hu = (
            image * slope
            + intercept
        )

        slices.append(
            hu
        )

    return np.stack(
        slices,
        axis=0
    )


# ================================================================
# FIND GTV1 MASK
# ================================================================

def find_gtv1_mask(patient_dir):

    possible_masks = glob.glob(
        os.path.join(
            patient_dir,
            "**",
            "GTV1_MASK.npy"
        ),
        recursive=True
    )

    if len(possible_masks) == 0:

        return None

    return possible_masks[0]


# ================================================================
# READ GTV1 MASK
# ================================================================

def read_gtv1_mask(mask_file, ct_volume):

    mask = np.load(
        mask_file
    )

    mask = np.asarray(
        mask
    )

    # GTV1_MASK.npy was prepared in CT volume order:
    # (Z, Y, X)

    if mask.shape != ct_volume.shape:

        raise ValueError(
            "CT and GTV1 mask dimensions do not match.\n"
            f"CT shape: {ct_volume.shape}\n"
            f"Mask shape: {mask.shape}"
        )

    return (
        mask > 0
    )


# ================================================================
# EXTRACT TUMOR PATCH
# ================================================================

def extract_tumor_patch(
    image,
    tumor_mask
):

    rows, cols = np.where(
        tumor_mask
    )

    if len(rows) == 0:

        return None


    r_min = int(
        np.min(rows)
    )

    r_max = int(
        np.max(rows)
    )

    c_min = int(
        np.min(cols)
    )

    c_max = int(
        np.max(cols)
    )


    patch = image[
        r_min:r_max + 1,
        c_min:c_max + 1
    ]

    patch_mask = tumor_mask[
        r_min:r_max + 1,
        c_min:c_max + 1
    ]


    return (
        patch,
        patch_mask
    )


# ================================================================
# RADIAL SPECTRAL ENERGY
# ================================================================

def calculate_radial_spectral_energy(
    power,
    radial_frequency
):

    max_radius = float(
        np.max(
            radial_frequency
        )
    )

    radial_bins = np.linspace(
        0.0,
        max_radius,
        50
    )

    radial_energy = []
    radial_centers = []


    for i in range(
        len(radial_bins) - 1
    ):

        lower = radial_bins[i]
        upper = radial_bins[i + 1]


        if i == len(radial_bins) - 2:

            region = (
                (radial_frequency >= lower)
                &
                (radial_frequency <= upper)
            )

        else:

            region = (
                (radial_frequency >= lower)
                &
                (radial_frequency < upper)
            )


        energy = float(
            np.sum(
                power[region]
            )
        )


        radial_energy.append(
            energy
        )


        radial_centers.append(
            (lower + upper) / 2.0
        )


    radial_energy = np.asarray(
        radial_energy,
        dtype=np.float64
    )

    radial_centers = np.asarray(
        radial_centers,
        dtype=np.float64
    )


    total = float(
        np.sum(
            radial_energy
        )
    )


    if total > 0:

        radial_probability = (
            radial_energy
            /
            total
        )

    else:

        radial_probability = np.zeros_like(
            radial_energy
        )


    return (
        radial_centers,
        radial_energy,
        radial_probability
    )


# ================================================================
# ANGULAR SPECTRAL ENERGY
# ================================================================

def calculate_angular_spectral_energy(
    power,
    FX,
    FY
):

    orientation = (

        np.degrees(
            np.arctan2(
                FY,
                FX
            )
        )

        % 180.0

    )


    angular_bins = np.linspace(
        0.0,
        180.0,
        37
    )


    angular_energy = []
    angular_centers = []


    for i in range(
        len(angular_bins) - 1
    ):

        lower = angular_bins[i]
        upper = angular_bins[i + 1]


        if i == len(angular_bins) - 2:

            region = (
                (orientation >= lower)
                &
                (orientation <= upper)
            )

        else:

            region = (
                (orientation >= lower)
                &
                (orientation < upper)
            )


        energy = float(
            np.sum(
                power[region]
            )
        )


        angular_energy.append(
            energy
        )


        angular_centers.append(
            (lower + upper) / 2.0
        )


    angular_energy = np.asarray(
        angular_energy,
        dtype=np.float64
    )

    angular_centers = np.asarray(
        angular_centers,
        dtype=np.float64
    )


    total = float(
        np.sum(
            angular_energy
        )
    )


    if total > 0:

        angular_probability = (
            angular_energy
            /
            total
        )

    else:

        angular_probability = np.zeros_like(
            angular_energy
        )


    return (
        angular_centers,
        angular_energy,
        angular_probability
    )


# ================================================================
# FOURIER TEXTURE
# ================================================================

def calculate_fourier_texture(
    image,
    tumor_mask
):

    extracted = extract_tumor_patch(
        image,
        tumor_mask
    )


    if extracted is None:

        return None


    patch, patch_mask = extracted


    tumor_values = patch[
        patch_mask
    ]


    if len(tumor_values) < 2:

        return None


    tumor_mean = float(
        np.mean(
            tumor_values
        )
    )


    analysis_image = np.array(
        patch,
        dtype=np.float64
    )


    # ------------------------------------------------------------
    # Replace background with tumor mean
    # ------------------------------------------------------------

    analysis_image[
        ~patch_mask
    ] = tumor_mean


    # ------------------------------------------------------------
    # Remove mean / DC component
    # ------------------------------------------------------------

    analysis_image = (
        analysis_image
        -
        np.mean(
            analysis_image
        )
    )


    # ------------------------------------------------------------
    # 2-D Fourier transform
    # ------------------------------------------------------------

    spectrum_complex = np.fft.fft2(
        analysis_image
    )


    spectrum_shifted = np.fft.fftshift(
        spectrum_complex
    )


    magnitude = np.abs(
        spectrum_shifted
    )


    # ------------------------------------------------------------
    # Power spectrum
    # ------------------------------------------------------------

    power = (
        magnitude ** 2
    )


    center_r = (
        power.shape[0] // 2
    )

    center_c = (
        power.shape[1] // 2
    )


    # Remove DC
    power[
        center_r,
        center_c
    ] = 0.0


    total_power = float(
        np.sum(
            power
        )
    )


    if total_power <= 0:

        return None


    # ============================================================
    # FREQUENCY COORDINATES
    # ============================================================

    rows, cols = power.shape


    fy = np.fft.fftshift(
        np.fft.fftfreq(
            rows
        )
    )


    fx = np.fft.fftshift(
        np.fft.fftfreq(
            cols
        )
    )


    FX, FY = np.meshgrid(
        fx,
        fy
    )


    radial_frequency = np.sqrt(
        FX ** 2
        +
        FY ** 2
    )


    # ============================================================
    # SPECTRAL PROBABILITY
    # ============================================================

    spectral_probability = (
        power
        /
        total_power
    )


    # ============================================================
    # SPECTRAL ENERGY
    # ============================================================

    spectral_energy = float(
        np.sum(
            spectral_probability ** 2
        )
    )


    # ============================================================
    # SPECTRAL ENTROPY
    # ============================================================

    nonzero_probability = (
        spectral_probability[
            spectral_probability > 0
        ]
    )


    spectral_entropy = float(
        -np.sum(
            nonzero_probability
            *
            np.log2(
                nonzero_probability
            )
        )
    )


    # ============================================================
    # DOMINANT FREQUENCY
    # ============================================================

    max_index = np.unravel_index(
        np.argmax(
            power
        ),
        power.shape
    )


    dominant_frequency = float(
        radial_frequency[
            max_index
        ]
    )


    # ============================================================
    # FUNDAMENTAL PERIOD
    # ============================================================

    if dominant_frequency > 0:

        fundamental_period = float(
            1.0
            /
            dominant_frequency
        )

    else:

        fundamental_period = np.nan


    # ============================================================
    # DOMINANT ORIENTATION
    # ============================================================

    dominant_fx = float(
        FX[
            max_index
        ]
    )

    dominant_fy = float(
        FY[
            max_index
        ]
    )


    dominant_orientation = (

        math.degrees(
            math.atan2(
                dominant_fy,
                dominant_fx
            )
        )

        % 180.0

    )


    # ============================================================
    # RADIAL SPECTRAL FUNCTION S(r)
    # ============================================================

    (
        radial_centers,
        radial_energy,
        radial_probability

    ) = calculate_radial_spectral_energy(
        power,
        radial_frequency
    )


    # ============================================================
    # ANGULAR SPECTRAL FUNCTION S(u)
    # ============================================================

    (
        angular_centers,
        angular_energy,
        angular_probability

    ) = calculate_angular_spectral_energy(
        power,
        FX,
        FY
    )


    # ============================================================
    # RADIAL MEAN
    # ============================================================

    radial_mean = float(
        np.sum(
            radial_centers
            *
            radial_probability
        )
    )


    # ============================================================
    # RADIAL VARIANCE
    # ============================================================

    radial_variance = float(
        np.sum(
            (
                radial_centers
                -
                radial_mean
            ) ** 2
            *
            radial_probability
        )
    )


    # ============================================================
    # RADIAL PEAK
    # ============================================================

    radial_peak_index = int(
        np.argmax(
            radial_energy
        )
    )


    radial_peak_frequency = float(
        radial_centers[
            radial_peak_index
        ]
    )


    # ============================================================
    # ANGULAR MEAN
    # ============================================================

    angular_mean = float(
        np.sum(
            angular_centers
            *
            angular_probability
        )
    )


    # ============================================================
    # ANGULAR VARIANCE
    # ============================================================

    angular_variance = float(
        np.sum(
            (
                angular_centers
                -
                angular_mean
            ) ** 2
            *
            angular_probability
        )
    )


    # ============================================================
    # ANGULAR PEAK
    # ============================================================

    angular_peak_index = int(
        np.argmax(
            angular_energy
        )
    )


    angular_peak_orientation = float(
        angular_centers[
            angular_peak_index
        ]
    )


    return {

        "Spectral_Energy":
            spectral_energy,

        "Spectral_Entropy":
            spectral_entropy,

        "Dominant_Frequency":
            dominant_frequency,

        "Fundamental_Period":
            fundamental_period,

        "Dominant_Orientation":
            dominant_orientation,

        "Radial_Mean":
            radial_mean,

        "Radial_Variance":
            radial_variance,

        "Radial_Peak_Frequency":
            radial_peak_frequency,

        "Angular_Mean":
            angular_mean,

        "Angular_Variance":
            angular_variance,

        "Angular_Peak_Orientation":
            angular_peak_orientation

    }


# ================================================================
# PROCESS ONE PATIENT
# ================================================================

def process_patient(
    patient_id
):

    patient_dir = os.path.join(
        BASE_ROOT,
        patient_id
    )


    if not os.path.isdir(
        patient_dir
    ):

        return None, "PATIENT_FOLDER_NOT_FOUND"


    # ------------------------------------------------------------
    # Find mask
    # ------------------------------------------------------------

    mask_file = find_gtv1_mask(
        patient_dir
    )


    if mask_file is None:

        return None, "GTV1_MASK_NOT_FOUND"


    # ------------------------------------------------------------
    # Find CT
    # ------------------------------------------------------------

    ct_files = find_ct_files(
        patient_dir
    )


    if len(ct_files) == 0:

        return None, "CT_NOT_FOUND"


    # ------------------------------------------------------------
    # Read CT
    # ------------------------------------------------------------

    ct_volume = read_ct_volume(
        ct_files
    )


    # ------------------------------------------------------------
    # Read mask
    # ------------------------------------------------------------

    binary_mask = read_gtv1_mask(
        mask_file,
        ct_volume
    )


    tumor_voxels = int(
        np.sum(
            binary_mask
        )
    )


    if tumor_voxels == 0:

        return None, "EMPTY_GTV1_MASK"


    # ------------------------------------------------------------
    # Find tumor slices
    # ------------------------------------------------------------

    tumor_slice_indices = np.where(
        np.any(
            binary_mask,
            axis=(1, 2)
        )
    )[0]


    if len(tumor_slice_indices) == 0:

        return None, "NO_TUMOR_SLICES"


    # ------------------------------------------------------------
    # Process slices
    # ------------------------------------------------------------

    slice_results = []


    for slice_index in tumor_slice_indices:

        image = ct_volume[
            slice_index
        ]

        slice_mask = binary_mask[
            slice_index
        ]


        result = calculate_fourier_texture(
            image,
            slice_mask
        )


        if result is None:

            continue


        row = {

            "Patient_ID":
                patient_id,

            "Slice":
                int(slice_index)

        }


        for feature in feature_columns:

            row[
                feature
            ] = result[
                feature
            ]


        slice_results.append(
            row
        )


    if len(slice_results) == 0:

        return None, "NO_SPECTRAL_RESULTS"


    slice_df = pd.DataFrame(
        slice_results
    )


    # ------------------------------------------------------------
    # Patient-level mean
    # ------------------------------------------------------------

    patient_record = {

        "Patient_ID":
            patient_id,

        "Tumor_Voxels":
            tumor_voxels,

        "Tumor_Slices_Analyzed":
            len(slice_df)

    }


    for feature in feature_columns:

        patient_record[
            feature
        ] = float(
            slice_df[
                feature
            ].mean()
        )


    return (
        patient_record,
        "SUCCESS"
    )


# ================================================================
# MAIN MULTI-PATIENT PROCESSING
# ================================================================

print("\n")
print("=" * 75)
print("PROCESSING ALL PATIENTS")
print("=" * 75)


all_patient_records = []
all_slice_records = []
processing_status = []


# ================================================================
# PATIENT LOOP
# ================================================================

for patient_number in range(
    1,
    423
):

    patient_id = (
        f"LUNG1-{patient_number:03d}"
    )


    print("\n")
    print(
        f"PROCESSING {patient_id}"
    )


    try:

        patient_record, status = process_patient(
            patient_id
        )


        if status == "SUCCESS":

            all_patient_records.append(
                patient_record
            )


            print(
                "SUCCESS"
            )

            print(
                "Spectral features calculated."
            )


            processing_status.append({

                "Patient_ID":
                    patient_id,

                "Status":
                    "SUCCESS",

                "Reason":
                    "Spectral texture calculated"

            })


        else:

            print(
                "FAILED:",
                status
            )


            processing_status.append({

                "Patient_ID":
                    patient_id,

                "Status":
                    "FAILED",

                "Reason":
                    status

            })


    except Exception as e:

        print(
            "ERROR:",
            e
        )


        processing_status.append({

            "Patient_ID":
                patient_id,

            "Status":
                "FAILED",

            "Reason":
                str(e)

        })


# ================================================================
# CREATE DATAFRAMES
# ================================================================

patient_features_df = pd.DataFrame(
    all_patient_records
)


status_df = pd.DataFrame(
    processing_status
)


# ================================================================
# SAVE PATIENT-LEVEL FEATURES
# ================================================================

print("\n")
print("=" * 75)
print("SAVING MULTI-PATIENT RESULTS")
print("=" * 75)


patient_features_file = os.path.join(
    OUTPUT_DIR,
    "STEP_13_All_Patients_Spectral_Features.csv"
)


patient_features_df.to_csv(
    patient_features_file,
    index=False
)


print(
    "Saved:",
    patient_features_file
)


# ================================================================
# SAVE PROCESSING STATUS
# ================================================================

status_file = os.path.join(
    OUTPUT_DIR,
    "STEP_13_Patient_Processing_Status.csv"
)


status_df.to_csv(
    status_file,
    index=False
)


print(
    "Saved:",
    status_file
)


# ================================================================
# FEATURE COVERAGE
# ================================================================

print("\n")
print("=" * 75)
print("FEATURE COVERAGE")
print("=" * 75)


coverage_rows = []


total_patients = 422


for feature in feature_columns:

    if len(
        patient_features_df
    ) == 0:

        available = 0

    else:

        available = int(
            patient_features_df[
                feature
            ]
            .notna()
            .sum()
        )


    coverage = (

        available
        /
        total_patients
        *
        100.0

    )


    coverage_rows.append({

        "Feature":
            feature,

        "Patients_Available":
            available,

        "Total_Patients":
            total_patients,

        "Coverage_Percent":
            coverage

    })


coverage_df = pd.DataFrame(
    coverage_rows
)


coverage_file = os.path.join(
    OUTPUT_DIR,
    "STEP_13_Spectral_Feature_Coverage.csv"
)


coverage_df.to_csv(
    coverage_file,
    index=False
)


print(
    coverage_df.to_string(
        index=False
    )
)


print(
    "\nSaved:",
    coverage_file
)


# ================================================================
# SUCCESSFUL PATIENTS
# ================================================================

successful_patients = status_df[
    status_df[
        "Status"
    ]
    == "SUCCESS"
][
    "Patient_ID"
].tolist()


successful_file = os.path.join(
    OUTPUT_DIR,
    "STEP_13_Successful_Patients_Spectral_Features.csv"
)


patient_features_df.to_csv(
    successful_file,
    index=False
)


print(
    "Saved:",
    successful_file
)


# ================================================================
# SUMMARY
# ================================================================

successful_count = int(
    (
        status_df[
            "Status"
        ]
        == "SUCCESS"
    )
    .sum()
)


failed_count = int(
    (
        status_df[
            "Status"
        ]
        == "FAILED"
    )
    .sum()
)


success_rate = (

    successful_count
    /
    total_patients
    *
    100.0

)


# ================================================================
# SAVE SUMMARY REPORT
# ================================================================

summary_report = os.path.join(
    OUTPUT_DIR,
    "STEP_13_Spectral_Texture_All_Patients_Report.txt"
)


with open(
    summary_report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 13 - SPECTRAL TEXTURE - ALL PATIENTS\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )


    f.write(
        "TOTAL PATIENTS: "
        f"{total_patients}\n"
    )

    f.write(
        "SUCCESSFUL PATIENTS: "
        f"{successful_count}\n"
    )

    f.write(
        "FAILED PATIENTS: "
        f"{failed_count}\n"
    )

    f.write(
        "SUCCESS RATE: "
        f"{success_rate:.2f}%\n\n"
    )


    f.write(
        "SPECTRAL FEATURES\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for feature in feature_columns:

        f.write(
            feature
            +
            "\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "METHOD\n"
    )

    f.write(
        "-" * 75 + "\n"
    )

    f.write(
        "2-D Fourier transform was applied to the "
        "tumor-region image patch.\n"
    )

    f.write(
        "Background pixels were replaced by the "
        "tumor-region mean intensity.\n"
    )

    f.write(
        "The DC component was removed before spectral analysis.\n"
    )

    f.write(
        "The Fourier power spectrum was analyzed using "
        "radial and angular frequency coordinates.\n"
    )

    f.write(
        "Patient-level values were calculated as the mean "
        "of the valid tumor-containing CT slices.\n"
    )

    f.write(
        "No skimage texture functions were used.\n\n"
    )


    f.write(
        "FEATURE COVERAGE\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for _, row in coverage_df.iterrows():

        f.write(
            f"{row['Feature']}: "
            f"{row['Patients_Available']}/"
            f"{row['Total_Patients']} "
            f"({row['Coverage_Percent']:.2f}%)\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "PROCESSING STATUS\n"
    )

    f.write(
        "-" * 75 + "\n"
    )


    for _, row in status_df.iterrows():

        f.write(
            f"{row['Patient_ID']} | "
            f"{row['Status']} | "
            f"{row['Reason']}\n"
        )


print(
    "\nSaved:",
    summary_report
)


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 75)
print("STEP 13 - SPECTRAL TEXTURE MULTI-PATIENT COMPLETE")
print("=" * 75)


print(
    "\nTotal patients:",
    total_patients
)

print(
    "Successful:",
    successful_count
)

print(
    "Failed:",
    failed_count
)

print(
    f"Success rate: {success_rate:.2f}%"
)


print("\n")
print("OUTPUT DIRECTORY:")
print(
    OUTPUT_DIR
)


print("\nFILES:")
print(
    patient_features_file
)

print(
    successful_file
)

print(
    status_file
)

print(
    coverage_file
)

print(
    summary_report
)


print("\n")
print("=" * 75)
print("SUCCESS - STEP 13 SPECTRAL TEXTURE ALL PATIENTS")
print("=" * 75)
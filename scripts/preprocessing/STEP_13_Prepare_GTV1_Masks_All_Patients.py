
# ================================================================
# PROJECT 7 - RADIOMICS
# STEP 13 - PREPARE GTV1 MASKS - ALL PATIENTS
#
# Purpose:
#   Find the DICOM SEG file for every Lung1 patient,
#   identify the GTV-1 / Neoplasm segment,
#   reconstruct the 3D binary tumor mask,
#   and save it as GTV1_MASK.npy.
#
# This step prepares the masks required by:
#   STEP 13 - Spectral Texture - All Patients
#
# ================================================================

import os
import glob
import numpy as np
import pydicom
import re


# ================================================================
# PATH
# ================================================================

BASE_ROOT = r"C:\Users\CeCe\Downloads\nsclc_radiomics"


# ================================================================
# HEADER
# ================================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 13 - PREPARE GTV1 MASKS - ALL PATIENTS")
print("=" * 75)

print("\nBase directory:")
print(BASE_ROOT)


# ================================================================
# CHECK BASE DIRECTORY
# ================================================================

if not os.path.exists(BASE_ROOT):
    raise FileNotFoundError(
        "\nBASE_ROOT was not found:\n" + BASE_ROOT
    )


# ================================================================
# FIND PATIENT DIRECTORIES
# ================================================================

patient_dirs = []

for name in os.listdir(BASE_ROOT):

    path = os.path.join(BASE_ROOT, name)

    if os.path.isdir(path):

        match = re.fullmatch(
            r"LUNG1-(\d+)",
            name.upper()
        )

        if match:
            patient_dirs.append(path)


patient_dirs = sorted(
    patient_dirs,
    key=lambda x: int(
        re.search(
            r"LUNG1-(\d+)$",
            os.path.basename(x),
            re.IGNORECASE
        ).group(1)
    )
)

print("\nPatients found:", len(patient_dirs))
# ================================================================
# RESULTS
# ================================================================

results = []


# ================================================================
# PROCESS PATIENTS
# ================================================================

for patient_dir in patient_dirs:

    patient_id = os.path.basename(patient_dir)

    print("\n")
    print("=" * 75)
    print("PROCESSING", patient_id)
    print("=" * 75)

    try:

        # ------------------------------------------------------------
        # FIND DICOM SEG FILES
        # ------------------------------------------------------------

        seg_candidates = []

        all_dcm = glob.glob(
            os.path.join(
                patient_dir,
                "**",
                "*.dcm"
            ),
            recursive=True
        )

        print("DICOM files found:", len(all_dcm))

        for dcm_path in all_dcm:

            try:

                ds = pydicom.dcmread(
                    dcm_path,
                    stop_before_pixels=False
                )

                if getattr(
                    ds,
                    "Modality",
                    ""
                ) == "SEG":

                    seg_candidates.append(
                        dcm_path
                    )

            except Exception:
                continue


        if len(seg_candidates) == 0:

            print("FAILED: SEG_FILE_NOT_FOUND")

            results.append({
                "Patient_ID": patient_id,
                "Status": "FAILED",
                "Reason": "SEG_FILE_NOT_FOUND"
            })

            continue


        print(
            "SEG files found:",
            len(seg_candidates)
        )


        # ------------------------------------------------------------
        # TRY SEG FILES UNTIL GTV-1 IS FOUND
        # ------------------------------------------------------------

        selected_seg = None
        selected_ds = None
        gtv_segment_number = None

        for seg_path in seg_candidates:

            try:

                ds = pydicom.dcmread(
                    seg_path,
                    stop_before_pixels=False
                )

                segment_sequence = getattr(
                    ds,
                    "SegmentSequence",
                    []
                )

                for segment in segment_sequence:

                    label = str(
                        getattr(
                            segment,
                            "SegmentLabel",
                            ""
                        )
                    ).strip().upper()

                    description = str(
                        getattr(
                            segment,
                            "SegmentDescription",
                            ""
                        )
                    ).strip().upper()

                    number = int(
                        segment.SegmentNumber
                    )

                    text = (
                        label
                        + " "
                        + description
                    )

                    if (
                        "GTV-1" in text
                        or
                        "GTV1" in text
                        or
                        "NEOPLASM" in text
                        or
                        "PRIMARY" in text
                    ):

                        selected_seg = seg_path
                        selected_ds = ds
                        gtv_segment_number = number

                        print(
                            "GTV-1 segment found."
                        )

                        print(
                            "Segment Number:",
                            number
                        )

                        print(
                            "Segment Label:",
                            label
                        )

                        print(
                            "Segment Description:",
                            description
                        )

                        break

                if selected_seg is not None:
                    break

            except Exception as e:

                print(
                    "Could not inspect SEG:",
                    seg_path
                )

                print(
                    "Reason:",
                    e
                )


        if selected_seg is None:

            print(
                "FAILED: GTV1_SEGMENT_NOT_FOUND"
            )

            results.append({
                "Patient_ID": patient_id,
                "Status": "FAILED",
                "Reason": "GTV1_SEGMENT_NOT_FOUND"
            })

            continue


        print(
            "\nSelected SEG:"
        )

        print(
            selected_seg
        )


        # ------------------------------------------------------------
        # READ PIXEL DATA
        # ------------------------------------------------------------

        pixel_array = selected_ds.pixel_array

        print(
            "\nSEG pixel array shape:",
            pixel_array.shape
        )


        # ------------------------------------------------------------
        # SEG FRAME INFORMATION
        # ------------------------------------------------------------

        number_of_frames = int(
            getattr(
                selected_ds,
                "NumberOfFrames",
                pixel_array.shape[0]
            )
        )

        rows = int(
            selected_ds.Rows
        )

        columns = int(
            selected_ds.Columns
        )

        print(
            "Number of frames:",
            number_of_frames
        )

        print(
            "Rows:",
            rows
        )

        print(
            "Columns:",
            columns
        )


        # ------------------------------------------------------------
        # CREATE FRAME-LEVEL GTV MASK
        # ------------------------------------------------------------

        frame_mask = (
            pixel_array > 0
        ).astype(
            np.uint8
        )


        # ------------------------------------------------------------
        # FIND REFERENCED CT INSTANCES
        # ------------------------------------------------------------

        referenced_slices = []

        try:

            per_frame = (
                selected_ds
                .PerFrameFunctionalGroupsSequence
            )

        except Exception:

            per_frame = []


        for frame_index, frame_item in enumerate(
            per_frame
        ):

            frame_segment_number = None

            try:

                frame_segment_number = int(
                    frame_item
                    .SegmentIdentificationSequence[0]
                    .ReferencedSegmentNumber
                )

            except Exception:
                pass


            if (
                frame_segment_number
                != gtv_segment_number
            ):
                continue


            sop_uid = None

            try:

                sop_uid = str(
                    frame_item
                    .DerivationImageSequence[0]
                    .SourceImageSequence[0]
                    .ReferencedSOPInstanceUID
                )

            except Exception:
                pass


            position = None

            try:

                position = (
                    frame_item
                    .PlanePositionSequence[0]
                    .ImagePositionPatient
                )

            except Exception:
                pass


            referenced_slices.append({
                "frame_index": frame_index,
                "sop_uid": sop_uid,
                "position": position
            })


        print(
            "GTV-1 frames:",
            len(referenced_slices)
        )


        if len(referenced_slices) == 0:

            print(
                "FAILED: GTV1_FRAMES_NOT_FOUND"
            )

            results.append({
                "Patient_ID": patient_id,
                "Status": "FAILED",
                "Reason": "GTV1_FRAMES_NOT_FOUND"
            })

            continue


        # ------------------------------------------------------------
        # FIND CT DICOM FILES
        # ------------------------------------------------------------

        ct_files = []

        for dcm_path in all_dcm:

            try:

                ds_ct = pydicom.dcmread(
                    dcm_path,
                    stop_before_pixels=True
                )

                if (
                    getattr(
                        ds_ct,
                        "Modality",
                        ""
                    ) == "CT"
                ):

                    ct_files.append(
                        (
                            dcm_path,
                            ds_ct
                        )
                    )

            except Exception:
                continue


        print(
            "CT files found:",
            len(ct_files)
        )


        if len(ct_files) == 0:

            print(
                "FAILED: CT_FILES_NOT_FOUND"
            )

            results.append({
                "Patient_ID": patient_id,
                "Status": "FAILED",
                "Reason": "CT_FILES_NOT_FOUND"
            })

            continue


        # ------------------------------------------------------------
        # CREATE CT SOP UID -> SLICE INDEX MAP
        # ------------------------------------------------------------

        ct_uid_map = {}

        for index, (
            ct_path,
            ct_ds
        ) in enumerate(ct_files):

            sop_uid = getattr(
                ct_ds,
                "SOPInstanceUID",
                None
            )

            if sop_uid is not None:

                ct_uid_map[
                    str(sop_uid)
                ] = index


        # ------------------------------------------------------------
        # SORT CT FILES BY IMAGE POSITION / INSTANCE NUMBER
        # ------------------------------------------------------------

        def ct_sort_key(item):

            ds = item[1]

            try:

                position = float(
                    ds.ImagePositionPatient[2]
                )

                return (
                    0,
                    position
                )

            except Exception:
                pass

            try:

                return (
                    1,
                    int(
                        ds.InstanceNumber
                    )
                )

            except Exception:

                return (
                    2,
                    item[0]
                )


        ct_files = sorted(
            ct_files,
            key=ct_sort_key
        )


        # ------------------------------------------------------------
        # REBUILD UID MAP AFTER SORTING
        # ------------------------------------------------------------

        ct_uid_map = {}

        for index, (
            ct_path,
            ct_ds
        ) in enumerate(ct_files):

            sop_uid = getattr(
                ct_ds,
                "SOPInstanceUID",
                None
            )

            if sop_uid is not None:

                ct_uid_map[
                    str(sop_uid)
                ] = index


        # ------------------------------------------------------------
        # CREATE 3D MASK
        # ------------------------------------------------------------

        mask = np.zeros(
            (
                len(ct_files),
                rows,
                columns
            ),
            dtype=np.uint8
        )


        matched_frames = 0


        for item in referenced_slices:

            frame_index = item[
                "frame_index"
            ]

            sop_uid = item[
                "sop_uid"
            ]


            if (
                sop_uid is not None
                and sop_uid in ct_uid_map
            ):

                ct_index = ct_uid_map[
                    sop_uid
                ]

                mask[
                    ct_index
                ] = frame_mask[
                    frame_index
                ]

                matched_frames += 1


        # ------------------------------------------------------------
        # FALLBACK: POSITION-BASED MATCHING
        # ------------------------------------------------------------

        if matched_frames == 0:

            print(
                "SOP UID matching produced zero matches."
            )

            print(
                "Trying position-based matching..."
            )


            ct_positions = []

            for index, (
                ct_path,
                ct_ds
            ) in enumerate(ct_files):

                try:

                    z = float(
                        ct_ds
                        .ImagePositionPatient[2]
                    )

                    ct_positions.append(
                        (
                            index,
                            z
                        )
                    )

                except Exception:
                    continue


            for item in referenced_slices:

                frame_index = item[
                    "frame_index"
                ]

                position = item[
                    "position"
                ]


                if position is None:
                    continue


                try:

                    z_seg = float(
                        position[2]
                    )

                except Exception:
                    continue


                if len(ct_positions) == 0:
                    continue


                best_index = min(
                    ct_positions,
                    key=lambda x:
                    abs(x[1] - z_seg)
                )[0]


                mask[
                    best_index
                ] = frame_mask[
                    frame_index
                ]

                matched_frames += 1


        print(
            "Matched GTV-1 frames:",
            matched_frames
        )


        # ------------------------------------------------------------
        # VALIDATE MASK
        # ------------------------------------------------------------

        tumor_voxels = int(
            np.sum(mask)
        )

        tumor_slices = np.where(
            np.sum(
                mask,
                axis=(1, 2)
            ) > 0
        )[0]


        print(
            "Tumor voxels:",
            tumor_voxels
        )

        print(
            "Tumor-containing CT slices:",
            tumor_slices.tolist()
        )

        print(
            "Number of tumor slices:",
            len(tumor_slices)
        )


        if tumor_voxels == 0:

            print(
                "FAILED: EMPTY_GTV1_MASK"
            )

            results.append({
                "Patient_ID": patient_id,
                "Status": "FAILED",
                "Reason": "EMPTY_GTV1_MASK"
            })

            continue


        # ------------------------------------------------------------
        # SAVE MASK
        # ------------------------------------------------------------

        output_dir = os.path.join(
            patient_dir,
            "GTV1_MASK"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )


        mask_output = os.path.join(
            output_dir,
            "GTV1_MASK.npy"
        )


        np.save(
            mask_output,
            mask
        )


        # ------------------------------------------------------------
        # SAVE METADATA
        # ------------------------------------------------------------

        metadata_output = os.path.join(
            output_dir,
            "GTV1_MASK_INFO.txt"
        )


        with open(
            metadata_output,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "PROJECT 7 - RADIOMICS\n"
            )

            f.write(
                "STEP 13 - GTV1 MASK PREPARATION\n"
            )

            f.write(
                "=" * 75 + "\n\n"
            )

            f.write(
                f"Patient ID: {patient_id}\n"
            )

            f.write(
                f"SEG file: {selected_seg}\n"
            )

            f.write(
                f"GTV1 segment number: "
                f"{gtv_segment_number}\n"
            )

            f.write(
                f"Mask shape: "
                f"{mask.shape}\n"
            )

            f.write(
                f"Tumor voxels: "
                f"{tumor_voxels}\n"
            )

            f.write(
                f"Tumor slices: "
                f"{tumor_slices.tolist()}\n"
            )

            f.write(
                f"Matched frames: "
                f"{matched_frames}\n"
            )


        print(
            "\nSUCCESS"
        )

        print(
            "Saved mask:"
        )

        print(
            mask_output
        )


        results.append({
            "Patient_ID": patient_id,
            "Status": "SUCCESS",
            "Reason": "",
            "Tumor_Voxels": tumor_voxels,
            "Tumor_Slices": len(tumor_slices),
            "Matched_Frames": matched_frames,
            "Mask_Path": mask_output
        })


    except Exception as e:

        print(
            "\nFAILED:",
            type(e).__name__
        )

        print(
            str(e)
        )

        results.append({
            "Patient_ID": patient_id,
            "Status": "FAILED",
            "Reason": str(e)
        })


# ================================================================
# SAVE PROCESSING STATUS
# ================================================================

import pandas as pd


print("\n")
print("=" * 75)
print("SAVING MULTI-PATIENT RESULTS")
print("=" * 75)


results_df = pd.DataFrame(
    results
)


status_output = os.path.join(
    BASE_ROOT,
    "STEP_13_GTV1_MASK_PREPARATION_STATUS.csv"
)


results_df.to_csv(
    status_output,
    index=False
)


print(
    "Saved:",
    status_output
)


# ================================================================
# SUMMARY
# ================================================================

total = len(
    results_df
)

successful = int(
    (
        results_df["Status"]
        == "SUCCESS"
    ).sum()
)

failed = total - successful


print("\n")
print("=" * 75)
print("STEP 13 - GTV1 MASK PREPARATION COMPLETE")
print("=" * 75)

print(
    "\nTotal patients:",
    total
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

if total > 0:

    print(
        "Success rate:",
        f"{successful / total * 100:.2f}%"
    )


print(
    "\nStatus file:"
)

print(
    status_output
)

print("\n")
print("=" * 75)
print("NEXT STEP")
print("=" * 75)

print(
    "Run STEP 13 - Spectral Texture - All Patients again."
)

print(
    "The spectral script should now search for:"
)

print(
    "GTV1_MASK\\GTV1_MASK.npy"
)

print("=" * 75)


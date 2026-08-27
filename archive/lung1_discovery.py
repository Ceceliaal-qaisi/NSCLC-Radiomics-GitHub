import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt


print("=" * 70)
print("LUNG1 - DATA DISCOVERY")
print("=" * 70)

# ============================================================
# 1. ENTER PATIENT FOLDER
# ============================================================

PATIENT_FOLDER = input(
    "\nPaste the full path of ONE Lung1 patient folder:\n> "
).strip().strip('"')


if not os.path.isdir(PATIENT_FOLDER):
    raise FileNotFoundError(
        "\nThe folder does not exist.\n"
        "Please check the path and try again."
    )


# ============================================================
# 2. FIND ALL FILES
# ============================================================

print("\nScanning patient folder...")

all_files = []

for root, dirs, files in os.walk(PATIENT_FOLDER):
    for file in files:
        all_files.append(
            os.path.join(root, file)
        )

print("Total files found:", len(all_files))


# ============================================================
# 3. IDENTIFY DICOM FILES
# ============================================================

ct_files = []
seg_files = []
rtstruct_files = []

for path in all_files:

    try:

        ds = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=True
        )

        modality = getattr(
            ds,
            "Modality",
            ""
        )

        if modality == "CT":
            ct_files.append(path)

        elif modality == "SEG":
            seg_files.append(path)

        elif modality == "RTSTRUCT":
            rtstruct_files.append(path)

    except Exception:
        continue


# ============================================================
# 4. PRINT DICOM SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DICOM SUMMARY")
print("=" * 70)

print("\nCT files       :", len(ct_files))
print("SEG files      :", len(seg_files))
print("RTSTRUCT files :", len(rtstruct_files))


# ============================================================
# 5. CT INFORMATION
# ============================================================

if len(ct_files) > 0:

    ds = pydicom.dcmread(
        ct_files[0]
    )

    print("\n" + "=" * 70)
    print("CT INFORMATION")
    print("=" * 70)

    print("\nPatient ID      :",
          getattr(ds, "PatientID", "N/A"))

    print("Modality        :",
          getattr(ds, "Modality", "N/A"))

    print("Rows            :",
          getattr(ds, "Rows", "N/A"))

    print("Columns         :",
          getattr(ds, "Columns", "N/A"))

    print("Pixel Spacing   :",
          getattr(ds, "PixelSpacing", "N/A"))

    print("Slice Thickness :",
          getattr(ds, "SliceThickness", "N/A"))

    print("Series UID      :",
          getattr(ds, "SeriesInstanceUID", "N/A"))


# ============================================================
# 6. RTSTRUCT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("RTSTRUCT INFORMATION")
print("=" * 70)

if len(rtstruct_files) == 0:

    print("\nNo RTSTRUCT file found.")

else:

    rt = pydicom.dcmread(
        rtstruct_files[0],
        force=True
    )

    if hasattr(
        rt,
        "StructureSetROISequence"
    ):

        print("\nStructures found:\n")

        for roi in rt.StructureSetROISequence:

            roi_number = getattr(
                roi,
                "ROINumber",
                "N/A"
            )

            roi_name = getattr(
                roi,
                "ROIName",
                "N/A"
            )

            print(
                "ROI Number:",
                roi_number,
                "| Name:",
                roi_name
            )

    else:

        print(
            "\nNo StructureSetROISequence found."
        )


# ============================================================
# 7. DICOM SEG INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DICOM SEG INFORMATION")
print("=" * 70)

if len(seg_files) == 0:

    print("\nNo DICOM SEG file found.")

else:

    seg = pydicom.dcmread(
        seg_files[0],
        force=True
    )

    print(
        "\nModality:",
        getattr(seg, "Modality", "N/A")
    )

    if hasattr(
        seg,
        "SegmentSequence"
    ):

        print("\nSegments found:\n")

        for segment in seg.SegmentSequence:

            number = getattr(
                segment,
                "SegmentNumber",
                "N/A"
            )

            label = getattr(
                segment,
                "SegmentLabel",
                "N/A"
            )

            description = getattr(
                segment,
                "SegmentDescription",
                "N/A"
            )

            print(
                "Segment Number:",
                number,
                "| Label:",
                label,
                "| Description:",
                description
            )

    else:

        print(
            "\nNo SegmentSequence found."
        )


# ============================================================
# 8. FIND CT SERIES
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR CT SERIES")
print("=" * 70)

reader = sitk.ImageSeriesReader()

series_ids = reader.GetGDCMSeriesIDs(
    PATIENT_FOLDER
)


if series_ids is None:

    raise RuntimeError(
        "\nNo DICOM series found in this folder."
    )


print(
    "\nNumber of series found:",
    len(series_ids)
)


for i, series_id in enumerate(series_ids):

    files = reader.GetGDCMSeriesFileNames(
        PATIENT_FOLDER,
        series_id
    )

    print(
        f"Series {i}: {len(files)} files"
    )


# ============================================================
# 9. SELECT LARGEST SERIES
# ============================================================

series_information = []

for series_id in series_ids:

    files = reader.GetGDCMSeriesFileNames(
        PATIENT_FOLDER,
        series_id
    )

    series_information.append(
        (
            series_id,
            len(files),
            files
        )
    )


series_information.sort(
    key=lambda x: x[1],
    reverse=True
)


selected_series_id = (
    series_information[0][0]
)

selected_files = (
    series_information[0][2]
)


print("\nSelected CT series:")
print(selected_series_id)

print(
    "Number of slices:",
    len(selected_files)
)


# ============================================================
# 10. LOAD CT
# ============================================================

reader.SetFileNames(
    selected_files
)

ct_image = reader.Execute()

ct_array = sitk.GetArrayFromImage(
    ct_image
)


print("\n" + "=" * 70)
print("CT VOLUME INFORMATION")
print("=" * 70)

print(
    "\nArray shape:",
    ct_array.shape
)

print(
    "Spacing:",
    ct_image.GetSpacing()
)

print(
    "Origin:",
    ct_image.GetOrigin()
)


# ============================================================
# 11. SELECT A REPRESENTATIVE SLICE
# ============================================================

slice_ranges = []

for i in range(
    ct_array.shape[0]
):

    current_slice = ct_array[i]

    value_range = (
        np.max(current_slice)
        -
        np.min(current_slice)
    )

    slice_ranges.append(
        value_range
    )


best_slice = int(
    np.argmax(slice_ranges)
)


print(
    "\nRepresentative slice:",
    best_slice
)


# ============================================================
# 12. DISPLAY ORIGINAL CT
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    ct_array[best_slice],
    cmap="gray"
)

plt.title(
    f"Lung1 CT - Slice {best_slice}"
)

plt.axis("off")

plt.show()


# ============================================================
# 13. DISPLAY CT WITH WINDOWING
# ============================================================

plt.figure(
    figsize=(7, 7)
)

plt.imshow(
    ct_array[best_slice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title(
    f"Lung1 CT Windowed - Slice {best_slice}"
)

plt.axis("off")

plt.show()


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "\nPatient folder:",
    PATIENT_FOLDER
)

print(
    "\nCT files:",
    len(ct_files)
)

print(
    "SEG files:",
    len(seg_files)
)

print(
    "RTSTRUCT files:",
    len(rtstruct_files)
)

print(
    "\nCT volume:",
    ct_array.shape
)

print(
    "CT spacing:",
    ct_image.GetSpacing()
)

print("\nDISCOVERY STEP COMPLETE.")

print(
    "\nNext step:"
)

print(
    "GTV-1 -> Binary Mask -> Boundary -> Ordered Boundary"
)

print("=" * 70)
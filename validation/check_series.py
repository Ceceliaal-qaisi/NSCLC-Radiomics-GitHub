import os
import pydicom


PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"


print("=" * 80)
print("LUNG1 - DICOM SERIES CHECK")
print("=" * 80)

print("\nPatient:")
print(PATIENT_FOLDER)


# ------------------------------------------------------------
# Find every folder
# ------------------------------------------------------------

folders = []

for root, dirs, files in os.walk(PATIENT_FOLDER):

    for d in dirs:
        folders.append(
            os.path.join(root, d)
        )


# ------------------------------------------------------------
# Check one DICOM file from each folder
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SERIES INFORMATION")
print("=" * 80)


checked_series = set()

for folder in folders:

    dicom_file = None

    for file in os.listdir(folder):

        path = os.path.join(
            folder,
            file
        )

        if not os.path.isfile(path):
            continue

        if not file.lower().endswith(".dcm"):
            continue

        dicom_file = path
        break


    if dicom_file is None:
        continue


    try:

        ds = pydicom.dcmread(
            dicom_file,
            stop_before_pixels=True,
            force=True
        )

        series_uid = getattr(
            ds,
            "SeriesInstanceUID",
            "N/A"
        )

        # Avoid printing the same series multiple times
        if series_uid in checked_series:
            continue

        checked_series.add(series_uid)


        modality = getattr(
            ds,
            "Modality",
            "N/A"
        )

        series_description = getattr(
            ds,
            "SeriesDescription",
            "N/A"
        )

        study_description = getattr(
            ds,
            "StudyDescription",
            "N/A"
        )

        sop_class = getattr(
            ds,
            "SOPClassUID",
            "N/A"
        )

        rows = getattr(
            ds,
            "Rows",
            "N/A"
        )

        columns = getattr(
            ds,
            "Columns",
            "N/A"
        )

        print("\n----------------------------------------")

        print("Folder:")
        print(
            os.path.relpath(
                folder,
                PATIENT_FOLDER
            )
        )

        print("Modality:", modality)

        print(
            "Series Description:",
            series_description
        )

        print(
            "Study Description:",
            study_description
        )

        print(
            "Rows x Columns:",
            rows,
            "x",
            columns
        )

        print(
            "SOP Class UID:",
            sop_class
        )

        print(
            "Series UID:",
            series_uid
        )


    except Exception as e:

        print(
            "\nCould not read:",
            dicom_file
        )

        print(
            "Reason:",
            e
        )


# ------------------------------------------------------------
# Finish
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)

input("\nPress Enter to close...")
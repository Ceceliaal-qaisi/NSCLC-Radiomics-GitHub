import os
import pydicom

# ============================================
# PATH TO PATIENT
# ============================================

PATIENT_PATH = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"


# ============================================
# FUNCTION TO CHECK DICOM
# ============================================

def check_dicom_file(filepath):

    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)

        modality = getattr(ds, "Modality", "UNKNOWN")
        sop_class = getattr(ds, "SOPClassUID", "UNKNOWN")
        series_desc = getattr(ds, "SeriesDescription", "UNKNOWN")
        study_desc = getattr(ds, "StudyDescription", "UNKNOWN")
        series_number = getattr(ds, "SeriesNumber", "UNKNOWN")
        instance_number = getattr(ds, "InstanceNumber", "UNKNOWN")

        return {
            "Modality": modality,
            "SeriesDescription": series_desc,
            "StudyDescription": study_desc,
            "SeriesNumber": series_number,
            "InstanceNumber": instance_number,
            "SOPClassUID": sop_class
        }

    except Exception as e:

        return {
            "Modality": "ERROR",
            "SeriesDescription": str(e),
            "StudyDescription": "",
            "SeriesNumber": "",
            "InstanceNumber": "",
            "SOPClassUID": ""
        }


# ============================================
# START
# ============================================

print("\n========================================")
print("       DICOM DATA CHECK")
print("========================================\n")

print("Patient folder:")
print(PATIENT_PATH)

print("\nSearching for DICOM files...\n")


total = 0

for root, dirs, files in os.walk(PATIENT_PATH):

    dcm_files = [f for f in files if f.lower().endswith(".dcm")]

    if not dcm_files:
        continue

    print("\n----------------------------------------")
    print("FOLDER:")
    print(root)
    print("DICOM files:", len(dcm_files))
    print("----------------------------------------")

    # Check only first 3 files from each folder
    for filename in dcm_files[:3]:

        filepath = os.path.join(root, filename)

        info = check_dicom_file(filepath)

        total += 1

        print("\nFILE:", filename)
        print("Modality:", info["Modality"])
        print("Series Description:", info["SeriesDescription"])
        print("Study Description:", info["StudyDescription"])
        print("Series Number:", info["SeriesNumber"])
        print("Instance Number:", info["InstanceNumber"])
        print("SOP Class UID:", info["SOPClassUID"])


print("\n========================================")
print("TOTAL CHECKED FILES:", total)
print("========================================")

input("\nPress ENTER to close...")
import pydicom
import os

RT_FILE = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\78236\5bcda93e-ef26-4a58-a7b4-47832c15a000.dcm"

print("\n========================================")
print("        RTSTRUCT ROI CHECK")
print("========================================\n")

print("File:")
print(RT_FILE)

if not os.path.exists(RT_FILE):
    print("\nERROR: File not found!")
    input("\nPress ENTER to close...")
    exit()

ds = pydicom.dcmread(RT_FILE, stop_before_pixels=True)

print("\nFile found!")
print("Modality:", getattr(ds, "Modality", "UNKNOWN"))

print("\n========================================")
print("             ROIs FOUND")
print("========================================")

if hasattr(ds, "StructureSetROISequence"):

    rois = ds.StructureSetROISequence

    print("\nNumber of ROIs:", len(rois))

    for i, roi in enumerate(rois, start=1):

        number = getattr(roi, "ROINumber", "UNKNOWN")
        name = getattr(roi, "ROIName", "UNKNOWN")

        print(f"\nROI {i}")
        print("ROI Number:", number)
        print("ROI Name:", name)

else:
    print("\nNo StructureSetROISequence found!")

print("\n========================================")
print("                 DONE")
print("========================================")

input("\nPress ENTER to close...")
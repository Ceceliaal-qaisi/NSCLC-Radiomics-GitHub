import os
import numpy as np
import pydicom

# ============================================================
# PROJECT 7
# Radiomic Feature Extraction and Outcome Classification
# Lung1 NSCLC-Radiomics
#
# STEP 1:
# CT DICOM + RTSTRUCT GTV-1
#              ↓
#       Binary Tumor Mask
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_FOLDER = os.path.join(PATIENT_FOLDER, "82046")

RTSTRUCT_FILE = os.path.join(
    PATIENT_FOLDER,
    "78236",
    "5bcda93e-ef26-4a58-a7b4-47832c15a000.dcm"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "GTV1_MASK"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# 2. CHECK FILES
# ------------------------------------------------------------

print("=" * 60)
print("STEP 1 - GTV-1 EXTRACTION")
print("=" * 60)

print("\nPatient folder:")
print(PATIENT_FOLDER)

print("\nCT folder:")
print(CT_FOLDER)

print("\nRTSTRUCT:")
print(RTSTRUCT_FILE)


if not os.path.exists(CT_FOLDER):
    print("\nERROR: CT folder not found!")
    input("\nPress ENTER to close...")
    exit()


if not os.path.exists(RTSTRUCT_FILE):
    print("\nERROR: RTSTRUCT file not found!")
    input("\nPress ENTER to close...")
    exit()


# ------------------------------------------------------------
# 3. LOAD CT DICOM FILES
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Loading CT images...")
print("-" * 60)

ct_files = []

for filename in os.listdir(CT_FOLDER):

    if filename.lower().endswith(".dcm"):

        path = os.path.join(CT_FOLDER, filename)

        try:
            ds = pydicom.dcmread(path, stop_before_pixels=False)

            if getattr(ds, "Modality", "") == "CT":
                ct_files.append(ds)

        except Exception as e:
            print("Could not read:", filename)
            print("Reason:", e)


print("\nNumber of CT slices found:", len(ct_files))


if len(ct_files) == 0:
    print("\nERROR: No CT images found!")
    input("\nPress ENTER to close...")
    exit()


# ------------------------------------------------------------
# 4. SORT CT SLICES
# ------------------------------------------------------------

def get_slice_position(ds):

    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])

    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)

    return float(getattr(ds, "InstanceNumber", 0))


ct_files.sort(key=get_slice_position)


print("\nCT slices sorted successfully.")


# ------------------------------------------------------------
# 5. READ CT GEOMETRY
# ------------------------------------------------------------

first_ct = ct_files[0]

rows = int(first_ct.Rows)
cols = int(first_ct.Columns)

pixel_spacing = [
    float(first_ct.PixelSpacing[0]),
    float(first_ct.PixelSpacing[1])
]

print("\nCT dimensions:")
print("Rows:", rows)
print("Columns:", cols)

print("\nPixel spacing:")
print(pixel_spacing)


# ------------------------------------------------------------
# 6. LOAD RTSTRUCT
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Loading RTSTRUCT...")
print("-" * 60)

rt = pydicom.dcmread(RTSTRUCT_FILE)

print("RTSTRUCT loaded successfully.")

# ------------------------------------------------------------
# 7. FIND GTV-1
# ------------------------------------------------------------

print("\nSearching for GTV-1...")

gtv_number = None

if hasattr(rt, "StructureSetROISequence"):

    for roi in rt.StructureSetROISequence:

        roi_name = str(getattr(roi, "ROIName", "")).strip()

        print(
            "ROI:",
            roi_name,
            "| Number:",
            getattr(roi, "ROINumber", "UNKNOWN")
        )

        if roi_name.upper() == "GTV-1":

            gtv_number = int(roi.ROINumber)

            break


if gtv_number is None:

    print("\nERROR: GTV-1 was not found!")

    input("\nPress ENTER to close...")
    exit()


print("\nGTV-1 FOUND!")
print("GTV-1 ROI Number:", gtv_number)


# ------------------------------------------------------------
# 8. FIND CONTOURS BELONGING TO GTV-1
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Reading GTV-1 contours...")
print("-" * 60)

gtv_contours = []

if hasattr(rt, "ROIContourSequence"):

    for roi_contour in rt.ROIContourSequence:

        referenced_number = getattr(
            roi_contour,
            "ReferencedROINumber",
            None
        )

        if referenced_number == gtv_number:

            if hasattr(roi_contour, "ContourSequence"):

                for contour in roi_contour.ContourSequence:

                    if not hasattr(contour, "ContourData"):
                        continue

                    data = np.array(
                        contour.ContourData,
                        dtype=float
                    )

                    points = data.reshape(-1, 3)

                    gtv_contours.append(points)


print("\nNumber of GTV-1 contours:", len(gtv_contours))


if len(gtv_contours) == 0:

    print("\nERROR: No GTV-1 contours found!")

    input("\nPress ENTER to close...")
    exit()


# ------------------------------------------------------------
# 9. CREATE EMPTY 3D BINARY MASK
# ------------------------------------------------------------

number_of_slices = len(ct_files)

mask = np.zeros(
    (number_of_slices, rows, cols),
    dtype=np.uint8
)


# ------------------------------------------------------------
# 10. CONVERT PATIENT COORDINATES TO PIXEL COORDINATES
# ------------------------------------------------------------

def patient_to_pixel(points, ct):

    image_position = np.array(
        ct.ImagePositionPatient,
        dtype=float
    )

    spacing = np.array(
        ct.PixelSpacing,
        dtype=float
    )

    row_spacing = spacing[0]
    col_spacing = spacing[1]

    x = points[:, 0]
    y = points[:, 1]

    x0 = image_position[0]
    y0 = image_position[1]

    column = (x - x0) / col_spacing
    row = (y - y0) / row_spacing

    return row, column


# ------------------------------------------------------------
# 11. MATCH EACH CONTOUR TO ITS CT SLICE
# ------------------------------------------------------------

print("\n" + "-" * 60)
print("Matching contours with CT slices...")
print("-" * 60)


slice_positions = np.array(
    [get_slice_position(ds) for ds in ct_files]
)


matched_contours = 0


for contour_points in gtv_contours:

    contour_z = np.mean(contour_points[:, 2])

    differences = np.abs(
        slice_positions - contour_z
    )

    slice_index = int(
        np.argmin(differences)
    )

    # Convert physical coordinates to pixel coordinates
    rows_xy, cols_xy = patient_to_pixel(
        contour_points,
        ct_files[slice_index]
    )

    # --------------------------------------------------------
    # Polygon rasterization FROM SCRATCH
    # --------------------------------------------------------

    min_row = max(
        0,
        int(np.floor(np.min(rows_xy)))
    )

    max_row = min(
        rows - 1,
        int(np.ceil(np.max(rows_xy)))
    )

    min_col = max(
        0,
        int(np.floor(np.min(cols_xy)))
    )

    max_col = min(
        cols - 1,
        int(np.ceil(np.max(cols_xy)))
    )

    if min_row > max_row or min_col > max_col:
        continue


    polygon_rows = rows_xy
    polygon_cols = cols_xy


    # Ray casting algorithm
    # Determines whether each pixel is inside the contour

    for r in range(min_row, max_row + 1):

        for c in range(min_col, max_col + 1):

            inside = False

            j = len(polygon_rows) - 1

            for i in range(len(polygon_rows)):

                ri = polygon_rows[i]
                ci = polygon_cols[i]

                rj = polygon_rows[j]
                cj = polygon_cols[j]

                if ((ri > r) != (rj > r)):

                    denominator = (rj - ri)

                    if abs(denominator) < 1e-12:
                        denominator = 1e-12

                    intersection = (
                        (cj - ci)
                        * (r - ri)
                        / denominator
                        + ci
                    )

                    if c < intersection:
                        inside = not inside

                j = i


            if inside:
                mask[slice_index, r, c] = 1


    matched_contours += 1


# ------------------------------------------------------------
# 12. CHECK MASK
# ------------------------------------------------------------

print("\nContours processed:", matched_contours)

tumor_pixels = int(np.sum(mask))

print("Total tumor pixels:", tumor_pixels)

tumor_slices = np.where(
    np.sum(mask, axis=(1, 2)) > 0
)[0]


print("Number of CT slices containing tumor:",
      len(tumor_slices))


if tumor_pixels == 0:

    print("\nWARNING:")
    print("The generated tumor mask is empty.")

else:

    print("\nSUCCESS!")
    print("Binary Tumor Mask generated.")


# ------------------------------------------------------------
# 13. SAVE MASK
# ------------------------------------------------------------

mask_file = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_binary_mask.npy"
)

np.save(mask_file, mask)


print("\nMask saved to:")
print(mask_file)


# ------------------------------------------------------------
# 14. SAVE INFORMATION
# ------------------------------------------------------------

info_file = os.path.join(
    OUTPUT_FOLDER,
    "mask_info.txt"
)

with open(info_file, "w") as f:

    f.write("LUNG1-001 GTV-1 Binary Tumor Mask\n")
    f.write("=" * 50 + "\n")

    f.write(
        f"Number of CT slices: {number_of_slices}\n"
    )

    f.write(
        f"Image dimensions: {rows} x {cols}\n"
    )

    f.write(
        f"Pixel spacing: {pixel_spacing}\n"
    )

    f.write(
        f"GTV-1 ROI Number: {gtv_number}\n"
    )

    f.write(
        f"Number of contours: {len(gtv_contours)}\n"
    )

    f.write(
        f"Tumor pixels: {tumor_pixels}\n"
    )

    f.write(
        f"Tumor slices: {len(tumor_slices)}\n"
    )


print("\nInformation saved to:")
print(info_file)


# ------------------------------------------------------------
# 15. FINAL RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 1 COMPLETED")
print("=" * 60)

print("\nCT:")
print("134 DICOM slices")

print("\nSegmentation:")
print("GTV-1")

print("\nOutput:")
print("3D Binary Tumor Mask")

print("\nOutput folder:")
print(OUTPUT_FOLDER)

print("\nNext step:")
print("Moore Boundary Tracking")

print("\n" + "=" * 60)

input("\nPress ENTER to close...")
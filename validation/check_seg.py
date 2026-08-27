import pydicom


SEG_FILE = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\9.554"


print("=" * 70)
print("LUNG1 - SEGMENT INFORMATION")
print("=" * 70)


# ------------------------------------------------------------
# Find the DICOM SEG file
# ------------------------------------------------------------

import os

seg_files = []

for root, dirs, files in os.walk(SEG_FILE):

    for file in files:

        if file.lower().endswith(".dcm"):

            seg_files.append(
                os.path.join(root, file)
            )


print("\nSEG files found:", len(seg_files))


if len(seg_files) == 0:

    print("\nERROR: No DICOM SEG file found.")

    input("\nPress Enter to close...")

    raise SystemExit


# ------------------------------------------------------------
# Read SEG
# ------------------------------------------------------------

seg_file = seg_files[0]

print("\nReading:")
print(seg_file)


ds = pydicom.dcmread(
    seg_file,
    stop_before_pixels=True,
    force=True
)


print("\nModality:")
print(
    getattr(
        ds,
        "Modality",
        "N/A"
    )
)


# ------------------------------------------------------------
# Segment Sequence
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SEGMENTS")
print("=" * 70)


if not hasattr(
    ds,
    "SegmentSequence"
):

    print("\nNo SegmentSequence found.")

else:

    for segment in ds.SegmentSequence:

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

        algorithm = getattr(
            segment,
            "SegmentAlgorithmType",
            "N/A"
        )

        print("\n----------------------------------------")

        print(
            "Segment Number:",
            number
        )

        print(
            "Segment Label:",
            label
        )

        print(
            "Description:",
            description
        )

        print(
            "Algorithm Type:",
            algorithm
        )


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)

input("\nPress Enter to close...")
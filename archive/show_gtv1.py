import os
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt


PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

CT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "82046"
)

SEG_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "9.554"
)


# ============================================================
# LOAD CT
# ============================================================

reader = sitk.ImageSeriesReader()

series_ids = reader.GetGDCMSeriesIDs(
    CT_FOLDER
)

series_data = []

for series_id in series_ids:

    files = reader.GetGDCMSeriesFileNames(
        CT_FOLDER,
        series_id
    )

    series_data.append(
        (
            series_id,
            len(files),
            files
        )
    )

series_data.sort(
    key=lambda x: x[1],
    reverse=True
)

reader.SetFileNames(
    series_data[0][2]
)

ct_image = reader.Execute()

ct_array = sitk.GetArrayFromImage(
    ct_image
)


# ============================================================
# LOAD SEG
# ============================================================

seg_file = None

for root, dirs, files in os.walk(SEG_FOLDER):

    for file in files:

        if file.lower().endswith(".dcm"):

            seg_file = os.path.join(
                root,
                file
            )

            break

    if seg_file:
        break


seg_ds = pydicom.dcmread(
    seg_file,
    force=True
)

pixel_array = seg_ds.pixel_array


# ============================================================
# FIND GTV-1
# ============================================================

gtv_number = 1

frame_segment_numbers = []

for frame in seg_ds.PerFrameFunctionalGroupsSequence:

    number = int(
        frame.SegmentIdentificationSequence[
            0
        ].ReferencedSegmentNumber
    )

    frame_segment_numbers.append(number)


frame_segment_numbers = np.array(
    frame_segment_numbers
)


gtv_frames = np.where(
    frame_segment_numbers == gtv_number
)[0]


gtv_mask = np.zeros(
    (
        134,
        512,
        512
    ),
    dtype=np.uint8
)


for i, frame_index in enumerate(gtv_frames):

    gtv_mask[i] = (
        pixel_array[frame_index] > 0
    ).astype(np.uint8)


# ============================================================
# SELECT SLICE 74
# ============================================================

slice_index = 74


ct_slice = ct_array[
    slice_index
]

mask_slice = gtv_mask[
    slice_index
]


# ============================================================
# DISPLAY CT
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    ct_slice,
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.title(
    "CT - Slice 74"
)

plt.axis("off")

plt.show()


# ============================================================
# DISPLAY BINARY MASK
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    mask_slice,
    cmap="gray"
)

plt.title(
    "GTV-1 Binary Mask - Slice 74"
)

plt.axis("off")

plt.show()


# ============================================================
# DISPLAY OVERLAY
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    ct_slice,
    cmap="gray",
    vmin=-1000,
    vmax=400
)

plt.imshow(
    mask_slice,
    cmap="Reds",
    alpha=0.5
)

plt.title(
    "CT + GTV-1 Overlay - Slice 74"
)

plt.axis("off")

plt.show()


print("\nGTV-1 pixels on slice 74:")
print(
    np.sum(mask_slice)
)

input(
    "\nPress Enter to close..."
)
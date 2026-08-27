# ============================================================
# NSCLC-Radiomics Lung1
# GTV-1 Shape + Regional Feature Extraction
# ============================================================

import os
import glob
import numpy as np
import pydicom
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS - ALREADY SET FOR YOUR DATA
# ============================================================

CT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\82046"

SEG_FILE = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331\9.554\553521b9-f9e8-4103-b04d-5f032b974b68.dcm"


# ============================================================
# 2. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = os.path.join(
    os.path.dirname(SEG_FILE),
    "Radiomics_Output"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

RESAMPLED_BOUNDARY_IMAGE = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_resampled_boundary.png"
)

RESAMPLED_CT_IMAGE = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_resampled_on_CT.png"
)

CONNECTED_BOUNDARY_IMAGE = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_8connected_boundary.png"
)

FEATURES_FILE = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_shape_regional_features.txt"
)


# ============================================================
# 4. CHECK PATHS
# ============================================================

print("=" * 60)
print("CHECKING DATA PATHS")
print("=" * 60)

print("\nCT folder:")
print(CT_FOLDER)

print("\nSEG file:")
print(SEG_FILE)

if not os.path.isdir(CT_FOLDER):
    raise FileNotFoundError(
        "\nCT folder was not found!"
    )

if not os.path.isfile(SEG_FILE):
    raise FileNotFoundError(
        "\nSEG file was not found!"
    )

print("\n✓ CT folder found")
print("✓ SEG file found")


# ============================================================
# 5. READ CT FILES
# ============================================================

print("\n" + "=" * 60)
print("READING CT DATA")
print("=" * 60)

ct_files = glob.glob(
    os.path.join(
        CT_FOLDER,
        "*.dcm"
    )
)

print(
    f"Total DICOM files in CT folder: "
    f"{len(ct_files)}"
)

if len(ct_files) == 0:
    raise FileNotFoundError(
        "No DICOM files found in CT folder."
    )


ct_slices = []


for file in ct_files:

    try:

        ds = pydicom.dcmread(
            file,
            stop_before_pixels=False
        )

        if getattr(
            ds,
            "Modality",
            ""
        ) == "CT":

            if hasattr(
                ds,
                "ImagePositionPatient"
            ):

                ct_slices.append(ds)

    except Exception:
        pass


if len(ct_slices) == 0:

    raise ValueError(
        "No valid CT slices were found."
    )


# ============================================================
# 6. SORT CT SLICES
# ============================================================

ct_slices.sort(
    key=lambda ds:
    float(
        ds.ImagePositionPatient[2]
    )
)


print(
    f"Valid CT slices: "
    f"{len(ct_slices)}"
)


# ============================================================
# 7. CREATE CT VOLUME
# ============================================================

ct_volume = np.stack(
    [
        ds.pixel_array
        for ds in ct_slices
    ],
    axis=0
)


print(
    f"CT volume shape: "
    f"{ct_volume.shape}"
)


# ============================================================
# 8. READ SEG
# ============================================================

print("\n" + "=" * 60)
print("READING DICOM SEG")
print("=" * 60)


seg_ds = pydicom.dcmread(
    SEG_FILE
)


print(
    f"Modality: "
    f"{getattr(seg_ds, 'Modality', 'Unknown')}"
)

print(
    f"Rows: "
    f"{getattr(seg_ds, 'Rows', 'Unknown')}"
)

print(
    f"Columns: "
    f"{getattr(seg_ds, 'Columns', 'Unknown')}"
)

print(
    f"Number of Frames: "
    f"{getattr(seg_ds, 'NumberOfFrames', 'Unknown')}"
)


# ============================================================
# 9. FIND SEGMENTS
# ============================================================

print("\n" + "=" * 60)
print("AVAILABLE SEGMENTS")
print("=" * 60)


if not hasattr(
    seg_ds,
    "SegmentSequence"
):

    raise ValueError(
        "This DICOM file does not contain SegmentSequence."
    )


segments = seg_ds.SegmentSequence


for segment in segments:

    number = getattr(
        segment,
        "SegmentNumber",
        "Unknown"
    )

    label = getattr(
        segment,
        "SegmentLabel",
        "Unknown"
    )

    description = getattr(
        segment,
        "SegmentDescription",
        "Unknown"
    )

    print(
        f"\nSegment Number: {number}"
    )

    print(
        f"Segment Label: {label}"
    )

    print(
        f"Segment Description: {description}"
    )


# ============================================================
# 10. SELECT GTV-1
# ============================================================

selected_segment_number = None


for segment in segments:

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


    text = (
        label
        +
        " "
        +
        description
    ).lower()


    if (
        "gtv-1" in text
        or
        "gtv 1" in text
        or
        "gtv1" in text
        or
        "neoplasm" in text
        or
        "tumor" in text
        or
        "tumour" in text
    ):

        selected_segment_number = int(
            segment.SegmentNumber
        )

        print(
            "\nSelected segment:"
        )

        print(
            f"Number: "
            f"{selected_segment_number}"
        )

        print(
            f"Label: "
            f"{label}"
        )

        print(
            f"Description: "
            f"{description}"
        )

        break


if selected_segment_number is None:

    raise ValueError(
        "\nGTV-1 was not automatically found.\n"
        "Check the AVAILABLE SEGMENTS printed above."
    )


# ============================================================
# 11. READ SEG PIXEL DATA
# ============================================================

seg_array = seg_ds.pixel_array


print(
    f"\nSEG pixel array shape: "
    f"{seg_array.shape}"
)


# ============================================================
# 12. GET GTV FRAMES
# ============================================================

gtv_frames = []


number_of_frames = int(
    seg_ds.NumberOfFrames
)


for frame_index in range(
    number_of_frames
):

    try:

        frame_info = (
            seg_ds
            .PerFrameFunctionalGroupsSequence[
                frame_index
            ]
        )


        if hasattr(
            frame_info,
            "SegmentIdentificationSequence"
        ):

            segment_id = int(
                frame_info
                .SegmentIdentificationSequence[0]
                .ReferencedSegmentNumber
            )


            if (
                segment_id
                ==
                selected_segment_number
            ):

                gtv_frames.append(
                    frame_index
                )

    except Exception:
        continue


print(
    f"GTV-1 frames found: "
    f"{len(gtv_frames)}"
)


if len(gtv_frames) == 0:

    raise ValueError(
        "No frames belonging to GTV-1 were found."
    )


# ============================================================
# 13. CT Z POSITIONS
# ============================================================

ct_z_positions = np.array(
    [
        float(
            ds.ImagePositionPatient[2]
        )
        for ds in ct_slices
    ]
)


# ============================================================
# 14. CREATE GTV MASK
# ============================================================

gtv_mask = np.zeros(
    (
        len(ct_slices),
        seg_ds.Rows,
        seg_ds.Columns
    ),
    dtype=np.uint8
)


matched_count = 0


# ============================================================
# 15. MATCH SEG TO CT
# ============================================================

for frame_index in gtv_frames:

    frame_info = (
        seg_ds
        .PerFrameFunctionalGroupsSequence[
            frame_index
        ]
    )


    frame_z = None


    try:

        if hasattr(
            frame_info,
            "PlanePositionSequence"
        ):

            position = (
                frame_info
                .PlanePositionSequence[0]
                .ImagePositionPatient
            )

            frame_z = float(
                position[2]
            )

    except Exception:
        pass


    if frame_z is not None:

        ct_index = int(
            np.argmin(
                np.abs(
                    ct_z_positions
                    -
                    frame_z
                )
            )
        )

    else:

        print(
            f"Warning: no position for frame "
            f"{frame_index}"
        )

        continue


    frame_mask = (
        seg_array[
            frame_index
        ]
        >
        0
    ).astype(np.uint8)


    # If dimensions match
    # combine masks.

    if (
        frame_mask.shape
        !=
        gtv_mask[
            ct_index
        ].shape
    ):

        print(
            "Warning: SEG and CT dimensions "
            "do not match."
        )

        continue


    gtv_mask[
        ct_index
    ] = np.logical_or(
        gtv_mask[
            ct_index
        ],
        frame_mask
    ).astype(
        np.uint8
    )


    matched_count += 1


print(
    f"\nMatched GTV frames: "
    f"{matched_count}"
)


# ============================================================
# 16. FIND TUMOR SLICES
# ============================================================

tumor_voxels = int(
    np.sum(gtv_mask)
)


tumor_slices = np.where(
    np.sum(
        gtv_mask,
        axis=(1, 2)
    )
    >
    0
)[0]


print("\n" + "=" * 60)
print("GTV-1 MASK")
print("=" * 60)


print(
    f"Tumor voxels: "
    f"{tumor_voxels}"
)


print(
    f"Tumor-containing slices: "
    f"{tumor_slices}"
)


print(
    f"Number of tumor slices: "
    f"{len(tumor_slices)}"
)


if len(tumor_slices) == 0:

    raise ValueError(
        "GTV-1 mask is empty."
    )


# ============================================================
# 17. SELECT MIDDLE TUMOR SLICE
# ============================================================

selected_tumor_slice = int(
    tumor_slices[
        len(tumor_slices) // 2
    ]
)


print(
    f"\nSelected tumor slice: "
    f"{selected_tumor_slice}"
)


mask_2d = (
    gtv_mask[
        selected_tumor_slice
    ]
    .astype(np.uint8)
)


tumor_pixels = int(
    np.sum(mask_2d)
)


print(
    f"2-D tumor pixels: "
    f"{tumor_pixels}"
)


if tumor_pixels == 0:

    raise ValueError(
        "Selected slice contains no tumor."
    )


# ============================================================
# 18. CONNECTED COMPONENTS
# ============================================================

def connected_components(
    binary
):

    binary = (
        binary
        .astype(bool)
    )

    rows, cols = binary.shape

    visited = np.zeros_like(
        binary,
        dtype=bool
    )

    components = []


    neighbors = [

        (-1, -1),
        (-1, 0),
        (-1, 1),

        (0, -1),
        (0, 1),

        (1, -1),
        (1, 0),
        (1, 1)

    ]


    for r in range(rows):

        for c in range(cols):

            if (
                binary[r, c]
                and
                not visited[r, c]
            ):

                stack = [
                    (r, c)
                ]

                visited[
                    r,
                    c
                ] = True

                component = []


                while stack:

                    rr, cc = stack.pop()

                    component.append(
                        (
                            rr,
                            cc
                        )
                    )


                    for dr, dc in neighbors:

                        nr = rr + dr

                        nc = cc + dc


                        if (
                            0 <= nr < rows
                            and
                            0 <= nc < cols
                            and
                            binary[nr, nc]
                            and
                            not visited[nr, nc]
                        ):

                            visited[
                                nr,
                                nc
                            ] = True

                            stack.append(
                                (
                                    nr,
                                    nc
                                )
                            )


                components.append(
                    component
                )


    return components


components = connected_components(
    mask_2d
)


if len(components) == 0:

    raise ValueError(
        "No tumor component found."
    )


largest_component = max(
    components,
    key=len
)


clean_mask = np.zeros_like(
    mask_2d,
    dtype=np.uint8
)


for r, c in largest_component:

    clean_mask[
        r,
        c
    ] = 1


mask_2d = clean_mask


print(
    f"Largest component size: "
    f"{np.sum(mask_2d)} pixels"
)


# ============================================================
# 19. PADDING
# ============================================================

padded_mask = np.pad(
    mask_2d,
    1,
    mode="constant",
    constant_values=0
)


# ============================================================
# 20. FIND STARTING PIXEL
# ============================================================

def find_starting_pixel(
    binary
):

    rows, cols = binary.shape


    for r in range(rows):

        for c in range(cols):

            if binary[
                r,
                c
            ] == 1:

                return (
                    r,
                    c
                )


    return None


# ============================================================
# 21. MOORE BOUNDARY
# ============================================================

def moore_boundary(
    binary
):

    binary = (
        binary
        .astype(np.uint8)
    )


    start = find_starting_pixel(
        binary
    )


    if start is None:

        return []


    directions = [

        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1)

    ]


    boundary = []

    current = start

    previous_direction = 7


    max_iterations = (
        binary.shape[0]
        *
        binary.shape[1]
        *
        20
    )


    for iteration in range(
        max_iterations
    ):

        boundary.append(
            current
        )


        r, c = current


        found = False


        search_start = (
            previous_direction
            +
            1
        ) % 8


        for k in range(8):

            d = (
                search_start
                +
                k
            ) % 8


            dr, dc = directions[d]


            nr = r + dr

            nc = c + dc


            if (
                0 <= nr < binary.shape[0]
                and
                0 <= nc < binary.shape[1]
                and
                binary[nr, nc] == 1
            ):

                current = (
                    nr,
                    nc
                )


                previous_direction = (
                    d + 5
                ) % 8


                found = True

                break


        if not found:

            break


        if (
            current == start
            and
            len(boundary) > 10
        ):

            break


    return boundary


boundary = moore_boundary(
    padded_mask
)


print("\n" + "=" * 60)
print("MOORE BOUNDARY")
print("=" * 60)


print(
    f"Boundary points: "
    f"{len(boundary)}"
)


if len(boundary) < 3:

    raise ValueError(
        "Boundary extraction failed."
    )


# ============================================================
# 22. REMOVE CONSECUTIVE DUPLICATES
# ============================================================

clean_boundary = []


for p in boundary:

    if (
        len(clean_boundary) == 0
        or
        p != clean_boundary[-1]
    ):

        clean_boundary.append(
            p
        )


boundary = clean_boundary


# Remove repeated closing point

if (
    len(boundary) > 1
    and
    boundary[0]
    ==
    boundary[-1]
):

    boundary.pop()


# ============================================================
# 23. CHECK 8-CONNECTIVITY
# ============================================================

def is_8_connected(
    p1,
    p2
):

    dr = abs(
        p2[0] - p1[0]
    )

    dc = abs(
        p2[1] - p1[1]
    )


    return (
        max(dr, dc) == 1
    )


invalid_steps = 0


for i in range(
    len(boundary)
):

    p1 = boundary[i]

    p2 = boundary[
        (i + 1)
        %
        len(boundary)
    ]


    if not is_8_connected(
        p1,
        p2
    ):

        invalid_steps += 1


print(
    f"Invalid 8-connected steps: "
    f"{invalid_steps}"
)


# ============================================================
# 24. FREEMAN CHAIN CODE
# ============================================================

def get_direction(
    p1,
    p2
):

    dr = (
        p2[0]
        -
        p1[0]
    )

    dc = (
        p2[1]
        -
        p1[1]
    )


    mapping = {

        (-1, 0): 0,
        (-1, 1): 1,
        (0, 1): 2,
        (1, 1): 3,
        (1, 0): 4,
        (1, -1): 5,
        (0, -1): 6,
        (-1, -1): 7

    }


    if (
        dr,
        dc
    ) not in mapping:

        raise ValueError(
            f"Invalid boundary step: "
            f"{(dr, dc)}"
        )


    return mapping[
        (dr, dc)
    ]


chain_code = []


for i in range(
    len(boundary)
):

    p1 = boundary[i]

    p2 = boundary[
        (i + 1)
        %
        len(boundary)
    ]


    chain_code.append(
        get_direction(
            p1,
            p2
        )
    )


print("\n" + "=" * 60)
print("FREEMAN CHAIN CODE")
print("=" * 60)


print(
    f"Chain code length: "
    f"{len(chain_code)}"
)


print(
    chain_code
)


# ============================================================
# 25. FIRST DIFFERENCE
# ============================================================

first_diff = []


for i in range(
    len(chain_code)
):

    diff = (
        chain_code[i]
        -
        chain_code[i - 1]
    ) % 8


    first_diff.append(
        int(diff)
    )


print("\n" + "=" * 60)
print("FIRST DIFFERENCE")
print("=" * 60)


print(
    first_diff
)


# ============================================================
# 26. MINIMUM MAGNITUDE CHAIN CODE
# ============================================================

def minimum_magnitude_chain_code(
    chain
):

    chain = list(
        chain
    )


    candidates = []


    for i in range(
        len(chain)
    ):

        rotated = (
            chain[i:]
            +
            chain[:i]
        )


        candidates.append(
            tuple(rotated)
        )


    return list(
        min(candidates)
    )


minimum_chain = (
    minimum_magnitude_chain_code(
        chain_code
    )
)


print("\n" + "=" * 60)
print("MINIMUM MAGNITUDE CHAIN CODE")
print("=" * 60)


print(
    minimum_chain
)


# ============================================================
# 27. BOUNDARY RESAMPLING
# ============================================================

def resample_closed_boundary(
    points,
    n_samples=33
):

    points = np.asarray(
        points,
        dtype=float
    )


    closed = np.vstack(
        [
            points,
            points[0]
        ]
    )


    differences = np.diff(
        closed,
        axis=0
    )


    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=1
        )
    )


    cumulative = np.concatenate(
        [
            [0],
            np.cumsum(
                distances
            )
        ]
    )


    total_length = (
        cumulative[-1]
    )


    sample_positions = np.linspace(
        0,
        total_length,
        n_samples,
        endpoint=False
    )


    result = []


    for s in sample_positions:

        index = (
            np.searchsorted(
                cumulative,
                s,
                side="right"
            )
            -
            1
        )


        index = min(
            index,
            len(distances) - 1
        )


        segment_start = (
            cumulative[index]
        )


        segment_length = (
            distances[index]
        )


        if segment_length == 0:

            alpha = 0

        else:

            alpha = (
                s
                -
                segment_start
            ) / segment_length


        p1 = closed[index]

        p2 = closed[
            index + 1
        ]


        point = (
            p1
            +
            alpha
            *
            (
                p2
                -
                p1
            )
        )


        result.append(
            point
        )


    return (
        np.asarray(result),
        total_length
    )


resampled_points, boundary_length = (
    resample_closed_boundary(
        boundary,
        33
    )
)


print("\n" + "=" * 60)
print("BOUNDARY RESAMPLING")
print("=" * 60)


print(
    f"Original points: "
    f"{len(boundary)}"
)


print(
    f"Resampled points: "
    f"{len(resampled_points)}"
)


print(
    f"Boundary length: "
    f"{boundary_length}"
)


# ============================================================
# 28. FOURIER DESCRIPTORS
# ============================================================

z = (
    resampled_points[:, 1]
    +
    1j
    *
    resampled_points[:, 0]
)


fourier_descriptors = np.fft.fft(
    z
)


print("\n" + "=" * 60)
print("FOURIER DESCRIPTORS")
print("=" * 60)


print(
    f"Number of descriptors: "
    f"{len(fourier_descriptors)}"
)


print(
    "\nFirst 10:"
)


for i, value in enumerate(
    fourier_descriptors[:10]
):

    print(
        f"FD[{i}] = {value}"
    )


# ============================================================
# 29. NORMALIZED FOURIER DESCRIPTORS
# ============================================================

nfd = (
    fourier_descriptors
    -
    fourier_descriptors[0]
)


if abs(
    nfd[1]
) > 0:

    nfd = (
        nfd
        /
        abs(
            nfd[1]
        )
    )


# ============================================================
# 30. BOUNDARY MOMENTS
# ============================================================

x_boundary = (
    resampled_points[:, 1]
)

y_boundary = (
    resampled_points[:, 0]
)


m00_boundary = len(
    resampled_points
)

m10_boundary = np.sum(
    x_boundary
)

m01_boundary = np.sum(
    y_boundary
)

m20_boundary = np.sum(
    x_boundary ** 2
)

m02_boundary = np.sum(
    y_boundary ** 2
)

m11_boundary = np.sum(
    x_boundary
    *
    y_boundary
)


centroid_x_boundary = (
    m10_boundary
    /
    m00_boundary
)

centroid_y_boundary = (
    m01_boundary
    /
    m00_boundary
)


mu20_boundary = np.sum(
    (
        x_boundary
        -
        centroid_x_boundary
    ) ** 2
)


mu02_boundary = np.sum(
    (
        y_boundary
        -
        centroid_y_boundary
    ) ** 2
)


mu11_boundary = np.sum(
    (
        x_boundary
        -
        centroid_x_boundary
    )
    *
    (
        y_boundary
        -
        centroid_y_boundary
    )
)


# ============================================================
# 31. REGIONAL FEATURES
# ============================================================

area = int(
    np.sum(mask_2d)
)


tumor_positions = np.argwhere(
    mask_2d == 1
)


yy = tumor_positions[:, 0].astype(
    float
)

xx = tumor_positions[:, 1].astype(
    float
)


regional_y = np.mean(
    yy
)

regional_x = np.mean(
    xx
)


# ============================================================
# 32. BOUNDING BOX
# ============================================================

r_min = np.min(
    tumor_positions[:, 0]
)

r_max = np.max(
    tumor_positions[:, 0]
)

c_min = np.min(
    tumor_positions[:, 1]
)

c_max = np.max(
    tumor_positions[:, 1]
)


bbox_height = (
    r_max
    -
    r_min
    +
    1
)


bbox_width = (
    c_max
    -
    c_min
    +
    1
)


bbox_area = (
    bbox_height
    *
    bbox_width
)


# ============================================================
# 33. PERIMETER
# ============================================================

perimeter = 0.0


rows, cols = mask_2d.shape


for r in range(rows):

    for c in range(cols):

        if mask_2d[r, c] == 1:

            if (
                r == 0
                or
                mask_2d[
                    r - 1,
                    c
                ] == 0
            ):

                perimeter += 1


            if (
                r == rows - 1
                or
                mask_2d[
                    r + 1,
                    c
                ] == 0
            ):

                perimeter += 1


            if (
                c == 0
                or
                mask_2d[
                    r,
                    c - 1
                ] == 0
            ):

                perimeter += 1


            if (
                c == cols - 1
                or
                mask_2d[
                    r,
                    c + 1
                ] == 0
            ):

                perimeter += 1


# ============================================================
# 34. REGIONAL SHAPE FEATURES
# ============================================================

compactness = (
    4
    *
    np.pi
    *
    area
    /
    (
        perimeter ** 2
    )
)


rectangularity = (
    area
    /
    bbox_area
)


equivalent_diameter = np.sqrt(
    4
    *
    area
    /
    np.pi
)


aspect_ratio = (
    bbox_height
    /
    bbox_width
)


# ============================================================
# 35. REGIONAL MOMENTS
# ============================================================

m00_region = len(
    tumor_positions
)

m10_region = np.sum(
    xx
)

m01_region = np.sum(
    yy
)

m20_region = np.sum(
    xx ** 2
)

m02_region = np.sum(
    yy ** 2
)

m11_region = np.sum(
    xx
    *
    yy
)


mu20_region = np.sum(
    (
        xx
        -
        regional_x
    ) ** 2
)


mu02_region = np.sum(
    (
        yy
        -
        regional_y
    ) ** 2
)


mu11_region = np.sum(
    (
        xx
        -
        regional_x
    )
    *
    (
        yy
        -
        regional_y
    )
)


# ============================================================
# 36. TOPOLOGICAL FEATURES
# ============================================================

connected_components_count = len(
    connected_components(
        mask_2d
    )
)


# ------------------------------------------------------------
# Hole detection
# ------------------------------------------------------------

padded_for_holes = np.pad(
    mask_2d,
    1,
    mode="constant",
    constant_values=0
)


background = (
    padded_for_holes == 0
)


visited = np.zeros_like(
    background,
    dtype=bool
)


stack = [(0, 0)]

visited[
    0,
    0
] = True


neighbors4 = [

    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)

]


while stack:

    r, c = stack.pop()


    for dr, dc in neighbors4:

        nr = r + dr

        nc = c + dc


        if (
            0 <= nr < background.shape[0]
            and
            0 <= nc < background.shape[1]
            and
            background[nr, nc]
            and
            not visited[nr, nc]
        ):

            visited[
                nr,
                nc
            ] = True


            stack.append(
                (
                    nr,
                    nc
                )
            )


holes_mask = (
    background
    &
    ~visited
)


holes = len(
    connected_components(
        holes_mask.astype(
            np.uint8
        )
    )
)


euler_number = (
    connected_components_count
    -
    holes
)


# ============================================================
# 37. SAVE IMAGES
# ============================================================

plt.figure(
    figsize=(7, 7)
)


plt.imshow(
    mask_2d,
    cmap="gray"
)


plt.plot(
    resampled_points[:, 1] - 1,
    resampled_points[:, 0] - 1,
    "r.-"
)


plt.axis("off")

plt.tight_layout()


plt.savefig(
    RESAMPLED_BOUNDARY_IMAGE,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------

ct_image = ct_volume[
    selected_tumor_slice
]


plt.figure(
    figsize=(8, 8)
)


plt.imshow(
    ct_image,
    cmap="gray"
)


plt.plot(
    resampled_points[:, 1] - 1,
    resampled_points[:, 0] - 1,
    "r.-"
)


plt.axis("off")

plt.tight_layout()


plt.savefig(
    RESAMPLED_CT_IMAGE,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------

plt.figure(
    figsize=(7, 7)
)


plt.imshow(
    mask_2d,
    cmap="gray"
)


cb = np.asarray(
    boundary
)


plt.plot(
    cb[:, 1] - 1,
    cb[:, 0] - 1,
    "r-"
)


plt.axis("off")

plt.tight_layout()


plt.savefig(
    CONNECTED_BOUNDARY_IMAGE,
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 38. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL FEATURE SUMMARY")
print("=" * 60)


print(
    f"Tumor slice: "
    f"{selected_tumor_slice}"
)


print(
    f"Tumor area: "
    f"{area}"
)


print(
    f"Boundary points: "
    f"{len(boundary)}"
)


print(
    f"Chain code length: "
    f"{len(chain_code)}"
)


print(
    f"Boundary length: "
    f"{boundary_length}"
)


print(
    f"Bounding box height: "
    f"{bbox_height}"
)


print(
    f"Bounding box width: "
    f"{bbox_width}"
)


print(
    f"Perimeter: "
    f"{perimeter}"
)


print(
    f"Compactness: "
    f"{compactness}"
)


print(
    f"Rectangularity: "
    f"{rectangularity}"
)


print(
    f"Equivalent diameter: "
    f"{equivalent_diameter}"
)


print(
    f"Aspect ratio: "
    f"{aspect_ratio}"
)


print(
    f"Regional centroid X: "
    f"{regional_x}"
)


print(
    f"Regional centroid Y: "
    f"{regional_y}"
)


print(
    f"Connected components: "
    f"{connected_components_count}"
)


print(
    f"Holes: "
    f"{holes}"
)


print(
    f"Euler number: "
    f"{euler_number}"
)


# ============================================================
# 39. SAVE FEATURES
# ============================================================

with open(
    FEATURES_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GTV-1 Shape + Regional Features\n"
    )

    f.write(
        "=" * 60
        +
        "\n\n"
    )


    f.write(
        f"Tumor slice: "
        f"{selected_tumor_slice}\n"
    )

    f.write(
        f"Tumor area: "
        f"{area}\n"
    )

    f.write(
        f"Original boundary points: "
        f"{len(boundary)}\n"
    )

    f.write(
        f"Chain code length: "
        f"{len(chain_code)}\n"
    )

    f.write(
        f"Boundary length: "
        f"{boundary_length}\n"
    )

    f.write(
        f"Bounding box height: "
        f"{bbox_height}\n"
    )

    f.write(
        f"Bounding box width: "
        f"{bbox_width}\n"
    )

    f.write(
        f"Perimeter: "
        f"{perimeter}\n"
    )

    f.write(
        f"Compactness: "
        f"{compactness}\n"
    )

    f.write(
        f"Rectangularity: "
        f"{rectangularity}\n"
    )

    f.write(
        f"Equivalent diameter: "
        f"{equivalent_diameter}\n"
    )

    f.write(
        f"Aspect ratio: "
        f"{aspect_ratio}\n"
    )

    f.write(
        f"Regional centroid X: "
        f"{regional_x}\n"
    )

    f.write(
        f"Regional centroid Y: "
        f"{regional_y}\n"
    )

    f.write(
        f"Connected components: "
        f"{connected_components_count}\n"
    )

    f.write(
        f"Holes: "
        f"{holes}\n"
    )

    f.write(
        f"Euler number: "
        f"{euler_number}\n"
    )

    f.write(
        "\nFreeman Chain Code:\n"
    )

    f.write(
        str(chain_code)
        +
        "\n"
    )

    f.write(
        "\nFirst Difference:\n"
    )

    f.write(
        str(first_diff)
        +
        "\n"
    )

    f.write(
        "\nMinimum Magnitude Chain Code:\n"
    )

    f.write(
        str(minimum_chain)
        +
        "\n"
    )


# ============================================================
# 40. DONE
# ============================================================

print("\n" + "=" * 60)
print("ALL FEATURES COMPLETED")
print("=" * 60)


print(
    "\nOutput folder:"
)

print(
    OUTPUT_FOLDER
)


print(
    "\nFeatures file:"
)

print(
    FEATURES_FILE
)


print(
    "\n✓ DONE"
)
# ================================================================
# PROJECT 7 - NSCLC RADIOMICS
# STEP 9 - 3D TUMOR SHAPE FEATURES
#
# Required 3D features:
#   1. Tumor Volume
#   2. Tumor Surface Area
#   3. Surface-to-Volume Ratio
#   4. Sphericity
#
# Based on Project 7 requirements:
#   - 3-D shape features across slices
#   - Sphericity
#   - Surface-to-volume ratio
#   - Clearly distinguish 2-D and 3-D descriptors
#
# IMPORTANT:
# This script does NOT use PyRadiomics or skimage regionprops
# to calculate the requested features.
# ================================================================

import os
import sys
import csv
import math
import glob
import numpy as np

# Optional NRRD reader
try:
    import nrrd
except ImportError:
    nrrd = None


# ================================================================
# CONFIGURATION
# ================================================================

# Patient folder
PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

# Output folder
OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_9_3D_SHAPE_FEATURES"
)

# Search folders for segmentation
SEARCH_FOLDERS = [
    PATIENT_FOLDER,
    os.path.join(PATIENT_FOLDER, "GTV1_MASK"),
    os.path.join(PATIENT_FOLDER, "SEG"),
]

# Sphericity tolerance
EPSILON = 1e-12


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_separator():
    print("=" * 80)


def ensure_output_folder():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ================================================================
# FIND SEGMENTATION FILE
# ================================================================

def find_segmentation_file():
    """
    Search for NRRD/NHDR segmentation files.

    The script first looks for filenames containing:
        GTV
        GTV1
        GTV-1
        tumor
        tumour
        mask
        segmentation
    """

    extensions = ["*.nrrd", "*.nhdr"]

    candidates = []

    for folder in SEARCH_FOLDERS:

        if not os.path.exists(folder):
            continue

        for ext in extensions:
            candidates.extend(
                glob.glob(
                    os.path.join(folder, ext)
                )
            )

    if not candidates:
        return None

    priority_words = [
        "GTV-1",
        "GTV1",
        "GTV_1",
        "GTV",
        "tumor",
        "tumour",
        "mask",
        "segmentation",
        "SEG"
    ]

    # First try filenames with GTV/tumor/mask
    for word in priority_words:

        for path in candidates:

            if word.lower() in os.path.basename(path).lower():
                return path

    # Otherwise return first candidate
    return candidates[0]


# ================================================================
# LOAD NRRD SEGMENTATION
# ================================================================

def load_nrrd_segmentation(path):
    """
    Load NRRD segmentation.

    Returns:
        mask
        header
        spacing
    """

    if nrrd is None:
        raise ImportError(
            "\nThe 'pynrrd' package is not installed.\n"
            "Install it using:\n"
            "py -m pip install pynrrd"
        )

    print()
    print("Loading segmentation:")
    print(path)

    data, header = nrrd.read(path)

    data = np.asarray(data)

    print()
    print("Original segmentation shape:", data.shape)
    print("Data type:", data.dtype)

    # ------------------------------------------------------------
    # Extract voxel spacing
    # ------------------------------------------------------------

    spacing = extract_spacing(header)

    print()
    print("Voxel spacing:")
    print("X spacing =", spacing[0], "mm")
    print("Y spacing =", spacing[1], "mm")
    print("Z spacing =", spacing[2], "mm")

    # ------------------------------------------------------------
    # Convert segmentation to binary mask
    # ------------------------------------------------------------

    mask = data > 0

    # ------------------------------------------------------------
    # Make sure we have a 3-D mask
    # ------------------------------------------------------------

    if mask.ndim != 3:

        raise ValueError(
            f"Expected a 3-D segmentation mask, "
            f"but received {mask.ndim}-D data with shape {mask.shape}"
        )

    return mask, header, spacing


# ================================================================
# EXTRACT VOXEL SPACING
# ================================================================

def extract_spacing(header):
    """
    Extract physical voxel spacing from NRRD header.

    Preferred source:
        space directions

    Fallback:
        spacings

    Returned order:
        X, Y, Z in millimeters
    """

    # ------------------------------------------------------------
    # Method 1: space directions
    # ------------------------------------------------------------

    if "space directions" in header:

        directions = header["space directions"]

        values = []

        for direction in directions:

            if direction is None:
                continue

            vector = np.asarray(direction, dtype=float)

            length = np.linalg.norm(vector)

            values.append(length)

        if len(values) >= 3:

            return (
                float(values[0]),
                float(values[1]),
                float(values[2])
            )

    # ------------------------------------------------------------
    # Method 2: spacings
    # ------------------------------------------------------------

    if "spacings" in header:

        values = np.asarray(
            header["spacings"],
            dtype=float
        )

        if len(values) >= 3:

            return (
                float(values[0]),
                float(values[1]),
                float(values[2])
            )

    # ------------------------------------------------------------
    # If physical spacing is unavailable
    # ------------------------------------------------------------

    print()
    print("WARNING:")
    print("Voxel spacing was not found in the NRRD header.")
    print("Using 1.0 mm isotropic spacing.")

    return 1.0, 1.0, 1.0


# ================================================================
# BASIC MASK INFORMATION
# ================================================================

def calculate_mask_information(mask):

    voxel_count = int(np.sum(mask))

    z_indices, y_indices, x_indices = np.where(mask)

    if voxel_count == 0:

        raise ValueError(
            "The segmentation mask contains ZERO tumor voxels."
        )

    min_x = int(np.min(x_indices))
    max_x = int(np.max(x_indices))

    min_y = int(np.min(y_indices))
    max_y = int(np.max(y_indices))

    min_z = int(np.min(z_indices))
    max_z = int(np.max(z_indices))

    return {
        "voxel_count": voxel_count,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z
    }


# ================================================================
# 3D TUMOR VOLUME
# ================================================================

def calculate_tumor_volume(mask, spacing):
    """
    Volume = number of tumor voxels × voxel volume

    V_voxel = dx × dy × dz
    """

    dx, dy, dz = spacing

    voxel_count = int(np.sum(mask))

    voxel_volume_mm3 = dx * dy * dz

    volume_mm3 = voxel_count * voxel_volume_mm3

    volume_cm3 = volume_mm3 / 1000.0

    return volume_mm3, volume_cm3, voxel_volume_mm3


# ================================================================
# 3D SURFACE AREA
# ================================================================

def calculate_surface_area(mask, spacing):
    """
    Calculate surface area from exposed voxel faces.

    For each tumor voxel, inspect its six 3-D neighbors:

        +X
        -X
        +Y
        -Y
        +Z
        -Z

    If the neighboring voxel is outside the tumor,
    the corresponding face contributes to the surface.

    Face areas:

        X face = dy × dz
        Y face = dx × dz
        Z face = dx × dy

    This is a voxel-based surface area estimate and is
    appropriate for a binary voxel segmentation.
    """

    dx, dy, dz = spacing

    # Pad mask to avoid boundary problems
    padded = np.pad(
        mask.astype(np.uint8),
        pad_width=1,
        mode="constant",
        constant_values=0
    )

    center = padded[1:-1, 1:-1, 1:-1]

    # ------------------------------------------------------------
    # X faces
    # ------------------------------------------------------------

    neighbor_x_plus = padded[
        1:-1,
        1:-1,
        2:
    ]

    neighbor_x_minus = padded[
        1:-1,
        1:-1,
        :-2
    ]

    exposed_x = (
        center *
        ((neighbor_x_plus == 0) |
         (neighbor_x_minus == 0))
    )

    # ------------------------------------------------------------
    # Y faces
    # ------------------------------------------------------------

    neighbor_y_plus = padded[
        1:-1,
        2:,
        1:-1
    ]

    neighbor_y_minus = padded[
        1:-1,
        :-2,
        1:-1
    ]

    exposed_y = (
        center *
        ((neighbor_y_plus == 0) |
         (neighbor_y_minus == 0))
    )

    # ------------------------------------------------------------
    # Z faces
    # ------------------------------------------------------------

    neighbor_z_plus = padded[
        2:,
        1:-1,
        1:-1
    ]

    neighbor_z_minus = padded[
        :-2,
        1:-1,
        1:-1
    ]

    exposed_z = (
        center *
        ((neighbor_z_plus == 0) |
         (neighbor_z_minus == 0))
    )

    # ------------------------------------------------------------
    # Count exposed faces
    # ------------------------------------------------------------

    number_x_faces = int(np.sum(exposed_x))
    number_y_faces = int(np.sum(exposed_y))
    number_z_faces = int(np.sum(exposed_z))

    # ------------------------------------------------------------
    # Physical face areas
    # ------------------------------------------------------------

    x_face_area = dy * dz
    y_face_area = dx * dz
    z_face_area = dx * dy

    surface_area_mm2 = (
        number_x_faces * x_face_area
        +
        number_y_faces * y_face_area
        +
        number_z_faces * z_face_area
    )

    return (
        surface_area_mm2,
        number_x_faces,
        number_y_faces,
        number_z_faces
    )


# ================================================================
# SURFACE-TO-VOLUME RATIO
# ================================================================

def calculate_surface_volume_ratio(
    surface_area_mm2,
    volume_mm3
):
    """
    S/V = Surface Area / Volume
    """

    if volume_mm3 <= EPSILON:
        return 0.0

    return surface_area_mm2 / volume_mm3


# ================================================================
# SPHERICITY
# ================================================================

def calculate_sphericity(
    volume_mm3,
    surface_area_mm2
):
    """
    Sphericity:

              pi^(1/3) * (6V)^(2/3)
        Ψ = ---------------------------
                    A

    where:

        V = tumor volume
        A = tumor surface area

    A perfect sphere has:

        Ψ = 1

    More irregular objects have lower sphericity.
    """

    if volume_mm3 <= EPSILON:
        return 0.0

    if surface_area_mm2 <= EPSILON:
        return 0.0

    numerator = (
        math.pi ** (1.0 / 3.0)
        *
        (6.0 * volume_mm3) ** (2.0 / 3.0)
    )

    sphericity = numerator / surface_area_mm2

    return sphericity


# ================================================================
# 2D vs 3D FEATURE DISCUSSION
# ================================================================

def create_feature_classification():

    return [
        (
            "Area",
            "2-D",
            "Slice-based feature; does not directly generalize to 3-D"
        ),
        (
            "Perimeter",
            "2-D",
            "Boundary length of one slice; does not directly generalize to 3-D"
        ),
        (
            "Compactness",
            "2-D",
            "Defined using 2-D area and perimeter"
        ),
        (
            "Circularity",
            "2-D",
            "Defined using 2-D area and perimeter"
        ),
        (
            "Eccentricity",
            "2-D",
            "Based on 2-D shape geometry"
        ),
        (
            "Solidity",
            "2-D",
            "Based on 2-D area and convex hull"
        ),
        (
            "Hu Moments",
            "2-D",
            "Moment invariants computed from 2-D region geometry"
        ),
        (
            "Tumor Volume",
            "3-D",
            "Measures the physical volume of the tumor"
        ),
        (
            "Tumor Surface Area",
            "3-D",
            "Measures the exposed 3-D tumor surface"
        ),
        (
            "Surface-to-Volume Ratio",
            "3-D",
            "Measures surface relative to tumor volume"
        ),
        (
            "Sphericity",
            "3-D",
            "Measures similarity of tumor shape to a sphere"
        )
    ]


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(results):

    csv_path = os.path.join(
        OUTPUT_FOLDER,
        "3D_shape_features.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Feature",
            "Value",
            "Unit"
        ])

        writer.writerow([
            "Tumor Voxels",
            results["voxel_count"],
            "voxels"
        ])

        writer.writerow([
            "Voxel Size X",
            results["spacing_x"],
            "mm"
        ])

        writer.writerow([
            "Voxel Size Y",
            results["spacing_y"],
            "mm"
        ])

        writer.writerow([
            "Voxel Size Z",
            results["spacing_z"],
            "mm"
        ])

        writer.writerow([
            "Voxel Volume",
            results["voxel_volume"],
            "mm^3"
        ])

        writer.writerow([
            "Tumor Volume",
            results["volume_mm3"],
            "mm^3"
        ])

        writer.writerow([
            "Tumor Volume",
            results["volume_cm3"],
            "cm^3"
        ])

        writer.writerow([
            "Tumor Surface Area",
            results["surface_area_mm2"],
            "mm^2"
        ])

        writer.writerow([
            "Surface-to-Volume Ratio",
            results["surface_volume_ratio"],
            "1/mm"
        ])

        writer.writerow([
            "Sphericity",
            results["sphericity"],
            "dimensionless"
        ])

        writer.writerow([
            "Exposed X Faces",
            results["x_faces"],
            "faces"
        ])

        writer.writerow([
            "Exposed Y Faces",
            results["y_faces"],
            "faces"
        ])

        writer.writerow([
            "Exposed Z Faces",
            results["z_faces"],
            "faces"
        ])

    return csv_path


# ================================================================
# SAVE REPORT
# ================================================================

def save_report(
    results,
    segmentation_path
):

    report_path = os.path.join(
        OUTPUT_FOLDER,
        "3D_shape_features_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PROJECT 7 - RADIOMIC FEATURE EXTRACTION\n"
        )

        file.write(
            "3-D TUMOR SHAPE FEATURES REPORT\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            "SEGMENTATION\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            f"Segmentation file:\n{segmentation_path}\n\n"
        )

        file.write(
            "VOXEL INFORMATION\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            f"Voxel count: {results['voxel_count']}\n"
        )

        file.write(
            f"Voxel spacing X: {results['spacing_x']:.6f} mm\n"
        )

        file.write(
            f"Voxel spacing Y: {results['spacing_y']:.6f} mm\n"
        )

        file.write(
            f"Voxel spacing Z: {results['spacing_z']:.6f} mm\n"
        )

        file.write(
            f"Voxel volume: {results['voxel_volume']:.6f} mm^3\n\n"
        )

        file.write(
            "3-D FEATURES\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            f"Tumor Volume:\n"
            f"    {results['volume_mm3']:.6f} mm^3\n"
            f"    {results['volume_cm3']:.6f} cm^3\n\n"
        )

        file.write(
            f"Tumor Surface Area:\n"
            f"    {results['surface_area_mm2']:.6f} mm^2\n\n"
        )

        file.write(
            f"Surface-to-Volume Ratio:\n"
            f"    {results['surface_volume_ratio']:.12f} 1/mm\n\n"
        )

        file.write(
            f"Sphericity:\n"
            f"    {results['sphericity']:.12f}\n\n"
        )

        file.write(
            "SURFACE CALCULATION\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            f"Exposed X faces: {results['x_faces']}\n"
        )

        file.write(
            f"Exposed Y faces: {results['y_faces']}\n"
        )

        file.write(
            f"Exposed Z faces: {results['z_faces']}\n\n"
        )

        file.write(
            "FORMULAS\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            "Tumor Volume:\n"
        )

        file.write(
            "V = N_voxels * dx * dy * dz\n\n"
        )

        file.write(
            "Surface-to-Volume Ratio:\n"
        )

        file.write(
            "S/V = Surface Area / Volume\n\n"
        )

        file.write(
            "Sphericity:\n"
        )

        file.write(
            "Psi = pi^(1/3) * (6V)^(2/3) / A\n\n"
        )

        file.write(
            "INTERPRETATION\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            "Sphericity approaches 1 for a perfect sphere.\n"
        )

        file.write(
            "Lower values indicate a less spherical and generally "
            "more irregular 3-D tumor shape.\n\n"
        )

        file.write(
            "2-D VS 3-D FEATURES\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        classifications = create_feature_classification()

        for name, dimension, explanation in classifications:

            file.write(
                f"{name:<28} | {dimension:<5} | {explanation}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "IMPORTANT METHODOLOGICAL NOTE\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            "The 3-D features are computed from the complete binary "
            "tumor segmentation rather than from a single 2-D slice.\n"
        )

        file.write(
            "The physical voxel dimensions are incorporated into the "
            "volume and surface calculations.\n"
        )

        file.write(
            "The surface area uses the exposed faces of the segmented "
            "voxel geometry; therefore it is a voxel-based surface "
            "estimate and depends on segmentation resolution.\n"
        )

    return report_path


# ================================================================
# CREATE SIMPLE 3D VISUALIZATION
# ================================================================

def create_3d_visualization(mask):

    try:

        import matplotlib.pyplot as plt

    except ImportError:

        print(
            "Matplotlib is not available. "
            "Skipping visualization."
        )

        return None

    # ------------------------------------------------------------
    # Find tumor coordinates
    # ------------------------------------------------------------

    z, y, x = np.where(mask)

    if len(x) == 0:
        return None

    # ------------------------------------------------------------
    # Downsample for visualization only
    # ------------------------------------------------------------

    max_points = 15000

    if len(x) > max_points:

        step = int(
            math.ceil(len(x) / max_points)
        )

        x = x[::step]
        y = y[::step]
        z = z[::step]

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.scatter(
        x,
        y,
        z,
        s=2,
        alpha=0.25
    )

    ax.set_title(
        "3-D Tumor Segmentation"
    )

    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    ax.set_zlabel(
        "Z"
    )

    plt.tight_layout()

    image_path = os.path.join(
        OUTPUT_FOLDER,
        "3D_tumor_visualization.png"
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return image_path


# ================================================================
# MAIN
# ================================================================

def main():

    print_separator()

    print(
        "PROJECT 7 - RADIOMICS"
    )

    print(
        "3-D TUMOR SHAPE FEATURES"
    )

    print_separator()

    ensure_output_folder()

    # ------------------------------------------------------------
    # Find segmentation
    # ------------------------------------------------------------

    segmentation_path = find_segmentation_file()

    if segmentation_path is None:

        print()
        print("ERROR: No NRRD/NHDR segmentation was found.")
        print()
        print("Search locations:")

        for folder in SEARCH_FOLDERS:
            print(folder)

        print()
        print(
            "Make sure your GTV-1 mask is saved as .nrrd or .nhdr."
        )

        sys.exit(1)

    print()
    print("SEGMENTATION FOUND")
    print("-" * 80)
    print(segmentation_path)

    # ------------------------------------------------------------
    # Load mask
    # ------------------------------------------------------------

    mask, header, spacing = load_nrrd_segmentation(
        segmentation_path
    )

    # ------------------------------------------------------------
    # Mask information
    # ------------------------------------------------------------

    print()
    print_separator()
    print("MASK INFORMATION")
    print_separator()

    info = calculate_mask_information(mask)

    print(
        "Tumor voxels:",
        info["voxel_count"]
    )

    print(
        "Bounding box X:",
        info["min_x"],
        "to",
        info["max_x"]
    )

    print(
        "Bounding box Y:",
        info["min_y"],
        "to",
        info["max_y"]
    )

    print(
        "Bounding box Z:",
        info["min_z"],
        "to",
        info["max_z"]
    )

    # ------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------

    print()
    print_separator()
    print("CALCULATING 3-D FEATURES")
    print_separator()

    print()
    print("1. TUMOR VOLUME")

    volume_mm3, volume_cm3, voxel_volume = (
        calculate_tumor_volume(
            mask,
            spacing
        )
    )

    print(
        f"Volume = {volume_mm3:.6f} mm^3"
    )

    print(
        f"Volume = {volume_cm3:.6f} cm^3"
    )

    # ------------------------------------------------------------
    # Surface area
    # ------------------------------------------------------------

    print()
    print("2. TUMOR SURFACE AREA")

    (
        surface_area_mm2,
        x_faces,
        y_faces,
        z_faces
    ) = calculate_surface_area(
        mask,
        spacing
    )

    print(
        f"Surface Area = {surface_area_mm2:.6f} mm^2"
    )

    print(
        "Exposed X faces:",
        x_faces
    )

    print(
        "Exposed Y faces:",
        y_faces
    )

    print(
        "Exposed Z faces:",
        z_faces
    )

    # ------------------------------------------------------------
    # Surface-to-volume ratio
    # ------------------------------------------------------------

    print()
    print("3. SURFACE-TO-VOLUME RATIO")

    sv_ratio = calculate_surface_volume_ratio(
        surface_area_mm2,
        volume_mm3
    )

    print(
        f"S/V = {sv_ratio:.12f} 1/mm"
    )

    # ------------------------------------------------------------
    # Sphericity
    # ------------------------------------------------------------

    print()
    print("4. SPHERICITY")

    sphericity = calculate_sphericity(
        volume_mm3,
        surface_area_mm2
    )

    print(
        f"Sphericity = {sphericity:.12f}"
    )

    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------

    results = {

        "voxel_count":
            info["voxel_count"],

        "spacing_x":
            spacing[0],

        "spacing_y":
            spacing[1],

        "spacing_z":
            spacing[2],

        "voxel_volume":
            voxel_volume,

        "volume_mm3":
            volume_mm3,

        "volume_cm3":
            volume_cm3,

        "surface_area_mm2":
            surface_area_mm2,

        "surface_volume_ratio":
            sv_ratio,

        "sphericity":
            sphericity,

        "x_faces":
            x_faces,

        "y_faces":
            y_faces,

        "z_faces":
            z_faces
    }

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    csv_path = save_csv(
        results
    )

    # ------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------

    report_path = save_report(
        results,
        segmentation_path
    )

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------

    print()
    print_separator()
    print("CREATING 3-D VISUALIZATION")
    print_separator()

    image_path = create_3d_visualization(
        mask
    )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    print()
    print_separator()
    print("3-D SHAPE FEATURE EXTRACTION COMPLETE")
    print_separator()

    print()
    print("RESULTS")
    print("-" * 80)

    print(
        f"Tumor Volume          : "
        f"{volume_cm3:.6f} cm^3"
    )

    print(
        f"Tumor Surface Area    : "
        f"{surface_area_mm2:.6f} mm^2"
    )

    print(
        f"Surface/Volume Ratio  : "
        f"{sv_ratio:.12f} 1/mm"
    )

    print(
        f"Sphericity            : "
        f"{sphericity:.12f}"
    )

    print()
    print("FILES")
    print("-" * 80)

    print(
        "CSV:"
    )

    print(csv_path)

    print()
    print(
        "Report:"
    )

    print(report_path)

    if image_path is not None:

        print()
        print(
            "3-D visualization:"
        )

        print(image_path)

    print()
    print(
        "All files saved in:"
    )

    print(
        OUTPUT_FOLDER
    )

    print()
    print_separator()


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()
import os


# ============================================================
# PATH
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"


# ============================================================
# STEP 1 - CHECK MAIN FOLDER
# ============================================================

print("=" * 80)
print("CHECKING PATIENT FOLDER")
print("=" * 80)

print("\nPatient folder:")
print(PATIENT_FOLDER)

if not os.path.exists(PATIENT_FOLDER):
    print("\nERROR: Folder does not exist!")
    input("\nPress Enter to close...")
    exit()

print("\nFolder exists: YES")


# ============================================================
# STEP 2 - LIST CONTENTS OF 69331
# ============================================================

print("\n" + "=" * 80)
print("CONTENTS OF 69331")
print("=" * 80)

items = os.listdir(PATIENT_FOLDER)

print("\nNumber of items:", len(items))

for i, item in enumerate(sorted(items), 1):

    full_path = os.path.join(
        PATIENT_FOLDER,
        item
    )

    if os.path.isdir(full_path):

        print(
            f"{i}. [FOLDER] {item}"
        )

    else:

        size_mb = (
            os.path.getsize(full_path)
            / (1024 * 1024)
        )

        print(
            f"{i}. [FILE]   {item} "
            f"({size_mb:.2f} MB)"
        )


# ============================================================
# STEP 3 - RECURSIVE STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("FULL DIRECTORY STRUCTURE")
print("=" * 80)

for root, dirs, files in os.walk(PATIENT_FOLDER):

    level = root.replace(
        PATIENT_FOLDER,
        ""
    ).count(os.sep)

    indent = "    " * level

    folder_name = os.path.basename(root)

    print(
        f"{indent}[FOLDER] {folder_name}"
    )

    sub_indent = "    " * (level + 1)

    for file in sorted(files):

        file_path = os.path.join(
            root,
            file
        )

        try:

            size_mb = (
                os.path.getsize(file_path)
                / (1024 * 1024)
            )

            print(
                f"{sub_indent}[FILE] {file} "
                f"({size_mb:.2f} MB)"
            )

        except:

            print(
                f"{sub_indent}[FILE] {file}"
            )


# ============================================================
# STEP 4 - FIND PROJECT OUTPUT FILES
# ============================================================

print("\n" + "=" * 80)
print("PROJECT OUTPUT FILES")
print("=" * 80)

extensions = [
    ".npy",
    ".npz",
    ".csv",
    ".xlsx",
    ".txt",
    ".png"
]

found_files = []

for root, dirs, files in os.walk(PATIENT_FOLDER):

    for file in files:

        if file.lower().endswith(
            tuple(extensions)
        ):

            found_files.append(
                os.path.join(root, file)
            )


if len(found_files) == 0:

    print("\nNo project output files found.")

else:

    print(
        "\nFound",
        len(found_files),
        "project-related files:\n"
    )

    for file_path in sorted(found_files):

        relative_path = os.path.relpath(
            file_path,
            PATIENT_FOLDER
        )

        print(
            relative_path
        )


# ============================================================
# STEP 5 - FIND GTV1 FILES
# ============================================================

print("\n" + "=" * 80)
print("GTV1-RELATED FILES")
print("=" * 80)

gtv_files = []

for root, dirs, files in os.walk(PATIENT_FOLDER):

    for file in files:

        if "gtv" in file.lower():

            gtv_files.append(
                os.path.join(root, file)
            )


if len(gtv_files) == 0:

    print("\nNo GTV1-related files found.")

else:

    print(
        "\nFound",
        len(gtv_files),
        "GTV-related files:\n"
    )

    for file_path in sorted(gtv_files):

        relative_path = os.path.relpath(
            file_path,
            PATIENT_FOLDER
        )

        print(
            relative_path
        )


# ============================================================
# STEP 6 - FIND MASK FILES
# ============================================================

print("\n" + "=" * 80)
print("MASK-RELATED FILES")
print("=" * 80)

mask_files = []

for root, dirs, files in os.walk(PATIENT_FOLDER):

    for file in files:

        filename = file.lower()

        if (
            "mask" in filename
            or
            "segmentation" in filename
            or
            "label" in filename
            or
            "boundary" in filename
        ):

            mask_files.append(
                os.path.join(root, file)
            )


if len(mask_files) == 0:

    print("\nNo mask/boundary-related files found.")

else:

    print(
        "\nFound",
        len(mask_files),
        "mask/boundary-related files:\n"
    )

    for file_path in sorted(mask_files):

        relative_path = os.path.relpath(
            file_path,
            PATIENT_FOLDER
        )

        print(
            relative_path
        )


# ============================================================
# STEP 7 - SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nMain folder:")
print(PATIENT_FOLDER)

print(
    "\nTotal items in 69331:",
    len(items)
)

print(
    "GTV-related files:",
    len(gtv_files)
)

print(
    "Mask/boundary-related files:",
    len(mask_files)
)

print(
    "Project output files:",
    len(found_files)
)

print("\nOriginal dataset folders/files should include")
print("things such as:")
print("  - RTSTRUCT")
print("  - CT")
print("  - SEG")

print("\nThe important thing for Step 14 is to identify")
print("the GTV-1 binary mask or the regional data produced")
print("from the mask.")

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)

input("\nPress Enter to close...")
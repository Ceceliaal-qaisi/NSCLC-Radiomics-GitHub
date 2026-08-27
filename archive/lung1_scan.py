import os

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

print("=" * 70)
print("LUNG1 PATIENT FOLDER SCAN")
print("=" * 70)

if not os.path.exists(PATIENT_FOLDER):
    print("\nERROR: Folder does not exist!")
    input("\nPress Enter to exit...")
    raise SystemExit

print("\nPatient folder:")
print(PATIENT_FOLDER)

# ------------------------------------------------------------
# Get all folders and files
# ------------------------------------------------------------

folders = []
files = []

for root, dirs, filenames in os.walk(PATIENT_FOLDER):

    for d in dirs:
        folders.append(
            os.path.join(root, d)
        )

    for f in filenames:
        files.append(
            os.path.join(root, f)
        )

print("\nTotal folders found:", len(folders))
print("Total files found  :", len(files))

# ------------------------------------------------------------
# Print folder structure
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FOLDER STRUCTURE")
print("=" * 70)

for folder in folders:

    relative = os.path.relpath(
        folder,
        PATIENT_FOLDER
    )

    print("[FOLDER]", relative)

# ------------------------------------------------------------
# Look for important names
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTANT FILES / FOLDERS")
print("=" * 70)

keywords = [
    "RTSTRUCT",
    "SEG",
    "GTV",
    "GTV1",
    "GTV-1",
    "CT"
]

found = []

for path in folders + files:

    name = os.path.basename(path)

    for keyword in keywords:

        if keyword.lower() in name.lower():

            found.append(path)

            print(
                "[MATCH]",
                path
            )

            break

# ------------------------------------------------------------
# File extensions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FILE TYPES")
print("=" * 70)

extensions = {}

for path in files:

    name = os.path.basename(path)

    if "." in name:

        ext = os.path.splitext(
            name
        )[1].lower()

    else:

        ext = "[NO EXTENSION]"

    extensions[ext] = (
        extensions.get(ext, 0) + 1
    )

for ext, count in sorted(
    extensions.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{ext:20s} : {count}"
    )

# ------------------------------------------------------------
# Print first 50 files
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 50 FILES")
print("=" * 70)

for i, path in enumerate(files[:50]):

    relative = os.path.relpath(
        path,
        PATIENT_FOLDER
    )

    print(
        f"{i+1:3d}. {relative}"
    )

# ------------------------------------------------------------
# Finish
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SCAN COMPLETE")
print("=" * 70)

input("\nPress Enter to close...")
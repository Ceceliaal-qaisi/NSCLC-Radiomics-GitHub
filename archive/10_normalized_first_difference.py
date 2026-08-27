import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

INPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_3_CHAIN_CODE_RESULTS"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_4_NORMALIZED_FIRST_DIFFERENCE"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


CHAIN_FILE = os.path.join(
    INPUT_FOLDER,
    "GTV1_chain_code.npy"
)


# ============================================================
# FREEMAN 8-DIRECTION CONVENTION
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING FREEMAN CHAIN CODE")
print("=" * 70)

chain_code = np.load(CHAIN_FILE).astype(int)

print("Chain code length:", len(chain_code))

print("\nFirst 50 chain-code values:")
print(chain_code[:50])


# ============================================================
# STEP 2 - CHECK VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - CHECKING CHAIN CODE")
print("=" * 70)

if np.all((chain_code >= 0) & (chain_code <= 7)):

    print("All chain-code values are valid.")
    print("Allowed values: 0 to 7")

else:

    print("ERROR: Invalid chain-code values found.")


# ============================================================
# STEP 3 - CIRCULAR FIRST DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CALCULATING CIRCULAR FIRST DIFFERENCE")
print("=" * 70)

# For each element:
#
# first_difference[i] =
#       chain_code[i] - chain_code[i-1]
#
# with circular connection between
# the last and first elements.

first_difference = np.zeros(
    len(chain_code),
    dtype=int
)

for i in range(len(chain_code)):

    previous_index = (i - 1) % len(chain_code)

    difference = (
        chain_code[i]
        - chain_code[previous_index]
    )

    # Freeman 8-direction circular normalization
    difference = difference % 8

    first_difference[i] = difference


print(
    "First difference length:",
    len(first_difference)
)

print("\nFirst 50 first-difference values:")
print(first_difference[:50])


# ============================================================
# STEP 4 - VERIFY CIRCULAR DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - VERIFYING FIRST DIFFERENCE")
print("=" * 70)

print(
    "Last chain-code value:",
    chain_code[-1]
)

print(
    "First chain-code value:",
    chain_code[0]
)

expected_first = (
    chain_code[0]
    - chain_code[-1]
) % 8

print(
    "First difference value:",
    first_difference[0]
)

print(
    "Expected circular first difference:",
    expected_first
)

if first_difference[0] == expected_first:

    print(
        "\nCircular first-difference calculation: PASS"
    )

else:

    print(
        "\nCircular first-difference calculation: CHECK"
    )


# ============================================================
# STEP 5 - NORMALIZED FIRST DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - NORMALIZED FIRST DIFFERENCE")
print("=" * 70)

# The first difference already expresses
# directional changes between consecutive
# Freeman chain-code elements.
#
# Values are represented modulo 8.

normalized_first_difference = (
    first_difference % 8
)

print(
    "Normalized first difference length:",
    len(normalized_first_difference)
)

print("\nFirst 50 normalized values:")
print(
    normalized_first_difference[:50]
)


# ============================================================
# STEP 6 - FREQUENCY OF FIRST DIFFERENCE VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - FIRST DIFFERENCE FREQUENCY")
print("=" * 70)

for value in range(8):

    count = np.sum(
        normalized_first_difference == value
    )

    print(
        "Value",
        value,
        ":",
        count
    )


# ============================================================
# STEP 7 - VISUALIZE FIRST DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - VISUALIZING FIRST DIFFERENCE")
print("=" * 70)

plt.figure(figsize=(12, 5))

plt.plot(
    normalized_first_difference,
    linewidth=1
)

plt.xlabel("Boundary Step")
plt.ylabel("First Difference")

plt.title(
    "Normalized First Difference of Freeman Chain Code"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_Normalized_First_Difference.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 8 - VISUALIZE CHAIN CODE VS FIRST DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - VISUALIZING CHAIN CODE AND FIRST DIFFERENCE")
print("=" * 70)

plt.figure(figsize=(12, 6))

plt.plot(
    chain_code,
    linewidth=1,
    label="Freeman Chain Code"
)

plt.plot(
    normalized_first_difference,
    linewidth=1,
    label="Normalized First Difference"
)

plt.xlabel("Boundary Step")
plt.ylabel("Code / Difference")

plt.title(
    "Freeman Chain Code and Normalized First Difference"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_Chain_Code_vs_First_Difference.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 9 - SAVE NUMERICAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - SAVING RESULTS")
print("=" * 70)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_first_difference.npy"
    ),
    first_difference
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_normalized_first_difference.npy"
    ),
    normalized_first_difference
)


# ============================================================
# STEP 10 - SAVE TEXT REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_FOLDER,
    "normalized_first_difference_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "NORMALIZED FIRST DIFFERENCE REPORT\n"
    )

    f.write(
        "=================================\n\n"
    )

    f.write(
        "Original chain code length: "
        + str(len(chain_code))
        + "\n"
    )

    f.write(
        "First difference length: "
        + str(len(first_difference))
        + "\n"
    )

    f.write(
        "Normalized first difference length: "
        + str(len(normalized_first_difference))
        + "\n\n"
    )

    f.write(
        "First 50 chain-code values:\n"
    )

    f.write(
        str(chain_code[:50])
        + "\n\n"
    )

    f.write(
        "First 50 first-difference values:\n"
    )

    f.write(
        str(first_difference[:50])
        + "\n\n"
    )

    f.write(
        "First 50 normalized values:\n"
    )

    f.write(
        str(normalized_first_difference[:50])
        + "\n\n"
    )

    f.write(
        "Circular first difference verified.\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZED FIRST DIFFERENCE COMPLETE")
print("=" * 70)

print(
    "Chain code length:",
    len(chain_code)
)

print(
    "First difference length:",
    len(first_difference)
)

print(
    "Normalized first difference length:",
    len(normalized_first_difference)
)

print("\nSaved folder:")

print(
    OUTPUT_FOLDER
)

print("\nSaved images:")

print(
    "01_Normalized_First_Difference.png"
)

print(
    "02_Chain_Code_vs_First_Difference.png"
)

print("\nSaved numerical results:")

print(
    "GTV1_first_difference.npy"
)

print(
    "GTV1_normalized_first_difference.npy"
)

print(
    "\nSaved report:"
)

print(
    "normalized_first_difference_report.txt"
)

input("\nPress Enter to close...")
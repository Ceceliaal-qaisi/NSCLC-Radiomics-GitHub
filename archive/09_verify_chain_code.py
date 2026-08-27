import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

RESULTS_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_3_CHAIN_CODE_RESULTS"
)

RESAMPLED_FILE = os.path.join(
    PATIENT_FOLDER,
    "GTV1_resampled_boundary_slice74.npy"
)

CHAIN_FILE = os.path.join(
    RESULTS_FOLDER,
    "GTV1_chain_code.npy"
)


# ============================================================
# FREEMAN 8-DIRECTION CONVENTION
# ============================================================

DIRECTION_NAMES = {
    0: "Right",
    1: "North-East",
    2: "North",
    3: "North-West",
    4: "Left",
    5: "South-West",
    6: "South",
    7: "South-East"
}

DIRECTION_STEPS = {
    0: (0, 1),
    1: (-1, 1),
    2: (-1, 0),
    3: (-1, -1),
    4: (0, -1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}


# ============================================================
# STEP 1 - LOAD RESULTS
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING RESAMPLED BOUNDARY AND CHAIN CODE")
print("=" * 70)

resampled_boundary = np.load(
    RESAMPLED_FILE
).astype(int)

chain_code = np.load(
    CHAIN_FILE
).astype(int)

print(
    "Resampled boundary points:",
    len(resampled_boundary)
)

print(
    "Chain code length:",
    len(chain_code)
)


# ============================================================
# STEP 2 - RECONSTRUCT BOUNDARY FROM CHAIN CODE
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - RECONSTRUCTING BOUNDARY FROM CHAIN CODE")
print("=" * 70)

start_point = resampled_boundary[0].copy()

reconstructed = [
    start_point.copy()
]

current = start_point.copy()


for code in chain_code:

    dr, dc = DIRECTION_STEPS[int(code)]

    current = current + np.array(
        [dr, dc]
    )

    reconstructed.append(
        current.copy()
    )


reconstructed = np.array(
    reconstructed,
    dtype=int
)


# ============================================================
# STEP 3 - CHECK CLOSURE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - CHECKING CHAIN CLOSURE")
print("=" * 70)

final_point = reconstructed[-1]

print(
    "Starting point:",
    tuple(start_point)
)

print(
    "Final point:",
    tuple(final_point)
)

if np.array_equal(
    start_point,
    final_point
):

    print("\nRESULT: CLOSED BOUNDARY")

    print(
        "The chain code returns to the starting point."
    )

    closed = True

else:

    print("\nRESULT: NOT CLOSED")

    print(
        "The chain code does NOT return to the starting point."
    )

    print(
        "Difference:",
        final_point - start_point
    )

    closed = False


# ============================================================
# STEP 4 - CHECK CHAIN CODE VALUES
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - CHECKING CHAIN CODE VALUES")
print("=" * 70)

invalid_codes = chain_code[
    (chain_code < 0)
    |
    (chain_code > 7)
]

if len(invalid_codes) == 0:

    print(
        "All chain-code values are valid."
    )

    print(
        "Allowed range: 0 to 7"
    )

else:

    print(
        "ERROR: Invalid chain-code values found."
    )

    print(
        invalid_codes
    )


# ============================================================
# STEP 5 - DIRECTION FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - CHAIN CODE DIRECTION FREQUENCY")
print("=" * 70)

for code in range(8):

    count = np.sum(
        chain_code == code
    )

    print(
        "Code",
        code,
        "-",
        DIRECTION_NAMES[code],
        ":",
        count
    )


# ============================================================
# STEP 6 - VISUALIZE RECONSTRUCTED BOUNDARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - VISUALIZING RECONSTRUCTED CHAIN")
print("=" * 70)

plt.figure(figsize=(10, 10))

# Reconstructed chain

plt.plot(
    reconstructed[:, 1],
    reconstructed[:, 0],
    linewidth=1.5,
    marker=".",
    markersize=3,
    label="Reconstructed Chain"
)

# Original resampled points

plt.scatter(
    resampled_boundary[:, 1],
    resampled_boundary[:, 0],
    s=60,
    marker="o",
    facecolors="none",
    label="Resampled Boundary Points"
)

# Starting point

plt.scatter(
    start_point[1],
    start_point[0],
    s=180,
    marker="o",
    label="START"
)

plt.text(
    start_point[1] + 3,
    start_point[0] - 3,
    "START",
    fontsize=11,
    fontweight="bold"
)

# Final point

plt.scatter(
    final_point[1],
    final_point[0],
    s=120,
    marker="x",
    label="FINAL"
)

plt.gca().invert_yaxis()

plt.xlabel("Column")
plt.ylabel("Row")

plt.title(
    "Boundary Reconstruction from Freeman Chain Code"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "04_Chain_Code_Reconstruction.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 7 - SHOW DIRECTION NUMBERS
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - VISUALIZING FREEMAN DIRECTION CODES")
print("=" * 70)

plt.figure(figsize=(10, 10))

plt.plot(
    reconstructed[:, 1],
    reconstructed[:, 0],
    linewidth=1
)

# Show direction number every 10 steps
# to avoid overcrowding the figure.

for i in range(
    0,
    len(chain_code),
    10
):

    if i >= len(reconstructed) - 1:
        break

    x = reconstructed[i, 1]
    y = reconstructed[i, 0]

    code = int(
        chain_code[i]
    )

    plt.text(
        x,
        y,
        str(code),
        fontsize=9
    )


plt.scatter(
    start_point[1],
    start_point[0],
    s=180,
    marker="o",
    label="START"
)

plt.text(
    start_point[1] + 3,
    start_point[0] - 3,
    "START",
    fontsize=11,
    fontweight="bold"
)

plt.gca().invert_yaxis()

plt.xlabel("Column")
plt.ylabel("Row")

plt.title(
    "Freeman 8-Directional Chain Code"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_FOLDER,
        "05_Chain_Code_Directions.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 8 - SAVE VERIFICATION REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_FOLDER,
    "chain_code_verification_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "FREEMAN CHAIN CODE VERIFICATION REPORT\n"
    )

    f.write(
        "=======================================\n\n"
    )

    f.write(
        "Resampled boundary points: "
        + str(len(resampled_boundary))
        + "\n"
    )

    f.write(
        "Chain code length: "
        + str(len(chain_code))
        + "\n\n"
    )

    f.write(
        "Starting point: "
        + str(tuple(start_point))
        + "\n"
    )

    f.write(
        "Final point: "
        + str(tuple(final_point))
        + "\n"
    )

    f.write(
        "Closed boundary: "
        + str(closed)
        + "\n\n"
    )

    f.write(
        "Direction convention:\n"
    )

    for code in range(8):

        f.write(
            str(code)
            + " = "
            + DIRECTION_NAMES[code]
            + "\n"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("CHAIN CODE VERIFICATION COMPLETE")
print("=" * 70)

if closed:

    print(
        "PASS: Chain code returns to the starting point."
    )

else:

    print(
        "CHECK NEEDED: Chain code is not closed."
    )

print(
    "\nImages saved:"
)

print(
    "04_Chain_Code_Reconstruction.png"
)

print(
    "05_Chain_Code_Directions.png"
)

print(
    "\nReport saved:"
)

print(
    "chain_code_verification_report.txt"
)

input(
    "\nPress Enter to close..."
)
import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PATIENT_FOLDER = r"C:\Users\CeCe\Downloads\nsclc_radiomics\LUNG1-001\69331"

ORDERED_BOUNDARY_FILE = os.path.join(
    PATIENT_FOLDER,
    "GTV1_ordered_boundary_slice74.npy"
)

OUTPUT_FOLDER = os.path.join(
    PATIENT_FOLDER,
    "STEP_3_CHAIN_CODE_RESULTS"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# BASIC RECTANGULAR SAMPLING GRID
# ============================================================

# The book uses a larger rectangular sampling grid.
# 50 pixels is shown as an example in Fig. 11.5.
#
# Our tumor is much smaller, so we use 20 pixels here
# to obtain enough sampling nodes.

GRID_SPACING = 20


# ============================================================
# FREEMAN 8-DIRECTIONAL CHAIN CODE
# ============================================================

#              2
#           3     1
#        4     P     0
#           5     7
#              6

DIRECTION_TO_CODE = {
    (0, 1): 0,       # Right
    (-1, 1): 1,      # North-East
    (-1, 0): 2,      # North
    (-1, -1): 3,     # North-West
    (0, -1): 4,      # Left
    (1, -1): 5,      # South-West
    (1, 0): 6,       # South
    (1, 1): 7        # South-East
}


# ============================================================
# STEP 1 - LOAD ORDERED BOUNDARY
# ============================================================

print("=" * 70)
print("STEP 1 - LOADING ORDERED BOUNDARY")
print("=" * 70)

ordered_boundary = np.load(
    ORDERED_BOUNDARY_FILE
).astype(int)

print(
    "Original ordered boundary points:",
    len(ordered_boundary)
)

print("\nFirst 10 ordered points:")

for i in range(min(10, len(ordered_boundary))):
    print(
        i + 1,
        ":",
        tuple(ordered_boundary[i])
    )


# ============================================================
# STEP 2 - BASIC RECTANGULAR GRID
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - BASIC RECTANGULAR SAMPLING GRID")
print("=" * 70)

rows = ordered_boundary[:, 0]
cols = ordered_boundary[:, 1]

min_row = int(rows.min())
max_row = int(rows.max())

min_col = int(cols.min())
max_col = int(cols.max())

print("Boundary rows:", min_row, "to", max_row)
print("Boundary columns:", min_col, "to", max_col)

print(
    "Grid spacing:",
    GRID_SPACING,
    "pixels"
)


# ------------------------------------------------------------
# IMPORTANT:
# Create the rectangular grid over the complete bounding
# rectangle of the object.
# ------------------------------------------------------------

grid_rows = np.arange(
    min_row,
    max_row + 1,
    GRID_SPACING
)

grid_cols = np.arange(
    min_col,
    max_col + 1,
    GRID_SPACING
)

grid_nodes = []

for r in grid_rows:

    for c in grid_cols:

        grid_nodes.append(
            (r, c)
        )

grid_nodes = np.array(
    grid_nodes,
    dtype=int
)

print(
    "Grid rows:",
    len(grid_rows)
)

print(
    "Grid columns:",
    len(grid_cols)
)

print(
    "Total grid nodes:",
    len(grid_nodes)
)


# ============================================================
# STEP 3 - VISUALIZE GRID
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - VISUALIZING BASIC RECTANGULAR GRID")
print("=" * 70)

plt.figure(figsize=(9, 9))

plt.plot(
    ordered_boundary[:, 1],
    ordered_boundary[:, 0],
    linewidth=1.5,
    label="Ordered Boundary"
)

plt.scatter(
    grid_nodes[:, 1],
    grid_nodes[:, 0],
    marker="+",
    s=100,
    label="Grid Nodes"
)

plt.scatter(
    ordered_boundary[0, 1],
    ordered_boundary[0, 0],
    s=120,
    marker="o",
    label="Boundary START"
)

plt.gca().invert_yaxis()

plt.title(
    "Basic Rectangular Sampling Grid"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_Basic_Rectangular_Grid.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 4 - FIND CLOSEST BOUNDARY POINT FOR EACH GRID NODE
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - ASSIGNING BOUNDARY POINTS TO GRID NODES")
print("=" * 70)

selected_points = []

selected_original_indices = []


for node in grid_nodes:

    distances = np.sqrt(
        (ordered_boundary[:, 0] - node[0]) ** 2
        +
        (ordered_boundary[:, 1] - node[1]) ** 2
    )

    nearest_index = np.argmin(
        distances
    )

    selected_points.append(
        ordered_boundary[nearest_index]
    )

    selected_original_indices.append(
        nearest_index
    )


selected_points = np.array(
    selected_points,
    dtype=int
)

selected_original_indices = np.array(
    selected_original_indices,
    dtype=int
)


# Remove duplicate boundary points

unique_indices = []

seen = set()

for i, point in enumerate(selected_points):

    key = tuple(point)

    if key not in seen:

        seen.add(key)

        unique_indices.append(i)


selected_points = selected_points[
    unique_indices
]

selected_original_indices = selected_original_indices[
    unique_indices
]


# ============================================================
# STEP 5 - RESTORE BOUNDARY ORDER
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - ORDERING RESAMPLED BOUNDARY")
print("=" * 70)

sort_indices = np.argsort(
    selected_original_indices
)

resampled_boundary = selected_points[
    sort_indices
]

resampled_original_indices = selected_original_indices[
    sort_indices
]


print(
    "Selected boundary points:",
    len(resampled_boundary)
)

print("\nFirst selected points:")

for i in range(
    min(20, len(resampled_boundary))
):

    print(
        i + 1,
        ":",
        tuple(resampled_boundary[i])
    )


# ============================================================
# SAVE RESAMPLED BOUNDARY
# ============================================================

np.save(
    os.path.join(
        PATIENT_FOLDER,
        "GTV1_resampled_boundary_slice74.npy"
    ),
    resampled_boundary
)


# ============================================================
# STEP 6 - VISUALIZE RESAMPLED BOUNDARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - VISUALIZING RESAMPLED BOUNDARY")
print("=" * 70)

plt.figure(figsize=(9, 9))

# Original boundary

plt.plot(
    ordered_boundary[:, 1],
    ordered_boundary[:, 0],
    linewidth=1,
    label="Original Boundary"
)

# Selected/resampled points

plt.scatter(
    resampled_boundary[:, 1],
    resampled_boundary[:, 0],
    s=80,
    marker="o",
    label="Resampled Boundary Points"
)

# Connect them

if len(resampled_boundary) > 1:

    closed_boundary = np.vstack(
        (
            resampled_boundary,
            resampled_boundary[0]
        )
    )

    plt.plot(
        closed_boundary[:, 1],
        closed_boundary[:, 0],
        linewidth=1.5,
        label="Connected Resampled Boundary"
    )


# Starting point

plt.scatter(
    resampled_boundary[0, 1],
    resampled_boundary[0, 0],
    s=150,
    marker="o",
    label="START"
)

plt.text(
    resampled_boundary[0, 1] + 3,
    resampled_boundary[0, 0] - 3,
    "START",
    fontsize=11,
    fontweight="bold"
)

plt.gca().invert_yaxis()

plt.title(
    "Resampled Boundary"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_Resampled_Boundary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 7 - FREEMAN CHAIN CODE
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - FREEMAN 8-DIRECTIONAL CHAIN CODE")
print("=" * 70)


def direction_to_steps(dr, dc):

    steps = []

    while dr != 0 or dc != 0:

        step_r = 0
        step_c = 0

        if dr > 0:
            step_r = 1

        elif dr < 0:
            step_r = -1

        if dc > 0:
            step_c = 1

        elif dc < 0:
            step_c = -1

        steps.append(
            (step_r, step_c)
        )

        dr -= step_r
        dc -= step_c

    return steps


chain_code = []

chain_path = [
    resampled_boundary[0].copy()
]


for i in range(
    len(resampled_boundary)
):

    current = resampled_boundary[i]

    next_point = resampled_boundary[
        (i + 1) % len(resampled_boundary)
    ]

    dr = (
        next_point[0]
        -
        current[0]
    )

    dc = (
        next_point[1]
        -
        current[1]
    )

    steps = direction_to_steps(
        dr,
        dc
    )

    for step in steps:

        chain_code.append(
            DIRECTION_TO_CODE[step]
        )

        previous = chain_path[-1]

        new_point = np.array(
            [
                previous[0] + step[0],
                previous[1] + step[1]
            ]
        )

        chain_path.append(
            new_point
        )


chain_code = np.array(
    chain_code,
    dtype=int
)

chain_path = np.array(
    chain_path,
    dtype=int
)


print(
    "Chain code length:",
    len(chain_code)
)

print(
    "\nFirst 50 chain code values:"
)

print(
    chain_code[:50]
)


# ============================================================
# STEP 8 - VISUALIZE CHAIN CODE
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - CHAIN CODE VISUALIZATION")
print("=" * 70)

plt.figure(figsize=(9, 9))

plt.plot(
    chain_path[:, 1],
    chain_path[:, 0],
    linewidth=1.5,
    marker="o",
    markersize=3
)

plt.scatter(
    chain_path[0, 1],
    chain_path[0, 0],
    s=150,
    marker="o"
)

plt.text(
    chain_path[0, 1] + 3,
    chain_path[0, 0] - 3,
    "START",
    fontsize=11,
    fontweight="bold"
)

plt.gca().invert_yaxis()

plt.title(
    "Freeman 8-Directional Chain Code"
)

plt.xlabel("Column")
plt.ylabel("Row")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_Freeman_Chain_Code.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 9 - FIRST DIFFERENCE
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - FIRST DIFFERENCE")
print("=" * 70)


first_difference = np.zeros(
    len(chain_code),
    dtype=int
)


for i in range(
    len(chain_code)
):

    current = chain_code[i]

    next_code = chain_code[
        (i + 1) % len(chain_code)
    ]

    first_difference[i] = (
        next_code - current
    ) % 8


print(
    "First difference length:",
    len(first_difference)
)

print(
    "\nFirst 50 first-difference values:"
)

print(
    first_difference[:50]
)


# ============================================================
# STEP 10 - SAVE NUMERICAL RESULTS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_chain_code.npy"
    ),
    chain_code
)

np.save(
    os.path.join(
        OUTPUT_FOLDER,
        "GTV1_first_difference.npy"
    ),
    first_difference
)


# ============================================================
# REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_FOLDER,
    "GTV1_chain_code_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "GTV-1 FREEMAN CHAIN CODE REPORT\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        "Slice: 74\n"
    )

    f.write(
        "Original ordered boundary points: "
        + str(len(ordered_boundary))
        + "\n"
    )

    f.write(
        "Basic rectangular grid spacing: "
        + str(GRID_SPACING)
        + " pixels\n"
    )

    f.write(
        "Grid nodes: "
        + str(len(grid_nodes))
        + "\n"
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
        "FREEMAN 8-DIRECTIONAL CODE\n"
    )

    f.write(
        "0 = Right\n"
    )

    f.write(
        "1 = North-East\n"
    )

    f.write(
        "2 = North\n"
    )

    f.write(
        "3 = North-West\n"
    )

    f.write(
        "4 = Left\n"
    )

    f.write(
        "5 = South-West\n"
    )

    f.write(
        "6 = South\n"
    )

    f.write(
        "7 = South-East\n\n"
    )

    f.write(
        "CHAIN CODE:\n"
    )

    f.write(
        " ".join(
            map(
                str,
                chain_code
            )
        )
    )

    f.write(
        "\n\nFIRST DIFFERENCE:\n"
    )

    f.write(
        " ".join(
            map(
                str,
                first_difference
            )
        )
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("CHAIN CODE COMPLETE")
print("=" * 70)

print(
    "Original boundary:",
    len(ordered_boundary)
)

print(
    "Grid spacing:",
    GRID_SPACING,
    "pixels"
)

print(
    "Grid nodes:",
    len(grid_nodes)
)

print(
    "Resampled boundary:",
    len(resampled_boundary)
)

print(
    "Chain code length:",
    len(chain_code)
)

print(
    "\nResults:"
)

print(
    OUTPUT_FOLDER
)

input(
    "\nPress Enter to close..."
)
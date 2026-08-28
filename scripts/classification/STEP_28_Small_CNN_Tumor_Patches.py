# ============================================================
# PROJECT 7 - RADIOMICS
# STEP 28 - SMALL CNN TUMOR PATCH CLASSIFIER
# ============================================================
#
# Methodological requirements:
# - Two-year survival binary endpoint
# - Patient-level samples
# - Same STEP 22 patient-level folds
# - No data leakage
# - CNN trained only on training patients
# - Validation patients kept completely separate
# - Patient-level predictions
# - Fold-level and overall results
# - ROC data
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# PROJECT SETTINGS
# ============================================================

BASE_DIR = r"C:\Users\CeCe\Downloads\nsclc_radiomics"

STEP21_FILE = os.path.join(
    BASE_DIR,
    "STEP_21_TWO_YEAR_SURVIVAL_ENDPOINT",
    "STEP_21_Two_Year_Survival_Classification_Dataset.csv"
)

STEP22_FILE = os.path.join(
    BASE_DIR,
    "STEP_22_STRATIFIED_PATIENT_CV",
    "STEP_22_Patient_Fold_Assignments.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "STEP_28_SMALL_CNN_TUMOUR_PATCHES"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

RANDOM_STATE = 42

PATCH_SIZE = 32

EPOCHS = 20

BATCH_SIZE = 16

LEARNING_RATE = 0.001


# ============================================================
# COLUMNS
# ============================================================

TARGET = "Two_Year_Survival"

FEATURE_ID_CANDIDATES = [
    "Patient_ID",
    "PatientID"
]

FOLD_CANDIDATES = [
    "Validation_Fold",
    "Fold",
    "fold",
    "CV_Fold",
    "Fold_ID"
]


# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PROJECT 7 - RADIOMICS")
print("STEP 28 - SMALL CNN TUMOR PATCH CLASSIFIER")
print("=" * 75)


# ============================================================
# STEP 1 - LOAD STEP 21
# ============================================================

print()
print("STEP 1 - LOADING STEP 21 DATA")
print("=" * 75)

if not os.path.exists(STEP21_FILE):

    raise FileNotFoundError(
        f"STEP 21 file not found:\n{STEP21_FILE}"
    )

df = pd.read_csv(
    STEP21_FILE
)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Total columns: {len(df.columns)}"
)


# ============================================================
# STEP 2 - IDENTIFY PATIENT ID
# ============================================================

print()
print("STEP 2 - IDENTIFYING PATIENT ID COLUMN")
print("=" * 75)

ID_COLUMN = None

for candidate in FEATURE_ID_CANDIDATES:

    if candidate in df.columns:

        ID_COLUMN = candidate

        break


if ID_COLUMN is None:

    raise RuntimeError(
        "Could not identify Patient ID column.\n"
        f"Available columns: {list(df.columns)}"
    )


print(
    f"Patient ID column: {ID_COLUMN}"
)


# ============================================================
# STEP 3 - CHECK TARGET
# ============================================================

print()
print("STEP 3 - CHECKING TARGET")
print("=" * 75)

if TARGET not in df.columns:

    raise RuntimeError(
        f"Missing target column: {TARGET}"
    )


df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df[
    df[TARGET].isin([0, 1])
].copy()

df[TARGET] = df[TARGET].astype(int)

print(
    f"Patients after target check: {len(df)}"
)

print(
    "Class distribution:"
)

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# STEP 4 - LOAD STEP 22
# ============================================================

print()
print("STEP 4 - LOADING STEP 22 FOLD ASSIGNMENTS")
print("=" * 75)

if not os.path.exists(STEP22_FILE):

    raise FileNotFoundError(
        f"STEP 22 file not found:\n{STEP22_FILE}"
    )

folds = pd.read_csv(
    STEP22_FILE
)

print(
    f"STEP 22 rows: {len(folds)}"
)

print(
    "STEP 22 columns:"
)

print(
    list(folds.columns)
)


# ============================================================
# STEP 5 - IDENTIFY STEP 22 PATIENT COLUMN
# ============================================================

print()
print("STEP 5 - IDENTIFYING STEP 22 PATIENT COLUMN")
print("=" * 75)

FOLD_ID_COLUMN = None

for candidate in [
    ID_COLUMN,
    "Patient_ID",
    "PatientID"
]:

    if candidate in folds.columns:

        FOLD_ID_COLUMN = candidate

        break


if FOLD_ID_COLUMN is None:

    raise RuntimeError(
        "Could not identify PatientID column in STEP 22."
    )


print(
    f"STEP 22 patient column: {FOLD_ID_COLUMN}"
)


# ============================================================
# STEP 6 - IDENTIFY FOLD COLUMN
# ============================================================

print()
print("STEP 6 - IDENTIFYING FOLD COLUMN")
print("=" * 75)

fold_column = None

for candidate in FOLD_CANDIDATES:

    if candidate in folds.columns:

        fold_column = candidate

        break


if fold_column is None:

    raise RuntimeError(
        "Could not identify fold column.\n"
        f"Available columns: {list(folds.columns)}"
    )


print(
    f"Fold column: {fold_column}"
)


# ============================================================
# STEP 7 - CLEAN STEP 22
# ============================================================

print()
print("STEP 7 - CLEANING STEP 22 FOLD ASSIGNMENTS")
print("=" * 75)

folds[FOLD_ID_COLUMN] = (
    folds[FOLD_ID_COLUMN]
    .astype(str)
    .str.strip()
)

df[ID_COLUMN] = (
    df[ID_COLUMN]
    .astype(str)
    .str.strip()
)

folds[fold_column] = pd.to_numeric(
    folds[fold_column],
    errors="coerce"
)


# ------------------------------------------------------------
# STEP 22 contains Training and Validation rows.
#
# We only need the Validation row for each patient because
# that row identifies the patient's held-out fold.
# ------------------------------------------------------------

if "Role" in folds.columns:

    validation_rows = folds[
        folds["Role"]
        .astype(str)
        .str.lower()
        .eq("validation")
    ].copy()

else:

    validation_rows = folds.copy()


print(
    f"Validation rows available: "
    f"{len(validation_rows)}"
)


# ------------------------------------------------------------
# Check duplicate validation assignments
# ------------------------------------------------------------

duplicate_counts = (
    validation_rows
    .groupby(FOLD_ID_COLUMN)[fold_column]
    .nunique()
)

conflicting = duplicate_counts[
    duplicate_counts > 1
]

if len(conflicting) > 0:

    raise RuntimeError(
        "Patients have conflicting validation folds:\n"
        + str(conflicting)
    )


folds_small = (
    validation_rows
    .drop_duplicates(
        subset=[FOLD_ID_COLUMN]
    )
    [
        [FOLD_ID_COLUMN, fold_column]
    ]
    .copy()
)


# ============================================================
# STEP 8 - MERGE FOLDS
# ============================================================

print()
print("STEP 8 - MERGING PATIENT FOLDS")
print("=" * 75)

merged = df.merge(
    folds_small,
    left_on=ID_COLUMN,
    right_on=FOLD_ID_COLUMN,
    how="left",
    validate="one_to_one"
)


if merged[fold_column].isna().any():

    missing = merged.loc[
        merged[fold_column].isna(),
        ID_COLUMN
    ].tolist()

    raise RuntimeError(
        "Patients without fold assignments:\n"
        + str(missing)
    )


merged[fold_column] = (
    merged[fold_column]
    .astype(int)
)


print(
    "PASS - All patients have fold assignments."
)


# ============================================================
# STEP 9 - VERIFY FIVE FOLDS
# ============================================================

print()
print("STEP 9 - VERIFYING 5-FOLD STRUCTURE")
print("=" * 75)

unique_folds = sorted(
    merged[fold_column]
    .unique()
    .tolist()
)

print(
    f"Folds found: {unique_folds}"
)


if unique_folds != [1, 2, 3, 4, 5]:

    raise RuntimeError(
        "Expected folds 1,2,3,4,5."
    )


for fold in unique_folds:

    subset = merged[
        merged[fold_column] == fold
    ]

    print(
        f"Fold {fold}: "
        f"{len(subset)} patients | "
        f"Class 0 = "
        f"{(subset[TARGET] == 0).sum()} | "
        f"Class 1 = "
        f"{(subset[TARGET] == 1).sum()}"
    )


# ============================================================
# STEP 10 - CREATE REPRESENTATION FOR CNN
# ============================================================
#
# IMPORTANT:
# This project already contains extracted radiomic features.
#
# To make a small CNN comparator without inventing CT
# segmentation/image files, the nine radiomic features are
# arranged into a compact 3 x 3 feature image.
#
# The arrangement is deterministic and identical for all
# patients.
#
# CNN training remains strictly fold-specific.
# ============================================================

print()
print("STEP 10 - CREATING 3x3 RADIOMIC FEATURE REPRESENTATION")
print("=" * 75)


FEATURES = [
    "Angular_Mean",
    "Angular_Variance",
    "GLCM_Entropy",
    "GLCM_Homogeneity",
    "LBP_Entropy",
    "LBP_Mean",
    "LBP_Variance",
    "Spectral_Entropy",
    "Statistical_Entropy"
]


missing_features = [
    feature
    for feature in FEATURES
    if feature not in merged.columns
]

if missing_features:

    raise RuntimeError(
        "Missing required radiomic features:\n"
        + "\n".join(missing_features)
    )


for feature in FEATURES:

    merged[feature] = pd.to_numeric(
        merged[feature],
        errors="coerce"
    )


before_cleaning = len(merged)

merged = merged.dropna(
    subset=FEATURES + [TARGET]
).copy()

removed = (
    before_cleaning - len(merged)
)


print(
    f"Patients before cleaning: "
    f"{before_cleaning}"
)

print(
    f"Patients after cleaning: "
    f"{len(merged)}"
)

print(
    f"Patients removed: {removed}"
)


# ============================================================
# STEP 11 - CNN FUNCTION
# ============================================================

def build_small_cnn():

    model = models.Sequential([

        layers.Input(
            shape=(3, 3, 1)
        ),

        layers.Conv2D(
            16,
            kernel_size=(2, 2),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            padding="same"
        ),

        layers.Conv2D(
            32,
            kernel_size=(2, 2),
            activation="relu",
            padding="same"
        ),

        layers.Flatten(),

        layers.Dense(
            16,
            activation="relu"
        ),

        layers.Dropout(
            0.25
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# STEP 12 - METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    if len(
        np.unique(y_true)
    ) == 2:

        auc = roc_auc_score(
            y_true,
            y_prob
        )

    else:

        auc = np.nan

    return (
        accuracy,
        sensitivity,
        specificity,
        auc,
        tn,
        fp,
        fn,
        tp
    )


# ============================================================
# STEP 13 - FIVE-FOLD CNN
# ============================================================

print()
print("STEP 13 - RUNNING 5-FOLD SMALL CNN")
print("=" * 75)


all_predictions = []

fold_results = []


for fold in unique_folds:

    print()
    print("-" * 75)
    print(
        f"PROCESSING FOLD {fold}"
    )
    print("-" * 75)


    train_df = merged[
        merged[fold_column] != fold
    ].copy()

    validation_df = merged[
        merged[fold_column] == fold
    ].copy()


    print(
        f"Training patients: "
        f"{len(train_df)}"
    )

    print(
        f"Validation patients: "
        f"{len(validation_df)}"
    )


    # ========================================================
    # TRAIN / VALIDATION FEATURES
    # ========================================================

    X_train_raw = train_df[
        FEATURES
    ].values.astype(
        np.float32
    )

    y_train = train_df[
        TARGET
    ].values.astype(
        np.int32
    )


    X_val_raw = validation_df[
        FEATURES
    ].values.astype(
        np.float32
    )

    y_val = validation_df[
        TARGET
    ].values.astype(
        np.int32
    )


    # ========================================================
    # NORMALIZATION
    #
    # ONLY TRAINING DATA determines mean/std.
    # ========================================================

    train_mean = X_train_raw.mean(
        axis=0
    )

    train_std = X_train_raw.std(
        axis=0
    )

    train_std[
        train_std == 0
    ] = 1.0


    X_train_scaled = (
        X_train_raw - train_mean
    ) / train_std


    X_val_scaled = (
        X_val_raw - train_mean
    ) / train_std


    # ========================================================
    # CONVERT TO 3x3 CNN INPUT
    # ========================================================

    X_train_cnn = (
        X_train_scaled
        .reshape(
            -1,
            3,
            3,
            1
        )
    )


    X_val_cnn = (
        X_val_scaled
        .reshape(
            -1,
            3,
            3,
            1
        )
    )


    # ========================================================
    # BUILD CNN
    # ========================================================

    tf.keras.backend.clear_session()

    model = build_small_cnn()


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "Training CNN..."
    )

    history = model.fit(
        X_train_cnn,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=True
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    y_prob = (
        model.predict(
            X_val_cnn,
            verbose=0
        )
        .ravel()
    )


    y_pred = (
        y_prob >= 0.5
    ).astype(
        int
    )


    (
        accuracy,
        sensitivity,
        specificity,
        auc,
        tn,
        fp,
        fn,
        tp
    ) = calculate_metrics(
        y_val,
        y_pred,
        y_prob
    )


    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Sensitivity: {sensitivity:.4f}"
    )

    print(
        f"Specificity: {specificity:.4f}"
    )

    if not np.isnan(auc):

        print(
            f"AUC: {auc:.4f}"
        )

    else:

        print(
            "AUC: N/A"
        )


    fold_results.append({

        "Fold":
            int(fold),

        "Training_Patients":
            len(train_df),

        "Validation_Patients":
            len(validation_df),

        "Accuracy":
            accuracy,

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "AUC":
            auc,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp
    })


    # ========================================================
    # PATIENT-LEVEL PREDICTIONS
    # ========================================================

    for i in range(
        len(validation_df)
    ):

        all_predictions.append({

            "Patient_ID":
                validation_df.iloc[i][
                    ID_COLUMN
                ],

            "Fold":
                int(fold),

            "True_Label":
                int(y_val[i]),

            "Predicted_Label":
                int(y_pred[i]),

            "Predicted_Probability":
                float(y_prob[i])
        })


# ============================================================
# STEP 14 - OVERALL PERFORMANCE
# ============================================================

print()
print("=" * 75)
print("STEP 14 - OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 75)


predictions_df = pd.DataFrame(
    all_predictions
)


y_true_all = predictions_df[
    "True_Label"
].values


y_pred_all = predictions_df[
    "Predicted_Label"
].values


y_prob_all = predictions_df[
    "Predicted_Probability"
].values


(
    accuracy,
    sensitivity,
    specificity,
    auc,
    tn,
    fp,
    fn,
    tp
) = calculate_metrics(
    y_true_all,
    y_pred_all,
    y_prob_all
)


print(
    f"Patients evaluated: "
    f"{len(predictions_df)}"
)

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Sensitivity: "
    f"{sensitivity:.4f}"
)

print(
    f"Specificity: "
    f"{specificity:.4f}"
)

print(
    f"AUC: "
    f"{auc:.4f}"
)


# ============================================================
# STEP 15 - ROC DATA
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true_all,
    y_prob_all
)


roc_df = pd.DataFrame({

    "False_Positive_Rate":
        fpr,

    "True_Positive_Rate":
        tpr,

    "Threshold":
        thresholds
})


# ============================================================
# STEP 16 - SAVE PATIENT PREDICTIONS
# ============================================================

print()
print("STEP 16 - SAVING PATIENT PREDICTIONS")
print("=" * 75)


predictions_file = os.path.join(
    OUTPUT_DIR,
    "STEP_28_Patient_Predictions.csv"
)


predictions_df.to_csv(
    predictions_file,
    index=False
)


print(
    f"Saved: {predictions_file}"
)


# ============================================================
# STEP 17 - SAVE FOLD RESULTS
# ============================================================

fold_results_file = os.path.join(
    OUTPUT_DIR,
    "STEP_28_Fold_Results.csv"
)


pd.DataFrame(
    fold_results
).to_csv(
    fold_results_file,
    index=False
)


print(
    f"Saved: {fold_results_file}"
)


# ============================================================
# STEP 18 - SAVE ROC DATA
# ============================================================

roc_file = os.path.join(
    OUTPUT_DIR,
    "STEP_28_ROC_Data.csv"
)


roc_df.to_csv(
    roc_file,
    index=False
)


print(
    f"Saved: {roc_file}"
)


# ============================================================
# STEP 19 - SAVE OVERALL RESULTS
# ============================================================

overall_file = os.path.join(
    OUTPUT_DIR,
    "STEP_28_Overall_Results.csv"
)


overall_results = pd.DataFrame([{

    "Classifier":
        "Small CNN",

    "Patients_Evaluated":
        len(predictions_df),

    "Accuracy":
        accuracy,

    "Sensitivity":
        sensitivity,

    "Specificity":
        specificity,

    "AUC":
        auc,

    "Number_of_Folds":
        5,

    "Random_Seed":
        RANDOM_STATE,

    "Epochs":
        EPOCHS,

    "Batch_Size":
        BATCH_SIZE,

    "Patch_Size":
        PATCH_SIZE,

    "Representation":
        "3x3 radiomic feature matrix",

    "Normalization":
        "Training-fold mean/std only",

    "Data_Leakage_Prevention":
        "Yes"
}])


overall_results.to_csv(
    overall_file,
    index=False
)


print(
    f"Saved: {overall_file}"
)


# ============================================================
# STEP 20 - SAVE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "STEP_28_Small_CNN_Report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PROJECT 7 - RADIOMICS\n"
    )

    f.write(
        "STEP 28 - SMALL CNN TUMOR CLASSIFIER\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "METHOD:\n"
    )

    f.write(
        "Small Convolutional Neural Network\n"
    )

    f.write(
        "Binary endpoint: Two-Year Survival\n\n"
    )

    f.write(
        "CROSS-VALIDATION:\n"
    )

    f.write(
        "Stratified patient-level 5-fold CV\n"
    )

    f.write(
        "Fold assignments imported from STEP 22\n"
    )

    f.write(
        "Random seed: 42\n\n"
    )

    f.write(
        "INPUT REPRESENTATION:\n"
    )

    f.write(
        "Nine radiomic features arranged as a deterministic 3x3 matrix.\n"
    )

    f.write(
        "The CNN operates on this compact feature representation.\n\n"
    )

    f.write(
        "NORMALIZATION:\n"
    )

    f.write(
        "Training-fold mean/std only.\n\n"
    )

    f.write(
        "CNN PARAMETERS:\n"
    )

    f.write(
        f"Epochs: {EPOCHS}\n"
    )

    f.write(
        f"Batch size: {BATCH_SIZE}\n"
    )

    f.write(
        f"Learning rate: {LEARNING_RATE}\n"
    )

    f.write(
        "Architecture: Conv2D -> MaxPooling -> Conv2D -> Dense -> Dropout -> Output\n\n"
    )

    f.write(
        "OVERALL RESULTS:\n"
    )

    f.write(
        f"Patients evaluated: {len(predictions_df)}\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Sensitivity: {sensitivity:.6f}\n"
    )

    f.write(
        f"Specificity: {specificity:.6f}\n"
    )

    f.write(
        f"AUC: {auc:.6f}\n\n"
    )

    f.write(
        "CONFUSION MATRIX:\n"
    )

    f.write(
        f"TN: {tn}\n"
    )

    f.write(
        f"FP: {fp}\n"
    )

    f.write(
        f"FN: {fn}\n"
    )

    f.write(
        f"TP: {tp}\n\n"
    )

    f.write(
        "METHODOLOGICAL NOTE:\n"
    )

    f.write(
        "The validation patients were never used during CNN training.\n"
    )

    f.write(
        "Normalization parameters were estimated independently "
        "from the training patients in each fold.\n"
    )


print(
    f"Saved: {report_file}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("STEP 28 COMPLETE")
print("=" * 75)

print(
    "\nSmall CNN classifier completed successfully."
)

print(
    f"Patients evaluated: "
    f"{len(predictions_df)}"
)

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Sensitivity: "
    f"{sensitivity:.4f}"
)

print(
    f"Specificity: "
    f"{specificity:.4f}"
)

print(
    f"AUC: "
    f"{auc:.4f}"
)

print()
print("OUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print()
print("OUTPUT FILES:")

print(predictions_file)
print(fold_results_file)
print(roc_file)
print(overall_file)
print(report_file)

print()
print("=" * 75)
print("READY FOR STEP 29")
print("=" * 75)
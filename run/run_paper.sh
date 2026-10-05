#!/usr/bin/env bash

source .venv/bin/activate

# Add all datasets you want to train on here
DS_VALUES=(
    "full"
    "corridor"
    "open"
)

# Config suffixes shared by every dataset
CONFIG_SUFFIXES=(
    "AE"
    "Constrained_AE"
    "FP"
    "PSSN"
    "Siamese_G"
    "Siamese"
    "Triplet"
    "UAGeo"
)

CONFIG_SUFFIXES_DIM_RED=(
    "Isomap"
    "PCA"
    "t-SNE"
    "UMAP"
)

RUNS=10

echo "Starting training runs with different datasets and configs..."
echo "=============================================================="

for DS in "${DS_VALUES[@]}"; do
    echo ""
    echo "Processing dataset: $DS"
    echo "=============================================================="

    for suffix in "${CONFIG_SUFFIXES[@]}"; do
        config="${DS}/${suffix}"

        echo ""
        echo "Running with config: $config..."
        echo "-----------------------------------"

        for ((i=1; i<=RUNS; i++)); do
            if ! python run/train.py --config-name "$config"; then
                echo "ERROR: Training with config=$config failed!"
                exit 1
            fi
        done

        echo "Completed run with config=$config"
    done

    for suffix in "${CONFIG_SUFFIXES_DIM_RED[@]}"; do
        config="${DS}/${suffix}"

        echo ""
        echo "Running with config: $config..."
        echo "-----------------------------------"

        for ((i=1; i<=RUNS; i++)); do
            if ! python run/dimensionality_reduction.py --config-name "$config"; then
                echo "ERROR: Training with config=$config failed!"
                exit 1
            fi
        done

        echo "Completed run with config=$config"
    done
done

echo ""
echo "=============================================================="
echo "All training runs completed successfully!"
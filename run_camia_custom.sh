#!/bin/bash
# Simplified CAMIA workflow - Skip baselines, use custom dataset

# CAMIA-only workflow for custom dataset
BASE_MODEL="common-pile/comma-v0.1-2t"
EXP_NAME="comma_custom_books"

echo "====== CAMIA Custom Dataset Workflow ======"
echo "Model: $BASE_MODEL"
echo "Experiment: $EXP_NAME"
echo ""

# Step 1: Construct MIA training data
echo "[1/3] Constructing MIA training data..."
CUDA_VISIBLE_DEVICES=0 python run_ours_construct_mia_data_custom.py \
    base_model=$BASE_MODEL \
    experiment_name=$EXP_NAME

# Step 2: Train logistic regression model
echo ""
echo "[2/3] Training logistic regression classifier..."
CUDA_VISIBLE_DEVICES=0 python run_ours_train_lr.py \
    base_model=$BASE_MODEL \
    experiment_name=$EXP_NAME

# Step 3: Generate ROC curves
echo ""
echo "[3/3] Generating ROC curves..."
CUDA_VISIBLE_DEVICES=0 python run_ours_get_roc.py \
    base_model=$BASE_MODEL \
    experiment_name=$EXP_NAME

echo ""
echo "====== CAMIA Workflow Complete ======"
echo "Results saved to: results_new/"

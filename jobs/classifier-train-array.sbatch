#!/usr/bin/env bash

#SBATCH --job-name=us8k-classifier-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.org
#SBATCH --output="classifier-train-%A-%a.out"
#SBATCH --err="classifier-train-%A-%a.err"


source ~/.bashrc
source activate l3embedding-cpu

SRCDIR=$HOME/dev/l3embedding
FEATURES_DIR=/scratch/jtc440/sonyc-usc/features/esc50/l3/original/music/cnn_L3_orig
OUTPUT_DIR=/scratch/jtc440/sonyc-usc
MODEL_TYPE='svm'
FEATURE_MODE='framewise'
GOOGLE_DEV_APP_NAME=''
GSHEET_ID=''
FOLD_NUM=$SLURM_ARRAY_TASK_ID

module purge

python $SRCDIR/06_train_classifier.py \
    --random-state 20171021 \
    --model-type $MODEL_TYPE \
    --feature-mode $FEATURE_MODE \
    --num-epochs 50 \
    --train-batch-size 32 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --parameter-search \
    --parameter-search-no-valid-fold \
    --parameter-search-valid-ratio 0.15 \
    --verbose \
    $FEATURES_DIR \
    $OUTPUT_DIR \
    $FOLD_NUM

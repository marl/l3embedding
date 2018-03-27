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
FEATURES_DIR=/scratch/jtc440/l3_features/us8k/l3_frames_uniform/l3embedding/cnn_L3_orig/original
OUTPUT_DIR=/scratch/jtc440/cls_output
MODEL_ID='mycls'
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
    --num-epochs 10 \
    --train-batch-size 128 \
    --train-epoch-size 16 \
    --norm-penalty-factor 1.0 \
    --svm-conv-tolerance 0.001 \
    --svm-max-iterations 1000000 \
    --svm-kernel-type rbf \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --verbose \
    $FEATURES_DIR \
    $OUTPUT_DIR \
    $MODEL_ID \
    $FOLD_NUM

#    --learning-rate 0.00001 \
#    --weight-decay 0.00001 \

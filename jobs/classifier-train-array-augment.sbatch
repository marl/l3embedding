#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=us8k-classifier-train-augment
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.here
#SBATCH --output="classifier-train-augment-%A-%a.out"
#SBATCH --err="classifier-train-augment-%A-%a.err"

source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding-new

SRCDIR=$HOME/dev/l3embedding
FEATURES_DIR=/scratch/jtc440/sonyc-usc-augmented/features/us8k/l3/original/music/cnn_L3_melspec2
OUTPUT_DIR=/scratch/jtc440/sonyc-usc-augmented
MODEL_TYPE='mlp'
FEATURE_MODE='framewise'
GOOGLE_DEV_APP_NAME='l3embeddingexperiments'
GSHEET_ID=''
FOLD_NUM=$SLURM_ARRAY_TASK_ID

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/06_train_classifier.py \
    --random-state 20171021 \
    --model-type $MODEL_TYPE \
    --feature-mode $FEATURE_MODE \
    --num-epochs 1050 \
    --train-batch-size 32 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --parameter-search \
    --parameter-search-train-without-valid \
    --parameter-search-no-valid-fold \
    --parameter-search-valid-ratio 0.15 \
    --augmented-train-data-dir /scratch/jtc440/sonyc-usc-augmented/batches/train \
    --augmented-standardizer-dir /scratch/jtc440/sonyc-usc-augmented/batches/standardizers \
    --nonaugmented-validation-data-dir /scratch/jtc440/sonyc-usc-augmented/batches/valid \
    --svm-kernel-type linear \
    --verbose \
    $FEATURES_DIR \
    $OUTPUT_DIR \
    $FOLD_NUM

#    --foo \

chgrp -R sonyc-audioset $OUTPUT_DIR
chmod -R g+rx $OUTPUT_DIR

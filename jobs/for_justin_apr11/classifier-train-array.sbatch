#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=us8k-classifier-train-us8k-music
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="classifier-train-us8k-music-%A-%a.out"
#SBATCH --err="classifier-train-us8k-music-%A-%a.err"

source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding

SRCDIR=$HOME/dev/l3embedding
FEATURES_DIR=/scratch/js7561/sonyc-usc-kaprefix/features/us8k/l3/original/music/cnn_L3_kapredbinputbn
OUTPUT_DIR=/scratch/js7561/sonyc-usc-kaprefix
MODEL_TYPE='mlp'
FEATURE_MODE='framewise'
GOOGLE_DEV_APP_NAME='l3embeddingexperiments'
GSHEET_ID='1eyFv_jUWJeEuG1VKQNfAmhQUREMRDgEMxfVDffRaHDo'
FOLD_NUM=$SLURM_ARRAY_TASK_ID

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/06_train_classifier.py \
    --random-state 20171021 \
    --model-type $MODEL_TYPE \
    --feature-mode $FEATURE_MODE \
    --num-epochs 50 \
    --train-batch-size 32 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --parameter-search \
    --parameter-search-train-without-valid \
    --parameter-search-no-valid-fold \
    --parameter-search-valid-ratio 0.15 \
    --svm-kernel-type linear \
    --verbose \
    $FEATURES_DIR \
    $OUTPUT_DIR \
    $FOLD_NUM

chgrp -R sonyc-audioset $OUTPUT_DIR
chmod -R g+rx $OUTPUT_DIR

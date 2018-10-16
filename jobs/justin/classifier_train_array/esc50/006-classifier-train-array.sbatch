#!/usr/bin/env bash

#SBATCH --job-name=006-e50-classifier-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="006-classifier-train-%A-%a.out"
#SBATCH --err="006-classifier-train-%A-%a.err"


source ~/.bashrc
source activate l3cpu

SRCDIR=$HOME/dev/l3embedding
FEATURES_DIR=/scratch/js7561/l3_features/features/esc50/l3/short/music/cnn_L3_orig
OUTPUT_DIR=/scratch/js7561/l3_cls
MODEL_TYPE='mlp'
FEATURE_MODE='framewise'	
GOOGLE_DEV_APP_NAME='l3embeddingexperiments'
GSHEET_ID='1eyFv_jUWJeEuG1VKQNfAmhQUREMRDgEMxfVDffRaHDo'
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
    --parameter-search-train-without-valid \
    $FEATURES_DIR \
    $OUTPUT_DIR \
    $FOLD_NUM

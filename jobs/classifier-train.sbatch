#!/usr/bin/env bash

##SBATCH --gres=gpu:1
#SBATCH --job-name=us8k-classifier-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.org
#SBATCH --output="classifier-train-%j.out"
#SBATCH --err="classifier-train-%j.err"


source ~/.bashrc
source activate l3embedding-cpu

SRCDIR=''
US8K_PATH=''
METADATA_PATH=''
DATA_DIR=''
OUTPUT_DIR=''
MODEL_PATH=''
MODEL_ID='mycls'
DATASET_NAME='us8k'

module purge
#module load cuda/8.0.44
#module load cudnn/8.0v6.0

python $SRCDIR/05_train_classifier.py \
    --random-state 20171021 \
    --features l3_frames_uniform \
    --label-format int \
    --model-type svm \
    --l3embedding-model-path $MODEL_PATH \
    --l3embedding-model-type cnn_L3_orig \
    --hop-size 0.25 \
    --verbose \
    $METADATA_PATH \
    $DATASET_NAME \
    $DATA_DIR \
    $MODEL_ID \
    $OUTPUT_DIR

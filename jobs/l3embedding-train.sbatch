#!/usr/bin/env bash

##SBATCH --gres=gpu:1
#SBATCH --job-name=l3embedding-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.org
#SBATCH --output="l3embedding-train-%j.out"
#SBATCH --err="l3embedding-train-%j.err"


source ~/.bashrc
source activate l3embedding

SRCDIR=''
TRAIN_DATA_DIR=''
VAL_DATA_DIR=''
OUTPUT_DIR=''

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/train.py \
    --num-epochs 150 \
    --train-epoch-size 128 \
    --train-batch-size 16 \
    --train-num-streamers 32 \
    --train-num-distractors 1 \
    --train-mux-rate 256 \
    --train-filter-path $SRCDIR/audioset_filter.csv \
    --train-metadata-path "$TRAIN_DATA_DIR/../csv/*.csv" \
    --ontology-path $SRCDIR/resources/ontology.json \
    --model-type tiny_L3 \
    --validation-epoch-size 128 \
    --validation-batch-size 16 \
    --validation-num-streamers 32 \
    --validation-num-distractors 1 \
    --validation-mux-rate 256 \
    --train-max-videos 100 \
    --validation-max-videos 100 \
    --checkpoint-interval 10 \
    --gpus 1 \
    --learning-rate 0.001 \
    --random-state 20171021 \
    --augment \
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR

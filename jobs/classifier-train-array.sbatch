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
FOLD_NUM=$SLURM_ARRAY_TASK_ID

module purge

python $SRCDIR/06_train_classifier.py \
    --random-state 20171021 \
    --model-type $MODEL_TYPE \
    --num-epochs 10 \
    --train-num-streamers 1024 \
    --train-mux-rate 16 \
    --train-batch-size 128 \
    --train-epoch-size 16 \
    --valid-num-streamers 1024 \
    --valid-mux-rate 1 \
    --valid-batch-size 512 \
    --norm-penalty-factor 0.0001 \
    --svm-conv-tolerance 0.001 \
    --svm-max-iterations 1000000 \
    --svm-reg-penalty-type l2 \
    --verbose \
    $FEATURES_DIR \
    $OUTPUT_DIR \
    $MODEL_ID \
    $FOLD_NUM

#    --learning-rate 0.00001 \
#    --weight-decay 0.00001 \

#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3embedding-train-filtered
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.com
#SBATCH --output="l3embedding-train-filtered-test-%j.out"
#SBATCH --err="l3embedding-train-filtered-test-%j.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding

SRCDIR=$HOME/dev/l3embedding
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples/filtered_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples/filtered_valid
OUTPUT_DIR=/scratch/myname/l3_output

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/03_train_embedding.py \
    --num-epochs 150 \
    --train-epoch-size 4096 \
    --train-batch-size 72 \
    --model-type cnn_L3_orig \
    --validation-epoch-size 4096 \
    --validation-batch-size 16 \
    --checkpoint-interval 10 \
    --gpus 1 \
    --learning-rate 0.0001 \
    --random-state 20171021 \
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR

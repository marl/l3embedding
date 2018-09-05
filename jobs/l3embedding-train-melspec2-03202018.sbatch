#!/usr/bin/env bash

#SBATCH --gres=gpu:4
#SBATCH --job-name=l3embedding-train-melspec2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=62GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@mail.com
#SBATCH --output="l3embedding-train-melspec2-test-%j.out"
#SBATCH --err="l3embedding-train-melspec2-test-%j.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding

SRCDIR=$HOME/dev/l3embedding
TRAIN_DATA_DIR=/beegfs/work/AudioSetSamples_environmental/urban_train
VAL_DATA_DIR=/beegfs/work/AudioSetSamples_environmental/urban_valid
OUTPUT_DIR=/scratch/jtc440/l3_output
GOOGLE_DEV_APP_NAME='l3embeddingexperiments'
GSHEET_ID='' # REPLACE THIS
NUM_GPUS=4

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0

python $SRCDIR/03_train_embedding.py \
    --num-epochs 300 \
    --train-epoch-size 4096 \
    --train-batch-size 64 \
    --model-type cnn_L3_melspec2 \
    --validation-epoch-size 1024 \
    --validation-batch-size 64 \
    --checkpoint-interval 10 \
    --gpus $NUM_GPUS \
    --learning-rate 0.00001 \
    --random-state 20180216 \
    --gsheet-id $GSHEET_ID \
    --google-dev-app-name $GOOGLE_DEV_APP_NAME \
    --verbose \
    $TRAIN_DATA_DIR \
    $VAL_DATA_DIR \
    $OUTPUT_DIR

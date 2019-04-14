#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=generate-l3embedding-samples
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="generate-l3embedding-samples-%A-%a.out"
#SBATCH --err="generate-l3embedding-samples-%A-%a.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding

SRCDIR=$HOME/dev/l3embedding
L3_MODEL_PATH='/scratch/jtc440/l3_output/converted_models/embedding/music/cnn_L3_melspec2/20190222215808/model_best_valid_accuracy.h5'
L3_POOLING_TYPE='original'
US8K_PATH=/beegfs/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=/scratch/js7561/sonyc-usc-kaprefix
FOLD=$SLURM_ARRAY_TASK_ID
DATASET='us8k'

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0
module load ffmpeg/intel/3.2.2

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --us8k-metadata-path $METADATA_PATH \
    --hop-size 0.1 \
    --gpus 1 \
    --fold $FOLD \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR

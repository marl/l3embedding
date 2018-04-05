#!/usr/bin/env bash

#SBATCH --job-name=generate-l3embedding-samples
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.com
#SBATCH --output="generate-l3embedding-samples-%A-%a.out"
#SBATCH --err="generate-l3embedding-samples-%A-%a.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding

SRCDIR=$HOME/dev/l3embedding
L3_MODEL_PATH=''
L3_POOLING_TYPE='original'
US8K_PATH=/scratch/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=/scratch/jtc440/l3_features
MODEL_ID='melspec1'

module purge

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --hop-size 0.1 \
    --gpus 0 \
    --fold 1 \
    $METADATA_PATH \
    $MODEL_ID \
    us8k \
    $DATA_DIR \
    $OUTPUT_DIR

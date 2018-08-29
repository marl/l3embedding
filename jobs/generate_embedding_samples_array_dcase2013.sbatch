#!/usr/bin/env bash

#SBATCH --job-name=generate-l3embedding-samples
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.com
#SBATCH --output="generate-l3embedding-samples-%A-%a.out"
#SBATCH --err="generate-l3embedding-samples-%A-%a.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding-cpu

SRCDIR=$HOME/dev/l3embedding
L3_MODEL_PATH=''
L3_POOLING_TYPE='original'
DATA_DIR=/scratch/hhw230/dcase2013/audio
OUTPUT_DIR=/scratch/jtc440/sonyc-usc
DATASET='dcase2013'

module purge
#module load cuda/8.0.44
#module load cudnn/8.0v6.0
module load ffmpeg/intel/3.2.2

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'l3' \
    --l3embedding-model-path $L3_MODEL_PATH \
    --l3embedding-pooling-type $L3_POOLING_TYPE \
    --hop-size 0.1 \
    --gpus 0 \
    --fold $SLURM_ARRAY_TASK_ID \
    $DATASET \
    $DATA_DIR \
    $OUTPUT_DIR

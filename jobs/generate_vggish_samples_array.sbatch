#!/usr/bin/env bash

#SBATCH --job-name=generate-vggish-samples
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=62GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.com
#SBATCH --output="generate-vggish-samples-%A-%a.out"
#SBATCH --err="generate-vggish-samples-%A-%a.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding-cpu

SRCDIR=$HOME/dev/l3embedding
US8K_PATH=/scratch/jtc440/UrbanSound8K
METADATA_PATH=$US8K_PATH/metadata/UrbanSound8K.csv
DATA_DIR=$US8K_PATH/audio
OUTPUT_DIR=/scratch/jtc440/l3_features
MODEL_ID='vggish'

module purge
#module load cuda/8.0.44
#module load cudnn/8.0v6.0
module load ffmpeg/intel/3.2.2

python $SRCDIR/05_generate_embedding_samples.py \
    --random-state 20180302 \
    --verbose \
    --features 'vggish_frames_uniform' \
    --label-format 'int' \
    --hop-size 0.1 \
    --gpus 0 \
    --fold $SLURM_ARRAY_TASK_ID \
    $METADATA_PATH \
    $MODEL_ID \
    us8k \
    $DATA_DIR \
    $OUTPUT_DIR

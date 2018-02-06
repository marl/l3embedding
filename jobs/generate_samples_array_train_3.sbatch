#!/usr/bin/env bash

#SBATCH --job-name=generate-samples-audioset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.com
#SBATCH --output="generate-samples-audioset-%A-%a.out"
#SBATCH --err="generate-samples-audioset-%A-%a.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding

SRCDIR=$HOME/dev/l3embedding
OUTPUT_DIR=/beegfs/work/AudioSetSamples/filtered_train
SUBSET_PATH=/scratch/jtc440/audioset_subsets/audioset_filtered_train.csv
USER_IDX=0
NUM_WORKERS=4
NUM_TASKS=12
BASE_RANDOM_STATE=20185000

module purge
module load ffmpeg/intel/3.2.2

python $SRCDIR/02_generate_samples.py \
    --batch-size 1024 \
    --num-streamers 20 \
    --mux-rate 20 \
    --augment \
    --precompute \
    --num-workers $NUM_WORKERS \
    --num-distractors 2 \
    --random-state $[$BASE_RANDOM_STATE + $NUM_WORKERS * ($SLURM_ARRAY_TASK_ID - 1  + $NUM_TASKS * $USER_IDX)] \
    --include-metadata \
    $SUBSET_PATH \
    $[5834624 / $NUM_TASKS] \
    $OUTPUT_DIR

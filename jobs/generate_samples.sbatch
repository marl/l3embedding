#!/usr/bin/env bash

#SBATCH --job-name=generate-samples-audioset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.com
#SBATCH --output="generate-samples-audioset-%j.out"
#SBATCH --err="generate-samples-audioset-%j.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding

SRCDIR=''
OUTPUT_DIR=''
SUBSET_PATH=''
BASE_RANDOM_STATE=20180118

module purge

python $SRCDIR/02_generate_samples.py \
    --batch-size 16 \
    --num-streamers 64 \
    --mux-rate 2 \
    --augment \
    --precompute \
    --num-workers 2 \
    --num-distractors 1 \
    --random-state $BASE_RANDOM_STATE \
    --include-metadata \
    $SUBSET_PATH \
    1000 \
    $OUTPUT_DIR

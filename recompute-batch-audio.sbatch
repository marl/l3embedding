#!/usr/bin/env bash

#SBATCH --job-name=recompute-batch-audio
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output="recompute-batch-audio_%j.out"
#SBATCH --err="recompute-batch-audio_%j.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding-new

SRCDIR=$HOME/dev/l3embedding
BATCH_DIR="/beegfs/work/AudioSetSamples/music_test"
SUBSET_PATH="/home/jtc440/dev/audioset_filtered_test.csv"

module purge

python $SRCDIR/recompute_batch_audio.py \
    $BATCH_DIR \
    $SUBSET_PATH \
    --n-jobs 16 \
    --verbose 50
    
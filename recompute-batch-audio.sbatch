#!/usr/bin/env bash

#SBATCH --job-name=recompute-batch-audio
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=62GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output="recompute-batch-audio_%j.out"
#SBATCH --err="recompute-batch-audio_%j.err"


source ~/.bashrc
cd /home/$USER/dev
source activate l3embedding-new

SRCDIR=$HOME/dev/l3embedding
BATCH_DIR="/beegfs/work/AudioSetSamples_environmental/urban_train"
SUBSET_PATH="/home/jtc440/dev/audioset_urban_train.csv"

module purge

python $SRCDIR/recompute_batch_audio.py \
    $BATCH_DIR \
    $SUBSET_PATH \
    --n-jobs 20 \
    --verbose 50
    

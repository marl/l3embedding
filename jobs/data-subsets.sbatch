#!/usr/bin/env bash

#SBATCH --job-name=data-subsets-audioset
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=1-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=name@email.org
#SBATCH --output="data-subsets-audioset-%j.out"
#SBATCH --err="data-subsets-audioset-%j.err"


source ~/.bashrc
source activate l3embedding

SRCDIR=''
DATA_DIR=''
OUTPUT_DIR=''

module purge

python $SRCDIR/01_create_subsets.py \
    --valid-ratio 0.1 \
    --test-ratio 0.1 \
    --metadata-path "$DATA_DIR/../csv/*.csv" \
    --ontology-path $SRCDIR/resources/ontology.json \
    --random-state 20180118 \
    $DATA_DIR \
    $OUTPUT_DIR \
    audioset_filtered

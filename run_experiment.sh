#!/bin/bash
set -e

# Usage: ./run_experiment.sh <experiment_name>

if [ -z "$1" ]; then
    echo "Usage: ./run_experiment.sh <experiment_name>"
    exit 1
fi

EXP_NAME=$1
CKPT_FILE="ckpt_${EXP_NAME}.pt"

echo "Running training for experiment: ${EXP_NAME}"
source ~/speedrun/env/bin/activate

# Add PYTHONUNBUFFERED=1 to ensure we see logs immediately
export PYTHONUNBUFFERED=1

time python train.py --data ../data/train_corpus.txt --steps 2000 --out ${CKPT_FILE}

echo "Training complete. Running evaluate.py..."
python evaluate.py --checkpoint ${CKPT_FILE} --text_file ../data/dev_eval.txt

echo "Done."

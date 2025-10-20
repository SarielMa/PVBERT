#!/bin/bash

# Check if two arguments were provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <topic> <result_path>"
    exit 1
fi

# Get the values from the command-line arguments
TOPIC=$1
RESULT_PATH=$2

# Run each training script one by one
echo "Starting training for one model..."
python fine_tune_all_remove.py -i pv_bert_large -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i pv_bert_base -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i bert-large-uncased -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i bert-base-uncased -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i emilyalsentzer/Bio_ClinicalBERT -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i allenai/scibert_scivocab_uncased -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i dmis-lab/biobert-v1.1 -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i cambridgeltl/SapBERT-from-PubMedBERT-fulltext -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "Starting training for one model..."
python fine_tune_all_remove.py -i Twitter/twhin-bert-base -m "$TOPIC" -p "$RESULT_PATH"_"$TOPIC"

echo "All training jobs completed!"

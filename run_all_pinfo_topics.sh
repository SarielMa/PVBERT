#!/bin/bash

# Activate conda environment (if needed)
# source  ~/.bashrc
conda activate amia2025
module load CUDA/12.6


# Run each training script one by one
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i eppc_bert_large
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i eppc_bert_base
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i bert-large-uncased
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i bert-base-uncased
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i emilyalsentzer/Bio_ClinicalBERT
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i allenai/scibert_scivocab_uncased
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i dmis-lab/biobert-v1.1
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i cambridgeltl/SapBERT-from-PubMedBERT-fulltext
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i Twitter/twhin-bert-base
echo "Starting training for one model..."
python train_all_pinfo_topics.py -i Twitter/twhin-bert-large
echo "All training jobs completed!"

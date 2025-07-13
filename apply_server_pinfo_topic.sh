#!/bin/bash
  

#SBATCH --job-name=bert_message_topic
#SBATCH --time=12:00:00
#SBATCH --mail-user=linhai.ma@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --constraint="a5000"
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=bert.txt

module load CUDA/12.6
module load miniconda
conda activate amia2025
cd /home/lm2445/project/bert_0620/exp_message_refine_0711/
sh run_all_pinfo_topics.sh
python eval_all_topics.py

# sleep 604800

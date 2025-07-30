#!/bin/bash
  

#SBATCH --job-name=bert_message_class
#SBATCH --time=12:00:00
#SBATCH --mail-user=linhai.ma@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --constraint="a5000"
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=bert_pinfo.txt

module load CUDA/12.6
module load miniconda
conda activate amia2025
cd /home/lm2445/project/bert_0620/exp_message_refine_0711/
bash run_all.sh pinfo /home/lm2445/palmer_scratch/results_071325_class
RESULT_PATH="/home/lm2445/palmer_scratch/results_071325_class"
python eval_all.py -m pinfo -i "$RESULT_PATH"_pinfo
# sleep 604800

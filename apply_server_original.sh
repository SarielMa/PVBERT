#!/bin/bash
  

#SBATCH --job-name=Bert_original
#SBATCH --time=6:00:00
#SBATCH --mail-user=linhai.ma@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=Bert_original.txt

module load miniconda
conda activate finben
cd /home/lm2445/project_pi_sjf37/lm2445/Bert_PV_classification_1013_New_Sample1-14/PPCBERT
sh run_original.sh 


# sleep 604800
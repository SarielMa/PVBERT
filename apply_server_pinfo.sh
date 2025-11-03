#!/bin/bash
  

#SBATCH --job-name=bert_pinfo
#SBATCH --time=21:00:00
#SBATCH --mail-user=linhai.ma@yale.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus=rtx_5000_ada:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --output=bert_pinfo.txt

module load miniconda
conda activate finben
cd /home/lm2445/project_pi_sjf37/lm2445/Bert_PV_classification_1013_New_Sample1-14/PPCBERT
sh run_pinfo.sh 


# sleep 604800
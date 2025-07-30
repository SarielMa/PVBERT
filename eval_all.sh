#!/bin/bash

RESULT_PATH="/home/lm2445/palmer_scratch/results_071325_class"

# Run each eval script one by one
echo "Starting testing one method"
python eval_all.py -m topic -i "$RESULT_PATH"_topic
echo "Starting testing one method"
python eval_all.py -m pinfo -i "$RESULT_PATH"_pinfo
echo "Starting testing one method"
python eval_all.py -m original -i "$RESULT_PATH"_original
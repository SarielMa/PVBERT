#!/bin/bash

# RESULT_PATH="/home/lm2445/project_pi_sjf37/lm2445/Bert_PV_classification_1013_New_Sample1-14/samplenew_1_14_BWSD"
# bash fine_tune_all.sh topic $RESULT_PATH
# python eval_all.py -m topic -i "$RESULT_PATH"_topic
# sleep 604800
# bash fine_tune_all.sh pinfo $RESULT_PATH
# python eval_all.py -m pinfo -i "$RESULT_PATH"_pinfo
# sleep 604800
# bash fine_tune_all.sh original $RESULT_PATH
# python eval_all.py -m original -i "$RESULT_PATH"_original
BASE_RESULT_PATH="/home/lm2445/project_pi_sjf37/lm2445/Bert_PV_classification_1013_New_Sample1-14/PV1_14_BWSD_pinfo"
for i in {0..2}; do
    RESULT_PATH="${BASE_RESULT_PATH}/run_${i}"
    echo "Running loop $i with RESULT_PATH=$RESULT_PATH"
    bash fine_tune_all.sh pinfo "$RESULT_PATH"
done

python eval_all.py -m pinfo -i "$BASE_RESULT_PATH"
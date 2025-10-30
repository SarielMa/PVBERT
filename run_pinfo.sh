#!/bin/bash

RESULT_PATH="/home/lm2445/project_pi_sjf37/lm2445/Bert_PV_classification_1013_New_Sample1-14/samplenew_1_14_BWSD"
# bash fine_tune_all.sh topic $RESULT_PATH
# python eval_all.py -m topic -i "$RESULT_PATH"_topic
# sleep 604800
bash fine_tune_all.sh pinfo $RESULT_PATH
python eval_all.py -m pinfo -i "$RESULT_PATH"_pinfo
# sleep 604800
# bash fine_tune_all.sh original $RESULT_PATH
# python eval_all.py -m original -i "$RESULT_PATH"_original
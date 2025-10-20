#!/bin/bash

RESULT_PATH="/home/lm2445/project_pi_sjf37/lm2445/Bert_PV_classification_1013_New_Sample1-14/samplenew_1_14_BWS_remove"
# bash fine_tune_all.sh topic $RESULT_PATH
# python eval_all.py -m topic -i "$RESULT_PATH"_topic
# sleep 604800
# bash fine_tune_all.sh pinfo $RESULT_PATH
# python eval_all.py -m pinfo -i "$RESULT_PATH"_pinfo
# sleep 604800
# maybe can use the method to replace the remove...
bash fine_tune_all_remove.sh original $RESULT_PATH
python eval_all_remove.py -m original -i "$RESULT_PATH"_original
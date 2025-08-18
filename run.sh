#!/bin/bash

RESULT_PATH="YOUR_OUTPUT_PATH"
bash fine_tune_all.sh topic $RESULT_PATH
python eval_all.py -m topic -i "$RESULT_PATH"_topic
# sleep 604800
bash fine_tune_all.sh original $RESULT_PATH
python eval_all.py -m original -i "$RESULT_PATH"_original
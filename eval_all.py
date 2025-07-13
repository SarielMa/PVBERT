import os
import csv
from evaluate_util import eval_for_classification as my_eval

models = [
    "eppc_bert_base",
    "dmis-lab/biobert-v1.1",
    "allenai/scibert_scivocab_uncased",
    "emilyalsentzer/Bio_ClinicalBERT",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "bert-base-uncased",
    "bert-large-uncased",
    "Twitter/twhin-bert-base",
    "eppc_bert_large"]
# models = [
#     "eppc_bert_base"]
prefix = "/home/lm2445/palmer_scratch/results_071125_class/eppc_model_"
suffix = "/checkpoint-2900"
stamp = "pinfo"
csv_res = []
trainset = "stratified_train_data.json"
testset = "stratified_test_data.json"

for i, m in enumerate(models):
    path = prefix + m + suffix
    res = []
    res = my_eval(path, trainset, testset, stamp)
    csv_res.append([m] + [str(round(i, 4))  for i in res])

with open('output_0711.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_res) 

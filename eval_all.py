import os
import csv
from evaluate_util import eval_for_classification as my_eval
import argparse
import numpy as np
import pandas as pd
# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-i", "--Input", help = "the base path to input model to be tested")
#parser.add_argument("-p", "--Path", help = "the output path")
parser.add_argument("-m", "--Method", help = "which method to train: original, pinfo, topic")

# Read arguments from command line
args = parser.parse_args()

models = [
    "dmis-lab/biobert-v1.1",
    "allenai/scibert_scivocab_uncased",
    "emilyalsentzer/Bio_ClinicalBERT",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "bert-base-uncased",
    "bert-large-uncased",
    "Twitter/twhin-bert-base",
    "eppc_bert_base",
    "eppc_bert_large"]


#prefix = "/home/lm2445/palmer_scratch/results_071125_class/eppc_model_"
prefix = args.Input
suffix = "checkpoint-"
stamp = args.Method # e.g. topic
csv_res = []
trainset = f"stratified_train_data_{args.Method}.json"
testset = f"stratified_test_data_{args.Method}.json"

def get_matrix(results, name):
    all_results = pd.concat(results, ignore_index=True)
    metrics = ['Precision', 'Recall', 'F1']
    # Group by ClassLabel
    agg_df = (
        all_results.groupby('ClassLabel')[metrics]
        .agg(['mean', 'std'])
    )
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    agg_df = agg_df.reset_index()
    agg_df.to_csv(name + "mean_std_per_class_table.csv", index=False)
        
for i, m in enumerate(models):
    subpath = "eppc_model_" + m
    path = os.path.join(prefix, subpath)
    print (path)
    folders = [name for name in os.listdir(path)
           if os.path.isdir(os.path.join(path, name)) and name.startswith(suffix)]
    results = []
    codem = []
    subcodem = []
    combom = []
    print ("folders ", folders)
    for i, folder in enumerate(folders):
        new_path = os.path.join(path, folder)
        res = []
        res, m1, m2, m3 = my_eval(new_path, trainset, testset, stamp + str(i))
        results.append(res)
        codem.append(m1)
        subcodem.append(m2)
        combom.append(m3)
    get_matrix(codem, stamp + "code")
    get_matrix(subcodem, stamp + "subcode")
    get_matrix(combom, stamp + "combo")
    #compute the major table
    results = np.array(results)
    means = np.mean(results, axis=0)
    stds  = np.std(results, axis=0)
    row = []
    for mean, std in zip(means, stds):
        row.extend([mean, std])
    csv_res.append([m] + [str(round(i, 4) * 100)  for i in row])

with open(f"{stamp}_output_0714.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_res) 

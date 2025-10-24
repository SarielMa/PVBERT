import torch
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
from utils.my_utils import my_eval_for_classification
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from tqdm import tqdm

def eval_for_classification(model_path, trainset_path, testset_path, stamp):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    with open(trainset_path, "r", encoding="utf-8") as f:
        trainset = json.load(f)
    with open(testset_path, "r", encoding="utf-8") as f:
        testset = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Recover label space from model if possible
    # Rebuild from train+test
    if hasattr(model.config, "label2id") and model.config.label2id:
        label2id = {k: int(v) for k, v in model.config.label2id.items()}
        id2label = {int(v): k for k, v in model.config.label2id.items()}
        all_labels = sorted(label2id.keys())
        print ("Loaded label2id from model config")
    else:
        all_labels = set()
        for dataset in [trainset, testset]:
            for example in dataset:
                for label in example["labels"]:
                    all_labels.add(label)
        all_labels = sorted(all_labels)
        label2id = {label: i for i, label in enumerate(all_labels)}
        id2label = {i: l for l, i in label2id.items()}
    

    # Decompose code/subcode
    code_list = []
    subcode_list = []
    combo_list = []
    for label in all_labels:
        combo_list.append(label)
        if "_" in label:
            code, subcode = label.split("_", 1)
        else:
            code, subcode = label, ""
        code_list.append(code)
        #if subcode != "None":
        if subcode == "Clinical care":
            subcode = "Clinical Care" 
        subcode_list.append(subcode)
    unique_codes = sorted(set(code_list))
    unique_subcodes = sorted(set(subcode_list))
    unique_subcodes.remove("None")
    unique_combos = sorted(set(combo_list))


    def encode(example):
        label_vector = [0] * len(label2id)
        for label in example["labels"]:
            if label in label2id:
                label_vector[label2id[label]] = 1
        tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return tokens, label_vector

    # Helper: extract multi-hot for code/subcode from a label_vector
    def extract_multi_hot(label_vec, which=None):
        indices = [i for i, v in enumerate(label_vec) if v]
        if which == "code":
            codes = set(code_list[i] for i in indices)
            return [int(c in codes) for c in unique_codes]
        elif which == "subcode":
            subcodes = set(subcode_list[i] for i in indices)
            return [int(s in subcodes) for s in unique_subcodes]
        elif which == "combo":
            # combos = set(combo_list[i] for i in indices)
            # return [int(s in combos) for s in unique_combos]
            return list(label_vec) # combo: original multi-hot
        else:
            return None

    # Process test set
    pred_codes, true_codes = [], []
    pred_sub_codes, true_sub_codes = [], []
    pred_combo, true_combo = [], []

    for example in tqdm(testset, desc="Evaluating"):
        inputs, label_vec = encode(example)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        pred_vec = (probs > 0.5).astype(int)

        # Multi-hot over codes
        pred_codes.append(extract_multi_hot(pred_vec, "code"))
        true_codes.append(extract_multi_hot(label_vec, "code"))

        # Multi-hot over subcodes
        pred_sub_codes.append(extract_multi_hot(pred_vec, "subcode"))
        true_sub_codes.append(extract_multi_hot(label_vec, "subcode"))

        # Multi-hot over combos (original labels)
        # pred_combo.append(list(pred_vec))
        # true_combo.append(list(label_vec))
        pred_combo.append(extract_multi_hot(pred_vec, "combo"))
        true_combo.append(extract_multi_hot(label_vec, "combo"))
    # Now pass to your metric function
    return my_eval_for_classification(pred_codes, true_codes, unique_codes, 
                                      pred_sub_codes, true_sub_codes, unique_subcodes,
                                      pred_combo, true_combo, unique_combos, stamp)






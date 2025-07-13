import torch
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
from my_utils import compute_classification_metric, my_eval_for_classification
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
    all_labels = set()
    for dataset in [trainset, testset]:
        for example in dataset:
            for label in example["labels"]:
                all_labels.add(label)
    all_labels = sorted(all_labels)
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    # Build code/subcode label sets
    # all_labels = sorted({l for ds in [trainset, testset] for ex in ds for l in ex["labels"]})
    # label2id = {label: i for i, label in enumerate(all_labels)}
    # id2label = {i: l for l, i in label2id.items()}

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
        subcode_list.append(subcode)
    unique_codes = sorted(set(code_list))
    unique_subcodes = sorted(set(subcode_list))
    unique_subcodes.remove("None")
    unique_combos = sorted(set(combo_list))
    # code2id = {c: i for i, c in enumerate(unique_codes)}
    # subcode2id = {s: i for i, s in enumerate(unique_subcodes)}

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




def eval_for_classification_code_only(model_path, trainset_path, testset_path):
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
    if hasattr(model.config, "label2id") and model.config.label2id:
        label2id = model.config.label2id
    else:
        # Rebuild from train+test
        all_labels = set()
        for dataset in [trainset, testset]:
            for example in dataset:
                for label in example["labels"]:
                    all_labels.add(label)
        all_labels = sorted(all_labels)
        label2id = {label: i for i, label in enumerate(all_labels)}

    id2label = {i: l for l, i in label2id.items()}

    # Encode function
    def encode(example):
        label_vector = [0] * len(label2id)
        #print ("example labels is", example["labels"])
        for label in example["labels"]:
            if label in label2id:
                label_vector[label2id[label]] = 1
        tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return tokens, label_vector

    # Process test set
    all_preds = []
    all_labels = []
    for example in tqdm(testset, desc="Evaluating"):
        inputs, label_vec = encode(example)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        pred = (probs > 0.5).astype(int)
        all_preds.append(pred)
        all_labels.append(label_vec)

    return compute_classification_metric(all_preds, all_labels)

def eval_code_only_ner(model_path, trainset, testset):
    # trainset = "processed_train_PInfo.json"
    # testset = "processed_test_PInfo.json"
    print (" code only eval")
    model_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    
    #  Load Processed Train & Test Data
    with open(trainset, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    with open(testset, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    #  Create Label Mappings from Both Train & Test Data
    all_labels = set()
    for dataset in [train_data, test_data]:
        for entry in dataset:
            for label in entry["labels"]:
                all_labels.add(label.strip().replace(" ", "_"))  # Normalize labels
    
    all_codes = set()
    #all_subcodes = set()
    
    for dataset in [train_data, test_data]:
        for entry in dataset:
            for label in entry["labels"]:
                #label = label.strip().replace(" ", "_")  # Normalize labels
                assert "+" not in label
                if "-" in label:
                    code = label.split("-")[1]
                    all_codes.add(code)
                #all_subcodes.add(subcode)
    
    code_list = sorted(all_codes)
    #subcode_list = sorted(all_subcodes)
    
    # mlb_codes = MultiLabelBinarizer(classes=code_list)
    # mlb_subcodes = MultiLabelBinarizer(classes=subcode_list)
    
    ########################################
    
    label_list = sorted(all_labels)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    #  Prepare True and Predicted Labels for Code, Sub-code, and Span
    true_codes, pred_codes = [], []
    #true_subcodes, pred_subcodes = [], []
    true_spans, pred_spans = [], []
    
    for entry in test_data:
        tokens = entry["tokens"]
        true_labels = entry["labels"]  # List of ground truth labels
    
        #  Tokenize input
        encoding = tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
    
        #  Get Model Predictions
        with torch.no_grad():
            outputs = model(**encoding).logits
    
        predicted_ids = torch.argmax(outputs, dim=2).tolist()[0]
        predicted_labels = [id2label[idx] for idx in predicted_ids]
    
        #  Extract Codes, Subcodes, and Spans
        def extract_components(labels, tokens):
            codes, spans = [], []
            current_span = []
            # current_code, current_subcode = None, None
            current_code = None
            for token, label in zip(tokens, labels):
                if label.startswith("B-"):
                    # If we have a previous entity, save it
                    if current_span:
                        spans.append(" ".join(current_span))
                        codes.append(current_code)
                        #subcodes.append(current_subcode)
    
                    # Extract new entity
                    current_code = label.split("-")[1]
                    # current_code = code_subcode[0]
                    # current_subcode = code_subcode[1] if len(code_subcode) > 1 else "O"
                    current_span = [token]
    
                elif label.startswith("I-") and current_span:
                    current_span.append(token)
    
                else:  # If "O" or a new entity starts
                    if current_span:
                        spans.append(" ".join(current_span))
                        codes.append(current_code)
                        #subcodes.append(current_subcode)
                        current_span = []
    
            if current_span:
                spans.append(" ".join(current_span))
                codes.append(current_code)
                #subcodes.append(current_subcode)
    
            return codes, spans
            
        #  Extract components for ground truth and model predictions
        gt_codes,  gt_spans = extract_components(true_labels, tokens)
        pr_codes,  pr_spans = extract_components(predicted_labels, tokens)
    
        true_codes.append(gt_codes)
        pred_codes.append(pr_codes)
        # true_subcodes.append(gt_subcodes)
        # pred_subcodes.append(pr_subcodes)
        true_spans.append(gt_spans)
        pred_spans.append(pr_spans)
    # print ("true_spans ", true_spans)
    # print ("pred_spans ", pred_spans)
    # print ("subcode sample gt ", true_subcodes)
    # print ("subcode sample pr ", pred_subcodes)
    print ("code list ",code_list)
    print ("code sample gt ", true_codes)
    print ("code sample pr ", pred_codes)
    # print ("subcode list is ", subcode_list)
    return my_eval_code_only(true_codes, pred_codes, true_spans, pred_spans, code_list)


def eval(model_path, trainset, testset):
    # trainset = "processed_train_PInfo.json"
    # testset = "processed_test_PInfo.json"
    model_path = model_path
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    
    #  Load Processed Train & Test Data
    with open(trainset, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    
    with open(testset, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    #  Create Label Mappings from Both Train & Test Data
    all_labels = set()
    for dataset in [train_data, test_data]:
        for entry in dataset:
            for label in entry["labels"]:
                all_labels.add(label.strip().replace(" ", "_"))  # Normalize labels
    
    all_codes = set()
    all_subcodes = set()
    
    for dataset in [train_data, test_data]:
        for entry in dataset:
            for label in entry["labels"]:
                label = label.strip().replace(" ", "_")  # Normalize labels
                if "+" in label:
                    code, subcode = label.split("+", 1)
                    all_codes.add(code[2:])
                    all_subcodes.add(subcode)
                else:
                    #all_codes.add(label)
                    pass
    
    code_list = sorted(all_codes)
    subcode_list = sorted(all_subcodes)
    
    # mlb_codes = MultiLabelBinarizer(classes=code_list)
    # mlb_subcodes = MultiLabelBinarizer(classes=subcode_list)
    
    ########################################
    
    label_list = sorted(all_labels)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    
    #  Prepare True and Predicted Labels for Code, Sub-code, and Span
    true_codes, pred_codes = [], []
    true_subcodes, pred_subcodes = [], []
    true_spans, pred_spans = [], []
    
    for entry in test_data:
        tokens = entry["tokens"]
        true_labels = entry["labels"]  # List of ground truth labels
    
        #  Tokenize input
        encoding = tokenizer(tokens, return_tensors="pt", is_split_into_words=True)
    
        #  Get Model Predictions
        with torch.no_grad():
            outputs = model(**encoding).logits
    
        predicted_ids = torch.argmax(outputs, dim=2).tolist()[0]
        predicted_labels = [id2label[idx] for idx in predicted_ids]
    
        #  Extract Codes, Subcodes, and Spans
        def extract_components(labels, tokens):
            codes, subcodes, spans = [], [], []
            current_span = []
            current_code, current_subcode = None, None
    
            for token, label in zip(tokens, labels):
                if label.startswith("B-"):
                    # If we have a previous entity, save it
                    if current_span:
                        spans.append(" ".join(current_span))
                        codes.append(current_code)
                        subcodes.append(current_subcode)
    
                    # Extract new entity
                    code_subcode = label[2:].split("+", 1)
                    current_code = code_subcode[0]
                    current_subcode = code_subcode[1] if len(code_subcode) > 1 else "O"
                    current_span = [token]
    
                elif label.startswith("I-") and current_span:
                    current_span.append(token)
    
                else:  # If "O" or a new entity starts
                    if current_span:
                        spans.append(" ".join(current_span))
                        codes.append(current_code)
                        subcodes.append(current_subcode)
                        current_span = []
    
            if current_span:
                spans.append(" ".join(current_span))
                codes.append(current_code)
                subcodes.append(current_subcode)
    
            return codes, subcodes, spans
            
        #  Extract components for ground truth and model predictions
        gt_codes, gt_subcodes, gt_spans = extract_components(true_labels, tokens)
        pr_codes, pr_subcodes, pr_spans = extract_components(predicted_labels, tokens)
    
        true_codes.append(gt_codes)
        pred_codes.append(pr_codes)
        true_subcodes.append(gt_subcodes)
        pred_subcodes.append(pr_subcodes)
        true_spans.append(gt_spans)
        pred_spans.append(pr_spans)
    # print ("true_spans ", true_spans)
    # print ("pred_spans ", pred_spans)
    # print ("subcode sample gt ", true_subcodes)
    # print ("subcode sample pr ", pred_subcodes)
    # print ("code list ",code_list)
    # print ("subcode list is ", subcode_list)
    return my_eval(true_codes, pred_codes, true_subcodes, pred_subcodes, true_spans, pred_spans, code_list, subcode_list)
    

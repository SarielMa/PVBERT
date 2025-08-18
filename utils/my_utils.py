from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix 
import pandas as pd
from datetime import datetime

def get_confusion_matrix(y_true, y_pred, label_save, classlabel_list, stamp):
    cm = multilabel_confusion_matrix(y_true, y_pred)
    rows = []
    ret = []
    total_tp = total_fp = total_fn = 0
    for i, cm_i in enumerate(cm):
        tn, fp, fn, tp = cm_i.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "ClassLabel": classlabel_list[i],
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Compute micro-averaged metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Append micro row
    rows.append({
        "ClassLabel": "All",
        "TN": "-",
        "FP": total_fp,
        "FN": total_fn,
        "TP": total_tp,
        "Precision": micro_precision,
        "Recall": micro_recall,
        "F1": micro_f1
    })

    # Save to CSV
    df_metrics = pd.DataFrame(rows)
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #df_metrics.to_csv(stamp + label_save + "_per_label_with_micro_metrics_for_last_model.csv", index=False)
    return df_metrics
    

def my_eval_for_classification(pred_codes, true_codes, codes, pred_sub_codes, true_sub_codes, subcodes, pred_combo, true_combo, combos, stamp):
    res1, m1 = compute_classification_metric(pred_codes, true_codes, "code", codes, stamp) 
    res2, m2 = compute_classification_metric(pred_sub_codes, true_sub_codes, "subcode", subcodes, stamp)
    res3, m3 = compute_classification_metric(pred_combo, true_combo, "combos", combos, stamp)
    return res1 + res2 + res3, m1, m2, m3

def compute_classification_metric(preds, true_labels, what, label_list, stamp):
    y_true = np.array(true_labels)
    y_pred = np.array(preds)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

    print("Micro F1     :", f1)
    print("Precision    :", prec)
    print("Recall       :", recall)
    matrix = get_confusion_matrix(true_labels, preds, what, label_list, stamp)
    return [prec, recall, f1], matrix

def compute_classification_metric_with_matrix(preds, true_labels, id2label=None):
    y_true = np.array(true_labels)
    y_pred = np.array(preds)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

    print("Micro F1     :", f1)
    print("Precision    :", prec)
    print("Recall       :", recall)

    if id2label:
        print("\n Label-wise counts:")
        for i, label in id2label.items():
            tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            print(f"{label:30s} | TP={tp}")

    return [prec, recall, f1]


def calculate_jaccard_for_tokens(phrase1, phrase2):
    """
    Calculate the Jaccard coefficient of two phrases (based on tokens)
    :param phrase1: first phrase string
    :param phrase2: second phrase string
    :return: Jaccard coefficient (0~1)
    """
    set1 = set(phrase1.lower().split())
    set2 = set(phrase2.lower().split())

    # Computing intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate the Jaccard coefficient
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

def is_full_containment_match(phrase1, phrase2):
    """
    Determine if phrase1 is completely contained in phrase2
    :param phrase1: first phrase (str) - true phrase
    :param phrase2: second phrase (str) - predicted phrase
    :return: True if phrase1 is completely contained in phrase2
    """
    set1 = set(phrase1.lower().split())
    set2 = set(phrase2.lower().split())
    
    # Determine whether set1 is a subset of set2
    return set1.issubset(set2)


def relaxed_match_evaluation_with_full_containment(true_entities_list, pred_entities_list, jaccard_threshold=0.5):
    """
    Evaluate partial matches of named entities using Relaxed Match, combining full containment logic and Jaccard coefficient
    :param true_entities_list: list of true entities, one per sentence [[str, ...], ...]
    :param pred_entities_list: list of predicted entities, one per sentence [[str, ...], ...]
    :param jaccard_threshold: Jaccard coefficient threshold for partial matches
    :return: Precision, Recall, F1 Score
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0

    # Iterate over the true and predicted entities in each sentence
    for true_entities, pred_entities in zip(true_entities_list, pred_entities_list):
        matched_true = [False] * len(true_entities)
        matched_pred = [False] * len(pred_entities)

        # For each predicted entity, check whether it partially matches a real entity
        for i, pred_entity in enumerate(pred_entities):
            for j, true_entity in enumerate(true_entities):
                if not matched_true[j] and not matched_pred[i]:
                    # If the real entity is completely contained in the predicted entity, or the predicted entity is completely contained in the real entity, it is considered a match.
                    if is_full_containment_match(true_entity, pred_entity) or is_full_containment_match(pred_entity, true_entity):
                        true_positive += 1
                        matched_true[j] = True
                        matched_pred[i] = True
                    # Otherwise, use the Jaccard coefficient for partial matching evaluation.
                    elif calculate_jaccard_for_tokens(pred_entity, true_entity) >= jaccard_threshold:
                        true_positive += 1
                        matched_true[j] = True
                        matched_pred[i] = True

        # Counting False Positives and False Negatives
        false_positive += matched_pred.count(False)  # No matching predicted entities found
        false_negative += matched_true.count(False)  # No real entity matched

    # Calculate Precision, Recall, F1 Score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
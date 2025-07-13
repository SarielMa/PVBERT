from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix 
import pandas as pd
from datetime import datetime

def get_confusion_matrix(y_true, y_pred, label_save, classlabel_list, stamp):
    cm = multilabel_confusion_matrix(y_true, y_pred)
    rows = []
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
    df_metrics.to_csv(stamp + label_save + "_per_label_with_micro_metrics_for_last_model.csv", index=False)

def my_eval_for_classification(pred_codes, true_codes, codes, pred_sub_codes, true_sub_codes, subcodes, pred_combo, true_combo, combos, stamp):
    return compute_classification_metric(pred_codes, true_codes, "code", codes, stamp) + compute_classification_metric(pred_sub_codes, true_sub_codes, "subcode", subcodes, stamp) + compute_classification_metric(pred_combo, true_combo, "combos", combos, stamp)

def compute_classification_metric(preds, true_labels, what, label_list, stamp):
    y_true = np.array(true_labels)
    y_pred = np.array(preds)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

    print("Micro F1     :", f1)
    print("Precision    :", prec)
    print("Recall       :", recall)
    get_confusion_matrix(true_labels, preds, what, label_list, stamp)
    return [prec, recall, f1]

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

def my_eval_code_only(true_codes, pred_codes, true_spans, pred_spans, code_list):

    code_mlb = MultiLabelBinarizer(classes=code_list)
    #sub_code_mlb = MultiLabelBinarizer(classes=subcode_list)
    
    ## calculate code
    true_code_binary = code_mlb.fit_transform(true_codes)
    pred_code_binary = code_mlb.transform(pred_codes)
    
    precision_code = precision_score(true_code_binary, pred_code_binary, average='micro')
    recall_code = recall_score(true_code_binary, pred_code_binary, average='micro')
    f1_code = f1_score(true_code_binary, pred_code_binary, average='micro')
    print(f"code Precision: {precision_code:.4f}, Recall: {recall_code:.4f}, F1 Score: {f1_code:.4f}")
    # get the span
    jaccard = 0.6
    precision_span, recall_span, f1_span = relaxed_match_evaluation_with_full_containment(true_spans,pred_spans, jaccard_threshold=jaccard)
    print(f"span Precision: {precision_span:.4f}, Recall: {recall_span:.4f}, F1 Score: {f1_span:.4f}, jaccard threshold is {jaccard}")
    return [precision_code, recall_code, f1_code, 
            0, 0, 0, 
            precision_span, recall_span, f1_span]


def my_eval(true_codes, pred_codes, true_sub_codes, pred_sub_codes, true_spans, pred_spans, code_list, subcode_list):
    # results is a list of list of dictionaries
    # data is directly from the json
    # eval the code
    ## code
    # true_codes = []
    # pred_codes = []
    
    # ##  sub-code
    # true_sub_codes = []
    # pred_sub_codes = []
    
    # ##  span
    # true_spans = []
    # pred_spans = []

        
    # for i, line in enumerate(results):
    #     code = [Code_mapping[anno.get("code")].lower() for anno in data[i]["annotations"]]
    #     pred_code = [pred["Code"].lower() for pred in line]
    #     pred_code = list(set(pred_code))
    #     true_codes.append(code)
    #     pred_codes.append(pred_code)
    
    #     sub_code = [Sub_Code_mapping[anno.get("subcode")].lower() for anno in data[i]["annotations"]]
    #     pred_sub_code = [pred["Sub-code"].lower() for pred in line]
    #     pred_sub_code = list(set(pred_sub_code))
    #     true_sub_codes.append(sub_code)
    #     pred_sub_codes.append(pred_sub_code)
    
    #     span = [anno.get("text").lower() for anno in data[i]["annotations"]]
    #     pred_span = [pred["Span"].lower() for pred in line]
    #     true_spans.append(span)
    #     pred_spans.append(pred_span)
        
    #Code_set_eval = [v.lower() for v in Code_mapping.values()]
    #Sub_Code_set_eval = [v.lower() for v in set(list(Sub_Code_mapping.values()))]
    
    code_mlb = MultiLabelBinarizer(classes=code_list)
    sub_code_mlb = MultiLabelBinarizer(classes=subcode_list)
    
    ## calculate code
    true_code_binary = code_mlb.fit_transform(true_codes)
    pred_code_binary = code_mlb.transform(pred_codes)
    
    precision_code = precision_score(true_code_binary, pred_code_binary, average='micro')
    recall_code = recall_score(true_code_binary, pred_code_binary, average='micro')
    f1_code = f1_score(true_code_binary, pred_code_binary, average='micro')
    print(f"code Precision: {precision_code:.4f}, Recall: {recall_code:.4f}, F1 Score: {f1_code:.4f}")
    ## calculate sub-code
    true_subcode_binary = sub_code_mlb.fit_transform(true_sub_codes)
    pred_subcode_binary = sub_code_mlb.transform(pred_sub_codes)
    precision_subcode = precision_score(true_subcode_binary, pred_subcode_binary, average='micro')
    recall_subcode = recall_score(true_subcode_binary, pred_subcode_binary, average='micro')
    f1_subcode = f1_score(true_subcode_binary, pred_subcode_binary, average='micro')
    print(f"subcode Precision: {precision_subcode:.4f}, Recall: {recall_subcode:.4f}, F1 Score: {f1_subcode:.4f}")
    jaccard = 0.6
    precision_span, recall_span, f1_span = relaxed_match_evaluation_with_full_containment(true_spans,pred_spans, jaccard_threshold=jaccard)
    print(f"span Precision: {precision_span:.4f}, Recall: {recall_span:.4f}, F1 Score: {f1_span:.4f}, jaccard threshold is {jaccard}")
    return [precision_code, recall_code, f1_code, 
            precision_subcode, recall_subcode, f1_subcode, 
            precision_span, recall_span, f1_span]
    # results.append([{"code p": precision_code}, 
    #                 {"code r": recall_code},
    #                 {"code f1": f1_code},
    #                 {"subcode p": precision_subcode}, 
    #                 {"subcode r": recall_subcode},
    #                 {"subcode f1": f1_subcode},
    #                 {"span p": precision_span}, 
    #                 {"span r": recall_span},
    #                 {"span f1": f1_span}
    #                ])
    # table = [
    #         ["code p", precision_code], 
    #         ["code r", recall_code],
    #         ["code f1", f1_code],
    #         ["subcode p", precision_subcode], 
    #         ["subcode r", recall_subcode],
    #         ["subcode f1", f1_subcode],
    #         ["span p", precision_span], 
    #         ["span r", recall_span],
    #         ["span f1", f1_span]
    #         ]
    # return results, table

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
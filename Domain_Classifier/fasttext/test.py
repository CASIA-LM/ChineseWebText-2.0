#coding = utf-8
import os
import re
import sys
import json
import jieba
import random
import argparse
import fasttext
from tqdm.auto import tqdm
from multiprocessing import Pool

jieba.setLogLevel(20)


stopwords = set([x.strip() for x in open("../cn_stopwords.txt").readlines()])

if __name__ == '__main__':
    model = fasttext.load_model("../model/model_domain.bin")
    test_file = "./data/test.txt"
    output_data = []
    total_precision_top1 = 0
    total_recall_top1 = 0
    total_precision_multi = 0
    total_recall_multi = 0
    num_samples = 0
    with open(test_file, 'r', encoding='utf-8') as r_f:
        for line in tqdm(r_f):
            num_samples += 1
            ori_text = line.strip()
            ref_labels = re.findall(r'__label__\S+', ori_text)
            text = re.sub(r'__label__\S+\s*', '', ori_text).strip()
            labels_multi, values_multi = model.predict(text, k=-1, threshold=0.3) # k=-1 indicates the return of all labels and their probabilities
            labels_top1, values_top1 = model.predict(text, k=1) #No threshold is set, k=-1 represents the label with the highest probability of return
            
            correct_predictions_top1 = len(set(ref_labels) & set(labels_top1))  
            precision_top1 = correct_predictions_top1 / len(labels_top1) if len(labels_top1) > 0 else 0
            recall_top1 = correct_predictions_top1 / len(ref_labels) if len(ref_labels) > 0 else 0

            correct_predictions_multi = len(set(ref_labels) & set(labels_multi))  
            precision_multi = correct_predictions_multi / len(labels_multi) if len(labels_multi) > 0 else 0
            recall_multi = correct_predictions_multi / len(ref_labels) if len(ref_labels) > 0 else 0
            
            total_precision_top1 += precision_top1
            total_recall_top1 += recall_top1
            total_precision_multi += precision_multi
            total_recall_multi += recall_multi

            result_dict = {
                "text": text,
                "label_top1": labels_top1,
                "value_top1": values_top1.tolist(), 
                "precious_top1": precision_top1,
                "recall_top1": recall_top1,
                "label_multi": labels_multi,
                "value_multi": values_multi.tolist(), 
                "precious_multi": precision_multi,
                "recall_multi": recall_multi,
            }
            output_data.append(result_dict)
    out_file = "./data/result_test.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print(f"Processing complete. Results saved to {out_file}.")

    average_precision_top1 = total_precision_top1 / num_samples
    average_recall_top1 = total_recall_top1 / num_samples
    average_precision_multi = total_precision_multi / num_samples
    average_recall_multi = total_recall_multi / num_samples

    print(f"Average accuracy (precision) of best label: {average_precision_top1:.4f}")
    print(f"Average recall rate of best label: {average_recall_top1:.4f}")
    print(f"Average accuracy (precision)of multi labels: {average_precision_multi:.4f}")
    print(f"Average recall rateof multi labels: {average_recall_multi:.4f}")

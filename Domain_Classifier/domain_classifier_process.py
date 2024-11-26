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

os.environ['JIEBA_CACHE_DIR'] = './tmp'
jieba.initialize()  

stopwords = set([x.strip() for x in open("./cn_stopwords.txt").readlines()])

def build(args):
    text = args
    segs = jieba.lcut(text)
    segs = [x for x in segs if len(x) > 1]
    segs = [x for x in segs if x not in stopwords]
    text = " ".join(segs)
    
    return (args,text)

def main(input_file):
    data = []
    model = fasttext.load_model("./model/model_domain.bin")
    #print("=======================reading file=======================")
    with open(input_file, 'r', encoding='utf-8') as r_f:
        for line in tqdm(r_f):
            #data.append(json.loads(line))
            #data.append(eval(line))
            dict_data = eval(line)
            #text = dict_data["text"]
            try:
                text = dict_data["text"]
            except KeyError:
                try:
                    text = dict_data["content"]
                except KeyError:
                    print("Error: Neither 'text' nor 'content' fields exist in the dictionary.")
                    text = None
            text = text.replace("\n", "").replace("\r","")
            data.append(text)
    data_cnt = 0
    #print("=======================processing data=======================")
    with Pool(48) as p:
        for d in tqdm(p.imap(build, data), total=len(data)):  #IMAP returns results in the same order as the input, while imap_unordered does not guarantee the order.
            #processed_data.append(d)
            text, segs = d
            labels_multi, values_multi = model.predict(segs, k=-1, threshold=0.3) # k=-1 indicates the return of all labels and their probabilities
            labels_top1, values_top1 = model.predict(segs, k=1) #No threshold is set, k=-1 represents the label with the highest probability of return
            result_dict = {
                "text": text,
                "segs": segs,
                "label_top1": labels_top1,
                "value_top1": values_top1.tolist(), 
                "label_multi": labels_multi,
                "value_multi": values_multi.tolist(), 
            }
            print(json.dumps(result_dict, ensure_ascii=False), flush=True)
            data_cnt += 1
    #print("data_cnt=", data_cnt)
    
    return
    

if __name__ == '__main__':
    topics = ['encyclopedia', 'book', 'dialogue', 'education', 'finance', 'law', 'math', 'news', 'medicine', 'technology', 'general']
    file_path = sys.argv[1]
    main(file_path)

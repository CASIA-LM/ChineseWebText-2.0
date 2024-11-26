#coding = utf-8
import os
import sys
import json
import jieba
import random
import argparse
import fasttext
from tqdm.auto import tqdm
from multiprocessing import Pool
import os
os.environ['JIEBA_CACHE_DIR'] = './tmp'
jieba.initialize()  
jieba.setLogLevel(20)

stopwords = set([x.strip() for x in open("../cn_stopwords.txt").readlines()])


def main():
    train_file = "./data/train.txt"
    test_file = "./data/test.txt"
    model = fasttext.train_supervised(input=train_file, lr=0.5, epoch=5, wordNgrams=2, loss='ova')
    model.save_model("../model/model_domain.bin")


if __name__ == '__main__':
    main()
    

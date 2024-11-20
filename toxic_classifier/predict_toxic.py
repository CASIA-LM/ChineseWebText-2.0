#coding = utf-8
import sys
import json
import fasttext
from tqdm.auto import tqdm
from multiprocessing import Pool

stopwords = set([x.strip() for x in open("./data/cn_stopwords.txt").readlines()])
model = fasttext.load_model("./output/toxic_classifier_r2.bin")

def load_data(line):
    dict_data = eval(line)
    text = dict_data["segs"]
    return text

def predict(seg):
    label, value = model.predict(seg, k=2)
    return (label, value)

def main(input_file, save_path):
    data = []

    with open(input_file, 'r', encoding='utf-8') as r_f:
        with Pool(32) as p:
            for idx, text in tqdm(enumerate(p.imap(load_data, r_f))):
                data.append(text)

    with Pool(32) as p:
        with open(save_path, 'w') as f:
            for prediction in tqdm(p.imap(predict, data), total=len(data)): 
                label, value = prediction
                toxic_label = int(label[0][-1])
                value_list = value.tolist()
                score = value_list[1-toxic_label]
                if score < 0.99:
                    toxic_label = 0
                result_dict = {
                    "toxicity": toxic_label,
                    "score": score,
                }
                f.write(json.dumps(result_dict, ensure_ascii=False)+'\n')
    
    return
    

if __name__ == '__main__':
    file_path = sys.argv[1]
    save_path = sys.argv[2]
    main(file_path, save_path)

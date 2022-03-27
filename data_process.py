# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : predict.py
# @Project: Douban_Bert
# @CreateTime : 2022/3/13 上午12:08:22
# @Version：V 0.1
'''
数据预处理
'''
import pandas as pd
import torch
from transformers import Trainer,TrainingArguments, BertTokenizer, BertModel, BertPreTrainedModel,BertConfig
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
import sys
sys.setrecursionlimit(3000)
import re

def tokenize(content):
    filters = ['\t','\n','\x97','\x96','#','$','%','&',':','，','。','\.','“','”','"','《','》'," ","@","、","-","（","）","0","1","2","3","4","5","6","7","8","9"]
    content = re.sub("|".join(filters),"",content)
    return content


def read_data(data_dir):
    data = pd.read_csv(data_dir)
    data['comments'] = data['comments'].fillna('')
    return data

def fill_paddings(data, maxlen):
    '''补全句长'''
    if len(data) < maxlen:
        pad_len = maxlen-len(data)
        paddings = [0 for _ in range(pad_len)]
        data = torch.tensor(data + paddings)
    else:
        data = torch.tensor(data[:maxlen])
    return data

class InputDataSet():

    def __init__(self,data,tokenizer,max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len#最大句长

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data['comments'][item])
        labels = self.data['rating'][item]
        labels = torch.tensor(labels, dtype=torch.long)

        ## 手动构建
        tokens = self.tokenizer.tokenize(text)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids = [101] + tokens_ids + [102]
        input_ids = fill_paddings(tokens_ids,self.max_len)

        attention_mask = [1 for _ in range(len(tokens_ids))]#这里注意传入的是tokens_ids
        attention_mask = fill_paddings(attention_mask,self.max_len)

        token_type_ids = [0 for _ in range(len(tokens_ids))]
        token_type_ids = fill_paddings(token_type_ids,self.max_len)

        return {
            'text':text,
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'labels':labels-1

        }


if __name__ == '__main__':
    train_dir = 'data/train.csv'
    dev_dir = 'data/test.csv'
    model_dir = 'bert-base-chinese'
    train = read_data(train_dir)
    test = read_data(dev_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    train_dataset = InputDataSet(train,tokenizer=tokenizer, max_len=128)
    train_dataloader = DataLoader(train_dataset,batch_size=4)
    batch = next(iter(train_dataloader))

    print(batch)
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['token_type_ids'].shape)
    print(batch['labels'].shape)







# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : predict.py
# @Project: Douban_Bert
# @CreateTime : 2022/3/13 上午12:08:22
# @Version：V 0.1

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
'''
封装模型
'''
def to_input_id(sentence_input):
    tokenizer = BertTokenizer(vocab_file='bert-base-chinese/vocab.txt')
    return tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_input)))
emotion_dict = {"很差":0, "较差":1, "一般":2, "还行":3, "力荐":4}

def getDictKey(myDict, value):
    return [k for k, v in myDict.items() if v == value]

def predict(text):
    # tokenizer = BertTokenizer(vocab_file='bert-base-chinese/vocab.txt')
    config = BertConfig.from_pretrained('cache/config.json')
    model = BertForSequenceClassification.from_pretrained('cache/pytorch_model.bin',
                                                          from_tf=bool('.ckpt' in 'bert-base-chinese'), config=config)
    model.eval()


    sentence = text

    input_id = to_input_id(sentence)
    assert len(input_id) <= 512
    input_ids = torch.LongTensor(input_id).unsqueeze(0)

    # predict时，沒有label所以沒有loss
    outputs = model(input_ids)

    prediction = torch.max(F.softmax(outputs[0], dim=-1), dim=1)[1]  # 返回索引值
    predict_label = prediction.data.cpu().numpy().squeeze()  # 降维

    result = getDictKey(emotion_dict, predict_label)

    return result







if __name__ == "__main__":

    result = predict("燃爆")
    print(result)
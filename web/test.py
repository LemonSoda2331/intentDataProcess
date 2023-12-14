#!/usr/bin/env python
# from sanic.response import json, text
# from feedparser import parse
import json
import preprocess
import utils.train

with open(r'D:\tmp\intents.json', 'r', encoding='utf-8') as f:
    res = json.load(f)
# 预处理传入参数intents.json，需要获取label，train，valid，encode_train，encode_valid
label, train_data, valid_data, encode_train, encode_valid, intent_data = preprocess.main(res)

# 传入上述参数，训练模型
utils.train.main_train(label, train_data, valid_data, encode_train, encode_valid, intent_data)

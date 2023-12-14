#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import re
from sanic import Sanic, response
# from sanic.response import json, text
# from feedparser import parse
import json
from werkzeug.datastructures import FileStorage
import utils
import chardet


app = Sanic(__name__)


# 处理post请求上传的json文件
def get_post(file):
    file = file.stream.body
    tmp = file.decode()
    tmp = re.sub('\r\n {4}', '', tmp)
    tmp = re.sub(' {4}', '', tmp)
    tmp = re.sub('\r\n', '', tmp)
    res = json.loads(tmp)

    data_list = res
    # 删除无效数据，即删除intent和text中有空值的数据
    filtered_array = [
        item for item in data_list
        if item.get('intent') != '' and item.get('text') != ''
    ]

    # 删除其他列，只取intent列和text列
    for item in filtered_array:
        for key in list(item.keys()):
            if key not in ['intent', 'text']:
                del item[key]
        # 删除句尾的标点符号
        item['text'] = re.sub(r'[。？，！；]$', '', item['text']).strip()
        item['intent'] = item['intent'].strip()

    # 根据 item['text']删除重复数据
    filtered_array = list({item['text']: item for item in filtered_array}.values())

    output_file_path = '../data/intents.json'
    json.dump(filtered_array, open(output_file_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    return filtered_array


def data_process(file):
    # 从post请求中接收.csv数据，并处理成需要的shape
    file = file.stream.body
    # result = chardet.detect(file)
    # encoding = result['encoding']
    # decoded_data = file.decode(encoding)
    encodings = ['utf-8', 'gbk', 'ISO-8859-1', 'ANSI']  # 按照可能的编码方式顺序进行尝试
    decoded_data = None
    for encoding in encodings:
        try:
            decoded_data = file.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded_data is None:
        return None

    tmp = decoded_data
    tmp = tmp.replace('\r', '')
    sentences = re.split('\n', tmp)

    data_list = []
    for i in range(len(sentences) - 1):
        if i == 0:
            continue
        intent, context, des, remark = sentences[i].split(',')
        data_list.append({"intent": intent, "text": context, "des": des, "remark": remark})

    # 删除无效数据，即删除intent和text中有空值的数据
    filtered_array = [
        item for item in data_list
        if item.get('intent') != '' and item.get('text') != ''
    ]

    # 删除其他列，只取intent列和text列
    for item in filtered_array:
        for key in list(item.keys()):
            if key not in ['intent', 'text']:
                del item[key]
        # 删除句尾的标点符号
        item['text'] = re.sub(r'[。？，！；]$', '', item['text']).strip()
        item['intent'] = item['intent'].strip()

    # 根据 item['text']删除重复数据
    filtered_array = list({item['text']: item for item in filtered_array}.values())

    output_file_path = '../data/intents.json'
    json.dump(filtered_array, open(output_file_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    return filtered_array

# @app.route('/cluster', methods=['POST'])
# async def cluster(request):
#     a = FileStorage(request.files.get('file'))
#     res = get_post(a)
#     clust = utils.clust(res)
#     return response.json(clust)


@app.route('/evaluate', methods=['POST'])
async def test_json(request):
    a = FileStorage(request.files.get('file'))
    name = a.filename
    if name.__contains__('.csv'):
        res = data_process(a)
    elif name.__contains__('.json'):
        res = get_post(a)
    else:
        return response.json('Wrong filetype')
    model_path = request.get_form('path')
    model_path = model_path['path'][0]
    eva = utils.eva(res, model_path)

    path = '../evaluate/_evaluate.json'
    with open(path, 'r', encoding='utf-8') as f:
        _evaluate = json.loads(f.read())
    clu = utils.clust(_evaluate)

    clust = []
    for i in range(len(clust)):
        clust.append({"text": clust[i]['text'], "intent": clust[i]['intent'], "predict_raw": clust[i]['predict_raw'],
                      "max_logits": clust[i]['max_logits'], "bert_predict_correct": clust[i]['bert_predict_correct'],
                      "cluster_label": clust[i]['cluster_label'], "distance": clust[i]['distance']})

    return response.json(clu)


@app.route('/train', methods=['POST'])
async def preprocess(request):
    # 前端获取post请求，获取上传的intents.json文件
    a = FileStorage(request.files.get('file'))
    b = request.get_form('description')
    df = data_process(a)

    # 预处理传入参数intents.json，生成label，train，valid，encode_train，encode_valid
    utils.process(df)

    # 传入上述参数，训练模型
    res_path = utils.model_train(b['description'][0])

    return response.json(res_path)


@app.route("/getIntents", methods=['POST'])
async def data(request):
    # 从post请求中接收.csv数据，并处理成需要的shape
    a = FileStorage(request.files.get('file'))
    file = a.stream.body
    tmp = file.decode()
    tmp = tmp.replace('\r', '')
    sentences = re.split('\n', tmp)

    data_list = []
    for i in range(len(sentences) - 1):
        if i == 0:
            continue
        intent, context, des, remark = sentences[i].split(',')
        data_list.append({"intent": intent, "text": context, "des": des, "remark": remark})

    # 删除无效数据，即删除intent和text中有空值的数据
    filtered_array = [
        item for item in data_list
        if item.get('intent') != '' and item.get('text') != ''
    ]

    # 删除其他列，只取intent列和text列
    for item in filtered_array:
        for key in list(item.keys()):
            if key not in ['intent', 'text']:
                del item[key]
        # 删除句尾的标点符号
        item['text'] = re.sub(r'[。？，！；]$', '', item['text']).strip()
        item['intent'] = item['intent'].strip()

    # 根据 item['text']删除重复数据
    filtered_array = list({item['text']: item for item in filtered_array}.values())

    # 保存数据
    # output_file_path = os.path.join(os.path.dirname(csv_file_path), 'intents.json')
    output_file_path = '../data/intents.json'
    json.dump(filtered_array, open(output_file_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    return response.json(filtered_array)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

from train import main_train
from mongoengine import *
import json
from utils.preprocess import main_preprocess
from evaluate import main_evaluate
import time
from 聚类分析 import main_Clustering
from rewrite import DataFrame1,call_apipost_api
# 连接MangoDB数据库
connect(host='mongodb://txt:iloveTxt123@dds-uf6e6d8a7e390fe4-pub.mongodb.rds.aliyuncs.com:3717/tx-wisdom-test?authSource=tx-wisdom-test')
class GetIntent(Document):
    meta = {'collection': 'intents','strict': False }
    # You need to specify the fields in your collection here
    _id = ObjectIdField()
    name = StringField()
    code = StringField()
    examples = ListField()
    # add more fields if necessary

#查询所有文档
intents = GetIntent.objects()

# 将查询结果转换为可以写入json的 格式
data = []
for intent in intents:
    for example in intent.examples:
     data.append({
         "intent":intent.code,
         "text":example
    })
#将结果写入json文件
with open('./data/train.json','w',encoding='utf-8')as f:
      json.dump(data,f,ensure_ascii=False,indent = 4)
      print("数据已经保存到train.json")

import pandas as pd
from bson import ObjectId

import json
from mongoengine import *

# def save_json(save_path,da):
#     assert save_path.split('.')[-1] == 'json'
#     with open(save_path,encoding='utf-8') as file:
#         json.dump(da,file)
#
connect(host='mongodb://txt:iloveTxt123@dds-uf6e6d8a7e390fe4-pub.mongodb.rds.aliyuncs.com:3717/tongping-uat?authSource=tongping-uat')

# 定义数据类型
class Conversations(Document):
    meta = {'collection': 'conversations'}
    slots = DictField()  # 字典字段，用于存储插槽数据
    latest_message = DictField()  # 字典字段，用于存储最新的消息
    latest_event_time = FloatField()  # 浮点数字段，用于存储最新事件的时间
    followup_action = DictField()  # 字典字段，用于存储后续动作数据
    paused = BooleanField()  # 布尔字段，用于标识会话是否暂停
    latest_input_channel = StringField()  # 字符串字段，用于存储最新的输入渠道
    active_loop = DictField()  # 字典字段，用于存储活动循环数据
    latest_action = StringField()  # 字符串字段，用于存储最新的动作
    action_name = StringField()  # 字符串字段，用于存储动作名称
    events = ListField()  # 列表字段，用于存储事件列表
    sender_id = StringField()  # 字符串字段，用于存储发送者ID
    latest_action_name = StringField()  # 字符串字段，用于存储最新动作名称


class Mediadeliverjobs(Document):
    meta = {'collection': 'mediadeliverjobs','strict': False }
    _id = ObjectIdField()  # 字符串字段，用于存储ID
    mediaDeliver = ObjectIdField()  # 字符串字段，用于存储媒体交付数据
    deliverType = DictField()  # 字典字段，用于存储交付类型数据
    messageDetail = DictField()  # 字典字段，用于存储消息详情数据
    sendVideoBotMsg = BooleanField()  # 布尔字段，用于标识是否发送视频机器人消息
    updatedAt = DateTimeField()  # 日期时间字段，用于存储更新时间
    createdAt = DateTimeField()  # 日期时间字段，用于存储创建时间
    userInfo = DictField()  # 字典字段，用于存储用户信息数据
    status = StringField()  # 字符串字段，用于存储状态
    setting = DictField()  # 字典字段，用于存储设置数据
    agentInfo = DictField()  # 字典字段，用于存储代理信息数据
    mediaData = DictField()  # 字典字段，用于存储媒体数据
    fiveDetail = DictField()  # 字典字段，用于存储五个详情数据

class Mediadelivers(Document):
    meta = {'collection':'mediadelivers','strict': False}
    _id = ObjectIdField()  # 字符串字段，用于存储ID
    videoBotId = StringField()  # 字符串字段，用于存储视频机器人ID

def is_valid_object_id(id):
    return ObjectId.is_valid(id)

# 自定义JSON编码器
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

def process_page(page,mediadeliverjobs_map, mediadelivers_map):
    data = []
    for doc in page:
        # 从最新消息中获取所需字段
        latest_message = doc.latest_message
        intent = latest_message.get('intent', {}).get('name', "")  # 获取意图名称，默认为空字符串
        confidence = latest_message.get('intent', {}).get('confidence', 0.0)  # 获取意图置信度，默认为0.0
        text = latest_message.get('text', "")  # 获取消息文本，默认为空字符串
        sender_id = doc.sender_id  # 获取发送者ID

        # 通过sender_id和mediadeliverjobs_map进行关联
        mdj = mediadeliverjobs_map.get(sender_id)
        media_Deliver = str(mdj.mediaDeliver) if mdj else None
        # if not mdj:
        #     print(f"No Mediadeliverjobs found for sender_id {sender_id}")

        # 通过media_Deliver和mediadelivers_map进行关联
        md = mediadelivers_map.get(media_Deliver)
        videoBotId = md.videoBotId if md else ""
        # if not md:
        #     print(f"No Mediadelivers found for mediaDeliver {media_Deliver}")
        if not videoBotId:
            continue
        data.append({
            'latest_message': {
                'intent': {
                    'name': intent,
                    'confidence': confidence
                },
                'text': text,
            },
            'sender_id': sender_id,
            'videoBotId': videoBotId
        })

    return data


# 分页加载数据
page_size = 1000  # 每页数据量
total_items = Conversations.objects.count()  # 总数据量
num_pages = (total_items + page_size - 1) // page_size  # 计算总页数
def write():
 with open('merged_data.json', 'w',encoding='utf-8') as f:
    for i in range(num_pages):
        skip = i * page_size  # 计算偏移量
        limit = page_size  # 设置每页数据量
        page = Conversations.objects.skip(skip).limit(limit)  # 查询数据

        sender_ids = [doc.sender_id for doc in page if is_valid_object_id(doc.sender_id)]  # 提取所有的sender_id
        mediadeliverjobs = Mediadeliverjobs.objects(_id__in=sender_ids)  # 根据sender_id查询mediadeliverjobs
        mediadeliverjobs_map = {str(job._id): job for job in mediadeliverjobs}  # 将查询结果转换为字典，以_id作为键

        mediaDelivers = [job.mediaDeliver for job in mediadeliverjobs if job.mediaDeliver and is_valid_object_id(job.mediaDeliver)]
        mediadelivers = Mediadelivers.objects(_id__in=mediaDelivers)  # 根据mediaDeliver查询mediadelivers
        mediadelivers_map = {str(deliver._id): deliver for deliver in mediadelivers}  # 将查询结果转换为字典，以_id作为键
        data1 = process_page(page, mediadeliverjobs_map, mediadelivers_map)  # 处理当前页数据
        for item in enumerate(data1):
            json.dump(item[1], f, cls=CustomJSONEncoder, separators=(',', ':'))  # 使用自定义的JSON编码器进行序列化
            f.write('\n')  # 写入换行符
        print("数据已保存到'merged_data.json'")

def save_to_json(df, save_path):
    data = df.to_dict('records')  # Convert the DataFrame to a list of dictionaries
    with open(save_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False,indent=4)  # Save the list of dictionaries as JSON

def save_to_json_lines(df, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f)
            f.write('\n')

def DataFrame(json_text):
    data_dic={'intent':[], 'text':[],'videoBotId':[]}
    with open(json_text,encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        json_data = json.loads(line)
        data_dic['intent'].append(json_data['latest_message']['intent']['name'])
        data_dic['text'].append(json_data['latest_message']['text'])
        data_dic['videoBotId'].append(json_data['videoBotId'])

    df=pd.DataFrame(data_dic)
    return df


if __name__ == '__main__':
    start_time=time.time()
    main_preprocess()
    main_train()
    write()
    df = DataFrame('merged_data.json')
    df = df[(df['intent'] == 'nlu_fallback') & (df['text'] != '')]
    save_to_json(df, './data/text.json')
    main_evaluate()
    main_Clustering()
    a = DataFrame1('result.json')
    # 调用函数发送请求
    call_apipost_api(a)
    total_time = float((time.time() - start_time))
    print('total time is{}s'.format(time.time()-start_time))


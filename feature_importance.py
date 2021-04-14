# -*- coding: utf-8 -*-

"""
1.中国基本面:
    Hyper parameter:
        num_epochs: 150
        lr_init: 0.001
        random_seed: 273
    Result:
        val accuracy_score: 0.5789473684210527
        val auc: 0.5911910669975187

2.美国基本面:
    Hyper parameter:
        num_epochs: 150
        lr_init: 0.01
        random_seed: 274
    Result:
        val accuracy_score: 0.5108225108225108
        val auc: 0.512374381280936

3.中国基本面 + 美国涨跌:
    Hyper parameter:
        num_epochs: 150
        lr_init: 0.001
        random_seed: 273
    Result:
        val accuracy_score: 0.6140350877192983
        val auc: 0.6265508684863523
"""

import os
import torch
import random

import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

from model import RLSTM, LSTM
from preprocessing import EmotionAnalysis, OriginalData

# -----------------------Training parameters-----------------------

num_epochs = 150
GPU_id = "cuda:0"  # Specifies the graphics card
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
lr_init = 0.01  # Initial learning rate
weights_path = "./pkl/"  # Model save path
weights_save_interval = 5  # Model save interval
random_seed = 274

# -----------------------Pytorch results reproduce-----------------------

# os.environ['PYTHONHASHSEED'] = str(random_seed)
# np.random.seed(random_seed)
# random.seed(random_seed)

torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False

# -----------------------Make train and val data-----------------------

ea = EmotionAnalysis(positive_words_path="data/positive.txt",
                     negative_words_path="data/negative.txt",
                     stop_words_path="data/stopwords.txt")
od = OriginalData(oil_data_path="./data/crudeoil.xlsx",
                  chinese_news_path="./data/china5e_news.csv",
                  out_chinese_news_emotion_path="./data/chinese_news_emotion.csv")

# Calculate emotional characteristics
od.make_news_emotion(ea)

# # 中国基本面
# dataset = od.transform(data_source="zh")
# 美国基本面
# dataset = od.transform(data_source="en")
# 中国基本面 + 美国涨跌
dataset = od.transform_zh_with_en_label()

feature_name_list = ['open', 'high', 'low', 'close', 'volume', 'oi', 'oi_chg', 'ADTM', 'BIAS', 'EXPMA', 'KDJ', 'MA',
                     'MACD']
feature_name_list_zh = ['开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量', '持仓量变化', 'ADTM动态买卖气指标[ADTM指标选项]ADTM[周期数1]23[周期数2]8',
                        'BIAS乖离率[周期数]12', 'EXPMA指数平滑移动平均[周期数]12', 'KDJ随机指标[KDJ指标选项]K[周期数1]9[周期数2]3[周期数3]3',
                        'MA简单移动平均[周期数]5', 'MACD指数平滑移动平均[MACD指标选项]DIFF[周期数]9[短期周期数]12[长期周期数]26']

# Split the data set
train, val = train_test_split(dataset, test_size=0.1, random_state=777)
inputs_train, _, labels_train = list(zip(*train))
inputs_val, _, labels_val = list(zip(*val))


train_x = []
train_y = []
test_x = []
test_y = []

for x_data, y_data in zip(inputs_train, labels_train):
    train_x.append(x_data[-1].tolist()[0])
    train_y.append(y_data.tolist()[0])

for x_data, y_data in zip(inputs_val, labels_val):
    test_x.append(x_data[-1].tolist()[0])
    test_y.append(y_data.tolist()[0])

# ---------------------------------------------------------------------------

dtrain = xgb.DMatrix(np.array(train_x), label=train_y)
deval = xgb.DMatrix(np.array(test_x), label=test_y)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:linear'}

evallist = [(deval, 'eval'), (dtrain, 'train')]

num_round = 100
bst = xgb.train(param, dtrain, num_round, evallist)

result = {}

for feature_index_name, val in bst.get_score(importance_type='gain').items():
    feature_index = int(feature_index_name[1:]) - 1
    result.setdefault(feature_name_list[feature_index], val)


for feature_name in feature_name_list:
    if feature_name not in result:
        result.setdefault(feature_name, 0.0)

print(result)
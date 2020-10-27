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

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

from model import RLSTM, LSTM
from preprocessing import EmotionAnalysis, OriginalData


# -----------------------Training parameters-----------------------

num_epochs = 150
GPU_id = "cuda:0"   # Specifies the graphics card
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
lr_init = 0.01      # Initial learning rate
weights_path = "./pkl/"     # Model save path
weights_save_interval = 5   # Model save interval
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
# # 美国基本面
dataset = od.transform(data_source="en")
# 中国基本面 + 美国涨跌
# dataset = od.transform_zh_with_en_label()

# Split the data set
train, val = train_test_split(dataset, test_size=0.1, random_state=777)
inputs, _, labels = list(zip(*train))

input_size = len(inputs[0][0].tolist()[0])

# -----------------------Model initialization-----------------------

model = LSTM(input_size=input_size, output_size=32, classification=True)
model = model.to(device)

# -----------------------Loss function、optimizer、learning rate decay-----------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# -----------------------Train-----------------------

model.train(mode=True)
print("Training...")

for epoch in range(num_epochs):
    iteration = 0
    total_loss = 0.0
    preds_train = []
    for input, label in zip(inputs, labels):
        iteration += 1
        optimizer.zero_grad()

        label = label.to(device)
        out = None
        hidden = model.init_hidden()
        cell = model.init_cell()
        hidden = hidden.to(device)
        cell = cell.to(device)
        for i in input:
            i = i.to(device)
            out, hidden, cell = model(i, hidden, cell)
        loss = criterion(out, label)
        _, out_binary= torch.max(out, 1)
        preds_train.append(out_binary.cpu().tolist()[0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch: {}\tLoss: {}".format(epoch, total_loss / iteration), end="\t")
    print("train_accuracy_score: {}".format(accuracy_score([x.tolist()[0] for x in labels], preds_train)))
    exp_lr_scheduler.step()

    if epoch % weights_save_interval == 0:
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        torch.save(model.state_dict(), weights_path + "epoch_{}.pkl".format(epoch))


# -----------------------Inference-----------------------

inputs_val, _, labels_val = list(zip(*val))
labels_val = [x.tolist()[0] for x in labels_val]
preds_val = []

model.eval()

for input_val in inputs_val:
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        hidden = model.init_hidden()
        cell = model.init_cell()
        hidden = hidden.to(device)
        cell = cell.to(device)
        out = None
        for i in input_val:
            i = i.to(device)
            out, hidden, cell = model(i, hidden, cell)
        _, out_val = torch.max(out, 1)
        preds_val.append(out_val.cpu().tolist()[0])

print("-------------------------------")
print("val accuracy_score: {}".format(accuracy_score(labels_val, preds_val)))
print("val auc: {}".format(roc_auc_score(labels_val, preds_val)))

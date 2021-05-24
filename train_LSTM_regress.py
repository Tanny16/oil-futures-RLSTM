# -*- coding: utf-8 -*-

"""
1.中国基本面:
    Hyper parameter:
        num_epochs: 100
        lr_init: 0.001
        random_seed: 273
    Result:
        val mse: 0.0006738732385148451
        test mse: 0.0006357924799960034
        val rmse: 0.018639429051468804
        test rmse: 0.018043088067150624
        val mape: nan
        test mape: nan


2.美国基本面:
    Hyper parameter:
        num_epochs: 100
        lr_init: 0.001
        random_seed: 273
    Result:
        val mse: 0.00018516456519314573
        test mse: 0.00019414878695212217
        val rmse: 0.009025249852762594
        test rmse: 0.009121698893903653
        val mape: nan
        test mape: nan

3.中国基本面 + 美国涨跌:
    Hyper parameter:
        num_epochs: 100
        lr_init: 0.001
        random_seed: 273
    Result:
        val mse: 0.0006890569731039551
        test mse: 0.00063317343537859
        val rmse: 0.018782420435197837
        test rmse: 0.017869759831590795
        val mape: nan
        test mape: nan

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

num_epochs = 100
GPU_id = "cuda:2"  # Specifies the graphics card
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
lr_init = 0.001  # Initial learning rate
weights_path = "./pkl/"  # Model save path
weights_save_interval = 5  # Model save interval
random_seed = 273

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
od.train_type = "regress"

# # 中国基本面
# dataset = od.transform(data_source="zh")
# # 美国基本面
# dataset = od.transform(data_source="en")
# 中国基本面 + 美国涨跌
dataset = od.transform_zh_with_en_label()

# Split the data set
train, others = train_test_split(dataset, test_size=0.2, random_state=777)
test, val = train_test_split(dataset, test_size=0.5, random_state=777)
inputs, _, labels = list(zip(*train))

input_size = len(inputs[0][0].tolist()[0])

# -----------------------Model initialization-----------------------

model = LSTM(input_size=input_size, output_size=32, num_classes=1, classification=True)
model = model.to(device)

# -----------------------Loss function、optimizer、learning rate decay-----------------------

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# -----------------------Train-----------------------

model.train(mode=True)
print("Training...")

for epoch in range(num_epochs):
    iteration = 0
    total_loss = 0.0
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
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch: {}\tLoss: {}".format(epoch, total_loss / iteration))
    exp_lr_scheduler.step()

    if epoch % weights_save_interval == 0:
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        torch.save(model.state_dict(), weights_path + "epoch_{}.pkl".format(epoch))

# -----------------------val-----------------------

inputs_val, _, labels_val = list(zip(*val))
preds_val = []

model.eval()

total_loss_val = 0.0
iteration_val = 0
for input_val, label_val in zip(inputs_val, labels_val):
    iteration_val += 1
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        hidden = model.init_hidden()
        cell = model.init_cell()
        hidden = hidden.to(device)
        label_val = label_val.to(device)
        cell = cell.to(device)
        out = None
        for i in input_val:
            i = i.to(device)
            out, hidden, cell = model(i, hidden, cell)
        _, out_val = torch.max(out, 1)
        preds_val.append(out_val.cpu().tolist()[0])
        loss = criterion(out, label_val)
        total_loss_val += loss.item()

print("-------------------------------")
print("val mse: {}".format(total_loss_val / iteration_val))

# -----------------------test-----------------------

inputs_test, _, labels_test = list(zip(*test))
preds_test = []

model.eval()

total_loss_test = 0.0
iteration_test = 0
for input_test, label_test in zip(inputs_test, labels_test):
    iteration_test += 1
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        hidden = model.init_hidden()
        cell = model.init_cell()
        hidden = hidden.to(device)
        label_test = label_test.to(device)
        cell = cell.to(device)
        out = None
        for i in input_test:
            i = i.to(device)
            out, hidden, cell = model(i, hidden, cell)
        _, out_val = torch.max(out, 1)
        preds_val.append(out_val.cpu().tolist()[0])
        loss = criterion(out, label_test)
        total_loss_test += loss.item()


print("-------------------------------")
print("test mse: {}".format(total_loss_test / iteration_test))

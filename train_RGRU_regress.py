# -*- coding: utf-8 -*-

"""
1.中国基本面:
    Hyper parameter:
        num_epochs: 100
        lr_init: 0.001
        random_seed: 273
    Result:
        val mse: 0.052746744677706146
        test mse: 0.0596894634087169

2.美国基本面:
    Hyper parameter:
        num_epochs: 100
        lr_init: 0.001
        random_seed: 273
    Result:
        val mse: 0.015680313917221994
        test mse: 0.017142928605936036

3.中国基本面 + 美国涨跌:
    Hyper parameter:
        num_epochs: 100
        lr_init: 0.001
        random_seed: 273
    Result:
        val mse: 0.0006727270719396752
        test mse: 0.0006183386526099977
"""

import os
import torch
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

from model import RGRU
from preprocessing import EmotionAnalysis, OriginalData
from losses import RMSELoss, MAPELoss, SMAPELoss

# -----------------------Training parameters-----------------------

num_epochs = 100
GPU_id = "cuda:1"   # Specifies the graphics card
device = torch.device(GPU_id if torch.cuda.is_available() else "cpu")
lr_init = 0.001      # Initial learning rate
weights_path = "./pkl/"     # Model save path
weights_save_interval = 5   # Model save interval
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

# 中国基本面
# dataset = od.transform(data_source="zh")
# 美国基本面
# dataset = od.transform(data_source="en")
# 中国基本面 + 美国涨跌
dataset = od.transform_zh_with_en_label()

# Split the data set
train, others = train_test_split(dataset, test_size=0.2, random_state=777)
test, val = train_test_split(dataset, test_size=0.5, random_state=777)

inputs, medias, labels = list(zip(*train))

input_size = len(inputs[0][0].tolist()[0])
media_size = len(medias[0][0].tolist()[0])

# -----------------------Model initialization-----------------------

model = RGRU(input_size=input_size, output_size=32, media_size=media_size, num_classes=1, classification=True)
model = model.to(device)

# -----------------------Loss function、optimizer、learning rate decay-----------------------

criterion = SMAPELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, betas=(0.9, 0.99))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# -----------------------Train-----------------------

model.train(mode=True)
print("Training...")

for epoch in range(num_epochs):
    iteration = 0
    total_loss = 0.0
    preds_train = []
    for input, media, label in zip(inputs, medias, labels):
        iteration += 1
        optimizer.zero_grad()

        label = label.to(device)
        out = None
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        for i, m in zip(input, media):
            i = i.to(device)
            m = m.to(device)
            out, hidden = model(i, m, hidden)
        loss = criterion(out, label)
        _, out_binary = torch.max(out, 1)
        preds_train.append(out_binary.cpu().tolist()[0])
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

inputs_val, medias_val, labels_val = list(zip(*val))
preds_val = []

output_result = []

model.eval()
total_loss_val = 0.0
iteration_val = 0
for input_val, media_val, label_val in zip(inputs_val, medias_val, labels_val):
    iteration_val += 1
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        label_val = label_val.to(device)
        out = None
        for i, m in zip(input_val, media_val):
            i = i.to(device)
            m = m.to(device)
            out, hidden = model(i, m, hidden)
        _, out_val = torch.max(out, 1)
        output_result.append(['val', out.cpu().tolist()[0][0], label_val.cpu().tolist()[0]])
        preds_val.append(out_val.cpu().tolist()[0])
        loss = criterion(out, label_val)
        total_loss_val += loss.item()

print("-------------------------------")
print("val mse: {}".format(total_loss_val / iteration_val))

# -----------------------test-----------------------

inputs_test, medias_test, labels_test = list(zip(*test))
preds_test = []

model.eval()
total_loss_test = 0.0
iteration_test = 0
for input_test, media_test, label_test in zip(inputs_test, medias_test, labels_test):
    iteration_test += 1
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        hidden = model.init_hidden()
        hidden = hidden.to(device)
        label_test = label_test.to(device)
        out = None
        for i, m in zip(input_test, media_test):
            i = i.to(device)
            m = m.to(device)
            out, hidden = model(i, m, hidden)
        _, out_val = torch.max(out, 1)
        output_result.append(['test', out.cpu().tolist()[0][0], label_test.cpu().tolist()[0]])
        preds_val.append(out_val.cpu().tolist()[0])
        loss = criterion(out, label_test)
        total_loss_test += loss.item()

print("-------------------------------")
print("test mse: {}".format(total_loss_test / iteration_test))

df = pd.DataFrame(output_result, columns=['type', 'predict', 'label'])
df.to_csv('./out/R-GRU_result_zh_with_en_label.csv', index=False)

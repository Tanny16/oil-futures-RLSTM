
import os
import re
import math
import torch
import jieba
import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

jieba.setLogLevel(logging.INFO)


class EmotionAnalysis:
    """
    Text emotion analysis

    Attributes:
        pos_words: List of positive words
        neg_words: List of negative words
        stop_words: List of stop words
    """

    def __init__(self,
                 positive_words_path,
                 negative_words_path,
                 stop_words_path):
        self.pos_words = self.read_files_to_words(positive_words_path)
        self.neg_words = self.read_files_to_words(negative_words_path)
        self.stop_words = self.read_files_to_words(stop_words_path)

    def forward(self, text):
        """
        Sentiment analysis calculation
        :param text: Input text
        :return:
            pos_num: Num of positive words
            neg_num: Num of negative words
            POS: Positive emotion score
            NEG: Negative emotion score
            DIV: Emotion divergence factor
        """
        text = self.preprocess(text)
        text = self.remove_html_label(text)
        text = self.remove_blank(text)

        seg_words = [x for x in jieba.cut(text, cut_all=False)]
        seg_words = [x for x in seg_words if x not in self.stop_words]

        total_num = len(seg_words)

        if total_num == 0:
            return 0, 0, 0, 0, 0

        pos_num = 0
        neg_num = 0

        for pos_word in self.pos_words:
            if pos_word not in seg_words:
                continue
            else:
                pos_num += 1
        for neg_word in self.neg_words:
            if neg_word not in seg_words:
                continue
            else:
                neg_num += 1

        NEG = neg_num / total_num
        POS = pos_num / total_num
        DIV = math.log((1 + pos_num) / (1 + neg_num))

        return pos_num, neg_num, POS, NEG, DIV

    @staticmethod
    def read_files_to_words(file_path):
        """
        Used to read word list from file
        """
        words = []
        with open(file_path, "r") as f:
            for line in f.readlines():
                words.append(line.strip())
        return words

    @staticmethod
    def remove_blank(sentence):
        """
        Remove extra characters such as blanks in sentences
        """
        sentence = map(
            lambda s: s.replace(' ', '').replace('　', '').replace('\r', '').replace('\n', '').replace('\t', ''),
            sentence)
        return "".join(list(sentence))

    @staticmethod
    def remove_html_label(sentence):
        """
        Remove html tags in sentences
        """
        pattern = re.compile('<.*?>')
        sentence = re.sub(pattern, ' ', sentence)
        return sentence

    @staticmethod
    def preprocess(text):
        """
        text pre-processing
        """
        text_list = [s.strip() for s in text]
        sentence = "".join(text_list)
        clean = u"\\(.*?\\)|\\{.*?}|\\[.*?]"
        sentence = re.sub(clean, '', sentence)
        return sentence


class OriginalData:

    def __init__(self,
                 oil_data_path,
                 chinese_news_path,
                 out_chinese_news_emotion_path):
        self.oil_data_path = oil_data_path
        self.chinese_news_path = chinese_news_path
        self.out_chinese_news_emotion_path = out_chinese_news_emotion_path
        self.days_iter = 5
        self.train_type = "classification"

    def make_news_emotion(self, emotion_analysis):
        """
        Generate sentiment analysis results
        """
        print("make news emotion...")
        if os.path.exists(self.out_chinese_news_emotion_path):
            return
        news_df = pd.read_csv(self.chinese_news_path)
        res_dict = {"id": [],
                    "content": [],
                    "date": [],
                    "pos_num": [],
                    "neg_num": [],
                    "POS": [],
                    "NEG": [],
                    "DIV": []}
        for index, row in news_df.iterrows():
            print(index)
            the_id, _, content, date, _, _, _, _ = row.tolist()
            pos_num, neg_num, POS, NEG, DIV = emotion_analysis.forward(content)
            res_dict["id"].append(the_id)
            res_dict["content"].append(content)
            res_dict["date"].append(date)
            res_dict["pos_num"].append(pos_num)
            res_dict["neg_num"].append(neg_num)
            res_dict["POS"].append(POS)
            res_dict["NEG"].append(NEG)
            res_dict["DIV"].append(DIV)

        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(self.out_chinese_news_emotion_path)

    def transform(self, data_source="zh"):
        """
        Construct a dataset

        Args:
            data_source: in {"zh", "en"}
        """
        if data_source == "en":
            sheet_id = 2
        else:
            sheet_id = 0
        data_df = pd.read_excel(self.oil_data_path, sheet_name=[0, 1, 2])[sheet_id]
        data_df = data_df.drop([0, 1, 2], axis=0)
        data_df = data_df.dropna(axis=0, how='any')
        emotion_dict = self.load_emotion_data()
        data_dict = {"base": [],
                     "emotion": []}
        for index, row in data_df.iterrows():
            date, *base_data, ratio = row.tolist()
            date = str(date).split(" ")[0]
            data_dict["base"].append(base_data)
            if date in emotion_dict:
                data_dict["emotion"].append(emotion_dict[date])
            else:
                data_dict["emotion"].append([0.0, 0.0, 0.0])

        if self.train_type == "classification":
            return self.min_max_data_dict(data_dict)
        else:
            return self.min_max_data_dict_regress(data_dict)

    def transform_zh_with_en_label(self):
        """
        Construct a zh_with_en_label dataset
        """
        zh_data_df = pd.read_excel(self.oil_data_path, sheet_name=[0, 1, 2])[0]
        en_data_df = pd.read_excel(self.oil_data_path, sheet_name=[0, 1, 2])[2]

        zh_data_df = zh_data_df.drop([0, 1, 2], axis=0)
        zh_data_df = zh_data_df.dropna(axis=0, how='any')
        en_data_df = en_data_df.drop([0, 1, 2], axis=0)
        en_data_df = en_data_df.dropna(axis=0, how='any')

        en_data_list = [[row.tolist()[0], row.tolist()[4]] for index, row in en_data_df.iterrows()]
        en_data_dict = {}

        for i in range(len(en_data_list) - 1):
            if en_data_list[i + 1][1] > en_data_list[i][1]:
                label = 1.0
            else:
                label = 0.0
            en_data_dict.setdefault(en_data_list[i][0], label)

        emotion_dict = self.load_emotion_data()
        zh_data_dict = {"base": [],
                     "emotion": []}
        for index, row in zh_data_df.iterrows():
            date, *base_data, ratio = row.tolist()
            date = str(date).split(" ")[0]

            if date in en_data_dict:
                en_feature = en_data_dict[date]
            else:
                en_feature = -1.0

            zh_data_dict["base"].append([*base_data, en_feature])
            if date in emotion_dict:
                zh_data_dict["emotion"].append(emotion_dict[date])
            else:
                zh_data_dict["emotion"].append([0.0, 0.0, 0.0])
        if self.train_type == "classification":
            return self.min_max_data_dict(zh_data_dict)
        else:
            return self.min_max_data_dict_regress(zh_data_dict)

    def min_max_data_dict(self, data_dict):
        """
        Data normalization
        """
        min_max_scaler = MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(data_dict["base"])
        data_dict.update({"base": X_train_minmax})

        data_list = [[a, b] for a, b in zip(data_dict["base"], data_dict["emotion"])]
        data_list_with_label = []
        for i in range(len(data_list) - (self.days_iter + 1)):
            base_temp = []
            emotion_temp = []
            for j in range(self.days_iter):
                base_temp.append(torch.Tensor([data_list[i + j][0]]))
                emotion_temp.append(torch.Tensor([data_list[i + j][1]]))
            open, high, low, close, *_ = data_list[i + j][0]
            next_open, next_high, next_low, next_close, *_ = data_list[i + j + 1][0]
            if next_close > close:
                label = torch.from_numpy(np.array([1]))
            else:
                label = torch.from_numpy(np.array([0]))
            data_list_with_label.append([base_temp, emotion_temp, label])
        return data_list_with_label

    def min_max_data_dict_regress(self, data_dict):
        """
        Data normalization
        """
        min_max_scaler = MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(data_dict["base"])
        data_dict.update({"base": X_train_minmax})

        data_list = [[a, b] for a, b in zip(data_dict["base"], data_dict["emotion"])]
        data_list_with_label = []
        for i in range(len(data_list) - (self.days_iter + 1)):
            base_temp = []
            emotion_temp = []
            for j in range(self.days_iter):
                base_temp.append(torch.Tensor([data_list[i + j][0]]))
                emotion_temp.append(torch.Tensor([data_list[i + j][1]]))
            open, high, low, close, *_ = data_list[i + j][0]
            next_open, next_high, next_low, next_close, *_ = data_list[i + j + 1][0]
            label = torch.from_numpy(np.array([next_close])).float()
            data_list_with_label.append([base_temp, emotion_temp, label])
        return data_list_with_label

    def load_emotion_data(self):
        """
        Calculate daily emotional features
        """
        news_df = pd.read_csv(self.out_chinese_news_emotion_path)
        emotion_dict = {}
        for index, row in news_df.iterrows():
            _, _, _, date, _, _, POS, NEG, DIV = row.tolist()
            date = date.split(" ")[0]
            if date not in emotion_dict:
                emotion_dict.setdefault(date, [[POS, NEG, DIV]])
            else:
                emotion_dict[date].append([POS, NEG, DIV])
        for key in list(emotion_dict.keys()):
            val = emotion_dict[key]
            if len(val) == 1:
                emotion_dict.update({key: val[0]})
            else:
                emotion_dict.update({key: [sum(x) / len(x) for x in list(zip(*val))]})
        return emotion_dict


if __name__ == "__main__":
    ea = EmotionAnalysis(positive_words_path="data/positive.txt",
                         negative_words_path="data/negative.txt",
                         stop_words_path="data/stopwords.txt")
    od = OriginalData(oil_data_path="data/crudeoil.xlsx",
                      chinese_news_path="data/china5e_news.csv",
                      out_chinese_news_emotion_path="./data/transform/chinese_news_emotion.csv")
    od.make_news_emotion(ea)
    od.transform()

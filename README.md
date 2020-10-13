# oil-futures-RLSTM

期货 R-LSTM模型

## Setup

    pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

## Download news data

    wget https://github.com/MagicianQi/oil-futures-RLSTM/releases/download/untagged-f85c553b57b2a85a6698/china5e_news.csv
    mv china5e_news.csv ./data/

## Run

    python train.py

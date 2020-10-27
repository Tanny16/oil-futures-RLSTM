# oil-futures-RLSTM

期货 R-LSTM模型

## Setup

    $ pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

## Download news data

    $ wget https://github.com/MagicianQi/oil-futures-RLSTM/releases/download/v1/china5e_news.csv
    $ mv china5e_news.csv ./data/

## Run

    $ ython train_LSTM.py
    $ python train_RLSTM.py

## Details

见 ‘train.py’

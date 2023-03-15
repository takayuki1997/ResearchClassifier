#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from flask import Flask, render_template, request
import torch
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
import pandas as pd
import csv

app = Flask(__name__)

# モデルをロード
data_path = "/home/ec2-user/env/data/"
loaded_model = BertForSequenceClassification.from_pretrained(data_path) 
# loaded_model.cuda() # GPU対応時はコメントアウトを外しcudaに変換
loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(data_path)

# 大区分の名称
Daikubun = ["Ａ", "Ｂ", "Ｃ", "Ｄ", "Ｅ", "Ｆ", "Ｇ", "Ｈ", "Ｉ", "Ｊ", "Ｋ"]



@app.route('/hw3beta', methods=['GET'])
def get2():
    return render_template('hw3beta.html')

@app.route('/hw3beta', methods=['POST'])
def post2():
    if request.method == 'POST':
        csv_data = request.files['csvfile'].read().decode('utf-8') # CSVファイルを文字列として取得
        csv_list = csv_data.splitlines() # 改行コードで分割
        csv_reader = csv.reader(csv_list) # CSVリーダーを作成
        csv_data_list = list(csv_reader) # CSVデータを2次元リストとして取得
        # ここからcsv_data_listを使った処理を記述
        # ...
    return render_template('hw3beta.html',
        data=csv_data_list,
        )

@app.route('/', methods=['GET'])
def get():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def post():
    sample_text = request.form['name']
    
    # Abst中の改行コードを削除
    sample_text = sample_text.replace('\r', '')
    sample_text = sample_text.replace('\n', '')

    max_length = 512
    words = loaded_tokenizer.tokenize(sample_text)
    word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
    word_tensor = torch.tensor([word_ids[:max_length]])  # テンソルに変換

    # x = word_tensor.cuda()  # GPU対応時はこちらを使う
    x = word_tensor # GPU未対応時はこちらを使う
    y = loaded_model(x)  # 予測
    y = y[0]
    pred = y.argmax(-1)  # 最大値のインデックス
    max_kubun = Daikubun[pred] # 最大値の大区分のアルファベット


    m = torch.nn.Softmax(dim=1) # Softmax関数で確率に変換
    y = m(y)
    yy = y.tolist()[0]
    yy = list(map(lambda x: int(x*100), yy))
    all_result = dict(zip(Daikubun, yy))

    # df_result = pd.DataFrame(data=yy, index=Daikubun)

    return render_template('index.html',
        max_kubun=max_kubun,
        all_result=all_result,
        sample_text=sample_text, # 判定するオリジナルのテキスト
        # result4=df_result,
        # result4=words, # トークナイズされたテキスト
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0')

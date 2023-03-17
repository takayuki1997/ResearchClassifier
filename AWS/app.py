#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from flask import Flask, render_template, request
import torch
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
import pandas as pd
import csv, io
import numpy as np

app = Flask(__name__)

# モデルをロード
data_path = "/home/ec2-user/env/data/"
loaded_model = BertForSequenceClassification.from_pretrained(data_path) 
# loaded_model.cuda() # GPU対応時はコメントアウトを外しcudaに変換
loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(data_path)

# 大区分の名称
Daikubun = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"] # 半角
# Daikubun = ["Ａ", "Ｂ", "Ｃ", "Ｄ", "Ｅ", "Ｆ", "Ｇ", "Ｈ", "Ｉ", "Ｊ", "Ｋ"] # 全角
num_Daikubun = len(Daikubun)
print('num_Daikubun: ', num_Daikubun)


def cat_estimation(sample_text, Daikubun):
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
    # all_result = dict(zip(Daikubun, yy))
    
    return max_kubun, yy


@app.route('/hw3beta.html', methods=['GET'])
def get2():
    return render_template('hw3beta.html')

@app.route('/hw3beta.html', methods=['POST'])
def post2():
    if request.method == 'POST':
        csv_data = request.files['csvfile'].read().decode('utf-8') # CSVファイルを文字列として取得
        csv_df = pd.read_csv(io.StringIO(csv_data), header=None) # dataframeに変換
        num_df = len(csv_df) # テキストの件数をカウント
        
        result_arr = np.zeros((num_df, num_Daikubun), dtype = int) # 分類推定の結果を入れるnumpy arrayを準備
        max_list = [None] * num_df # 推定した大区分を格納するリストを準備


        for i in range(num_df):

            sample_text = csv_df.iloc[i,0] # dataframeから一つのテキストを取り出す
            
            max_kubun, yy = cat_estimation(sample_text, Daikubun)
            
            max_list[i] = max_kubun
            result_arr[i] = yy
        
        result_df  = pd.DataFrame(result_arr, columns=Daikubun, index=[x+1 for x in range(num_df)])
        result_arr = result_arr.tolist()

    return render_template('hw3beta.html',
        result_arr=result_arr,
        max_list=max_list,
        result_df=result_df,
        )


@app.route('/', methods=['GET'])
def get():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def post():
    sample_text = request.form['name']
    
    max_kubun, yy = cat_estimation(sample_text, Daikubun)
    
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

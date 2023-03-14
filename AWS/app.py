#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from flask import Flask, render_template, request
import torch
from transformers import BertForSequenceClassification, BertJapaneseTokenizer

app = Flask(__name__)

# モデルをロード
data_path = "/home/ec2-user/env/data/"
loaded_model = BertForSequenceClassification.from_pretrained(data_path) 
# loaded_model.cuda() # GPU未対応===========================================
loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(data_path)

# 大区分の名称
Daikubun = ["Ａ", "Ｂ", "Ｃ", "Ｄ", "Ｅ", "Ｆ", "Ｇ", "Ｈ", "Ｉ", "Ｊ", "Ｋ"]


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

    # x = word_tensor.cuda()  # GPU対応
    x = word_tensor # GPU未対応==============================================
    y = loaded_model(x)  # 予測
    y = y[0]
    pred = y.argmax(-1)  # 最大値のインデックス
    out_put = "大区分"+Daikubun[pred]


    m = torch.nn.Softmax(dim=1) # Softmax関数で確率に変換
    y = m(y)
    yy = y.tolist()[0]
    yy = list(map(lambda x: int(x*100), yy))
    all_result = dict(zip(Daikubun, yy))

    # return render_template('index.html', result1=out_put, result2=all_result, result3=sample_text, result4=words)
    return render_template(
        'index.html',
        result1=out_put,
        result2=all_result,
        result3=sample_text,
        # result4=words,
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0')

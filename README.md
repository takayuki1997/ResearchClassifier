# BERTによる科研費審査区分の推定
自然言語処理モデル『BERT』を用いた下記のサンプルコードを修正し、科研費概要テキスト程度の学術文書から科研費審査区分（大区分）を推定するプログラムを作成した。  
ファインチューニング用データとして、科研費データベースからダウンロードした2019～2022年度開始の基盤BCの課題の「研究開始時の研究の概要」のテキストデータを用いた。  
基盤BCの小区分データを大区分に変換するための審査区分表を読み込み処理した。  
ファインチューニングはSageMaker Studio Labで行う。GPUも用いる。  
ウェブアプリに仕立てるため、AmazonのAWSのサービス（EC2）を用いた。１年間は基本無料。ただ、固定IPにするので少しお金がかかる。  
venvで環境を構築。flaskでウェブサーバを運用。  

# ウェブアプリケーションの運用
まずはWindows PowerShellで以下のコマンドによりAWSのサーバに接続
```
ssh -i "C:\Users\［ユーザ名］\.ssh\awsKeyPair.pem" ec2-user@［175.41.232.240］
```

サーバの起動
```
source env/bin/actiavate
cd env
nohup python app.py nohup flask run >out.log 2>err.log &
```
サーバの終了（当該PIDを確認し、その番号のプロセスを終了させる。）
```
ps -fA | grep python
kill ［ＰＩＤ］
```

簡易的なサーバの起動（terminalを閉じるとサーバが終了してしまうが、エラーログがすぐ表示されるので便利）
```
source env/bin/actiavate
cd env
python app.py
```
簡易的に立ち上げたサーバの終了
```
^C［Ctr+c］
```

# コード編集
## ウェブアプリケーションに関して
ローカルのコードをVScode等のエディタで編集し、以下のコマンドでサーバにアップする。
```
scp -i"C:\Users\［ユーザ名］\.ssh\awsKeyPair.pem" C:\Users\［ユーザ名］\Documents\GitHub\HW2beta\AWS\app.py ec2-user@［175.41.232.240］:~/env/app.py
scp -i"C:\Users\［ユーザ名］\.ssh\awsKeyPair.pem" C:\Users\［ユーザ名］\Documents\GitHub\HW2beta\AWS\templates\hw3beta.html ec2-user@［175.41.232.240］:~/env/templates/hw3beta.html
scp -i"C:\Users\［ユーザ名］\.ssh\awsKeyPair.pem" C:\Users\［ユーザ名］\Documents\GitHub\HW2beta\AWS\static\styles.css ec2-user@［175.41.232.240］:~/env/static/styles.css
```
いくつか修正し、一段落ついたポイントでGitHubにプッシュ

## 機械学習について
Amazonの「SageMaker Studio Lab」を利用  
https://studiolab.sagemaker.aws

# ウェブサーバの構築
Amazon AWS EC2 「Amazon Linux 2 AMI (HVM) - Kernel 5.10, SSD Volume Type」を選択した。

ssh接続のための鍵を設定

無料のインスタンスだとメモリ不足が生じるので、スワップファイルを設定する。  
https://repost.aws/ja/knowledge-center/ec2-memory-swap-file

```
sudo yum update
python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install flask
pip install transformers
pip install torch （そのままでは入らない。工夫が必要）
```

固定IP  
ポート5000を開ける

## 元となったプログラム
### BERTによる自然言語処理を学ぼう！ -Attention、TransformerからBERTへとつながるNLP技術-
自然言語処理の様々なタスクで高い性能を発揮する、「BERT」を学ぶ講座です。  
BERT（Bidirectional Encoder Representations from Transformers ）は2018年10月にGoogleが公開して以来、世界中のAI関係者の注目を集めています。  
Udemyコース: [BERTによる自然言語処理を学ぼう！](https://www.udemy.com/course/nlp-bert/?referralCode=276BD5473E099ACEAFCD)

# BERTによる科研費審査区分の推定
自然言語処理モデル『BERT』を用いた下記のサンプルコードを修正し、科研費概要テキスト程度の学術文書から科研費審査区分（大区分）を推定するプログラムを作成した。  
ファインチューニング用データとして、科研費データベースからダウンロードした2019～2022年度開始の基盤BCの課題の「研究開始時の研究の概要」のテキストデータを用いた。  
基盤BCの小区分データを大区分に変換するための審査区分表を読み込み処理した。  
ファインチューニングはSageMaker Studio Labで行う。GPUも用いる。  
ウェブアプリに仕立てるため、AmazonのAWSのサービス（EC2）を用いた。１年間は基本無料。ただ、固定IPにするので少しお金がかかる。  
venvで環境を構築。flaskでウェブサーバを運用。  

## 元となったプログラム
### BERTによる自然言語処理を学ぼう！ -Attention、TransformerからBERTへとつながるNLP技術-
自然言語処理の様々なタスクで高い性能を発揮する、「BERT」を学ぶ講座です。  
BERT（Bidirectional Encoder Representations from Transformers ）は2018年10月にGoogleが公開して以来、世界中のAI関係者の注目を集めています。 
Udemyコース: [BERTによる自然言語処理を学ぼう！](https://www.udemy.com/course/nlp-bert/?referralCode=276BD5473E099ACEAFCD)

# ハイパラチューニングの例

https://chat.openai.com/share/bdccdffb-52db-4240-8cf8-60946498ecb6
https://qiita.com/nynyny67/items/03b4dd785b597bcdc052
https://qiita.com/kenta1984/items/bad75a37d552510e4682
https://qiita.com/horiem/items/91d76ea25cb27e259495
https://dreamer-uma.com/pytorch-optuna-hyperparameter-tuning/


https://dreamer-uma.com/pytorch-mlp-mnist/

# やってみる


上のchatgpt様が書いたコードをそのまま実行したら動いた。一晩ねかせておいたら終わっていた。
中身を改めてみるとoptunaをimportしてnn.moduleを継承したNNを定義して
`optuna.create_study()`でstudyを作り
`study.optimize()`をしている。`study.optimize()`の中でtrainとtestをしているのでstudy.optimizeが重要っぽい。trial.suggest_floatやsuggent_categorical、suggest_intでパラメータを指定している。
その後dataloaderでデータを読み込んでtestもdataloaderで読み込んでいる
```
> test_loader には、手書き数字の画像データと、それに対応するラベル（0から9までの数字）が含まれています。これらのデータは、ネットワークの性能評価時に使用され、画像は正規化されたテンソル形式で提供されます。また、shuffle=True が設定されているため、データはランダムにシャッフルされます。

User
> trainのdataとtestのdataの違いが判らないです。どちらも手書きの数字と正解のラベルじゃないんですか？
ChatGPT

> はい、あなたの理解は正しいです。実際、trainデータセットとtestデータセットの両方とも手書きの数字の画像と、それに対応する正解のラベル（0から9までの数字）を含んでいます。主な違いは、これらのデータセットが使用される目的にあります。

trainデータセットは、モデルの訓練に使用されます。つまり、このデータを使用してモデルが数字を認識する方法を学習します。
testデータセットは、訓練されたモデルを評価するために使用されます。モデルがどれだけうまく未知のデータ（訓練に使用されなかったデータ）を処理できるかをテストするために使われます。
このように、両者の内容は同じですが、用途が異なります。モデルの一般化能力を適切に評価するため、通常、訓練データとテストデータは分けられます。

```
理解。麻雀のデータに正解は無いのでこれをやるのは難しそう。

麻雀における使用例をいくつかgithubであさってたんですが、テンパイ予測とかは使えそうだなと思った。
あとは正解の手牌(luckeyJの牌譜)があってそれに近づけていくみたいな。でもこれに関してはハイパラチューニングより強化学習するべきでは？と思った。

```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/hyper-param-tuning-example# python sample.py 
[I 2023-12-10 06:48:55,167] A new study created in memory with name: no-name-bd0d789f-e59a-42d5-9d29-430752e25b09
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST/raw/t10k-labels-idx1-ubyte.gz
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4542/4542 [00:00<00:00, 15289348.93it/s]
Extracting ./MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST/raw

[I 2023-12-10 06:49:30,612] Trial 0 finished with value: 0.919 and parameters: {'lr': 0.0010157768460326217, 'batch_size': 512, 'epochs': 3}. Best is trial 0 with value: 0.919.
[I 2023-12-10 09:10:38,179] Trial 99 finished with value: 0.9246 and parameters: {'lr': 0.0007466015332111168, 'batch_size': 64, 'epochs': 10}. Best is trial 38 with value: 0.9268.
Best trial:
  Value:  0.9268
  Params: 
    lr: 0.0011327711018591948
    batch_size: 64
    epochs: 10
```
import torch


# cnnで使われている各関数について

# nn.conv1d
# conv1dのconvはconvolutionの略
# nnconv1dには3つの引数がある
# in_channels: 入力のチャンネル数
# out_channels: 出力のチャンネル数
# kernel_size: カーネルサイズ

# nn.conv1d.weight.shapeでデバッグ出来る
# conv1dの引数には3次元の配列を渡す必要がある
# 1次元目はバッチサイズ
# 2次元目はチャンネル
# 3次元目は入力のサイズ


## conv1dには重みとバイアスが存在する
# 重みはnn.conv1d.weightで取得できる
# 重みのshapeは(out_channels, in_channels, kernel_size)である
# バイアスはnn.conv1d.biasで取得できる

# nn.conv1dにおいてバイアスの数は出力チャンネルの数と一致する
# はい、Conv1dにおいて、通常は出力チャンネルの数とバイアスの数は一致します。出力チャンネル数がnである場合、通常、Conv1dレイヤーのバイアスの数もn個です。各出力チャンネルに対して異なるバイアスが存在し、それぞれの出力チャンネルに対するオフセットを表現します。

# 重みとはカーネルサイズの係数
# 重みとは入力データと畳み込みを行う際に使用されるパラメータのことです。畳み込みの際には、入力データとカーネルの要素ごとに掛け算を行い、その総和を出力します。このとき、カーネルの要素を重みと呼びます。
# conv1d.weight = torch.nn.Parameter(torch.randn(16, 1, 3))のように重みを設定することができる(conv1dの引数とは別で)。設定値にはtorch.Tensorを渡す必要がある
# 同じようにバイアスも設定できる

# チャネル数とは
# https://www.atmarkit.co.jp/ait/articles/2004/02/news016.html

# conv2d
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html


## torch.nn.sequential

# torch.nn.Sequential() # で簡単に各層をつなげることができる

## nn.MaxPool1d
# 1次元の最大プーリングを行う
# 最大プーリングは入力データから最大値を取り出す操作を行う。プーリングは畳み込みと同様に、入力データに対してカーネルを適用することで行われる。ただし、プーリングではカーネルの要素ごとに掛け算を行うのではなく、カーネルの要素の最大値を出力する。

## nn.Linear
## nn.Linearは層を表すが、蜜結合層や全結合層とも呼ばれる。線形変換を定義する。線形返還は、入力と重み行列の積にバイアスを加えることで定義される。


# nn.Linearは、PyTorchのニューラルネットワークモデルで使用される線形変換（全結合層または密結合層とも呼ばれる）を定義するためのクラスです。線形変換は、入力データと重み行列の行列積を計算し、バイアスを加えて出力を生成する操作を行います。これは、多くの異なるタイプのニューラルネットワークアーキテクチャで使用されます。

# nn.Linearの主な引数は以下の通りです：

# in_features: 入力の特徴量の数。つまり、入力データの次元です。
# out_features: 出力の特徴量の数。つまり、この層の出力の次元です。
# bias: バイアス項を使用するかどうかを指定するブール値。デフォルトではTrueです。


import torch.nn as nn

# 入力特徴量数が3、出力特徴量数が2の線形変換層を定義
linear_layer = nn.Linear(in_features=3, out_features=2)

# 入力データを生成（バッチサイズ1、入力特徴量数3）
input_data = torch.tensor([1.0, 2.0, 3.0])

# 線形変換を適用
output = linear_layer(input_data)
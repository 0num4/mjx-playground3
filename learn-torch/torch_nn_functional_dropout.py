import torch

sample = torch.tensor([1, 2, 3, 4, 5], dtype=float)

# class sampleNN(torch.nn.Module):
#     __init__(self):
#         self.fc = 

# p=0.5なので50%の確立でtensorの値が落ちるはず
# 訓練時にはtrue、評価時にはfalse
sample2 = torch.nn.functional.dropout(sample, p=0.5, training=False)
print(sample)
print(sample2)

# training=Trueのdropout
# dropoutしなかった要素が2倍になっているのは全体の平均を取るためスケーリングされる。そのノードの値は1/(1-p)倍されます。
# tensor([ 1.,  2.,  3.,  4.,  5.], dtype=torch.float64)
# tensor([ 0.,  4.,  0.,  8., 10.], dtype=torch.float64)

# false
# tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
# tensor([1., 2., 3., 4., 5.], dtype=torch.float64)


# torch.nn.functional.dropoutを使用する際にテンソルの要素が2倍になる現象は、ドロップアウトの動作に起因しています。ドロップアウトでは、ノード（この場合はテンソルの要素）がランダムに選択され、確率pでゼロに設定されます。残りのノードは、ドロップアウトしなかったノードの値を補正するためにスケーリングされます。
# 具体的には、ドロップアウトを適用しない場合（つまり、ノードが保持される場合）、そのノードの値は1/(1-p)倍されます。これは、ドロップアウトによってゼロに設定されたノードの影響を平均化し、全体の出力の期待値を維持するためです。
# 例えば、p=0.5（50%のノードがドロップアウトされる）の場合、残ったノードは2倍されます。これは、全体のノードの半分がゼロになるため、残りのノードの出力を2倍して全体の平均を保つためです。
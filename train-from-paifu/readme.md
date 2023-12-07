# 天鳳の牌譜を学習してMjx(v0.1.0)で使えるAgentを作る

https://note.com/oshizo/n/n61441adc340c

batchsize1024 10epoch 122牌譜で計算してもこのくらいでした。
```
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots.py 
[-135, -135, -135, 90, -135, -135, -135, -135, 0, -135, -135, 0, 0, -135, -135, 0, 0, -135, 0, -135, 0, -135, -135, 0, 0, 0, -135, -135, 0, -135, -135, 45, -135, 0, 45, -135, -135, 0, 90, -135, 90, 90, -135, -135, -135, 45, 45, 45, 0, 90, 45, -135, 90, 45, 0, 0, 45, 0, 45, -135, -135, -135, 45, 0, -135, -135, -135, 0, 0, 0, 90, -135, -135, -135, 0, 90, 0, 0, 0, 0, -135, -135, -135, -135, -135, 90, 45, 45, -135, -135, 0, 45, 45, -135, -135, -135, 0, -135, 90, -135]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# 
```
1位: 28,2位: 14,3位: 10, 4位: 48
shantenbotを100半荘で学習させた時よりちょっと強くなってるっぽいです。


# 全ての天鳳位の牌譜データを読み込む
read_from_mjxproto.old.pyで読み込んでいたが、重すぎて一日たっても終わらなかった

全ての天鳳位の牌譜データを読み込んで、zipで解凍し、mjxproto.jsonに変換して、使えないファイルは除き、0バイトを含む10kb以下のファイルは除き
使えるデータだけに絞った結果65909牌譜になった。


## npyファイルを作る

普通にread_from_mjxproto.pyを実行すると、メモリ不足で死んだり、重すぎて一日たっても終わらなかったりした。

最初に動かしたread_from_mjxproto.old.pyは、一日たっても終わらず、15%ぐらいしか進まなかった。最初のほうは早かったんですが読み込むにつれてだんだん速度が落ちていったのでobs_histなどの配列の数が回す旅に比例して増えているのが原因だと思われる。

```
MPSだとGPUをtrainで使う設定にしても20%ぐらいしか使用率があがらない(タスクマネージャーで見た)
`    trainer = pl.Trainer(max_epochs=2, accelerator="cuda", devices="1")
`

```
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning2.py
loading data
loaded obs
loaded end
/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2.py:42: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
  dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
loaded dataset
loaded dataloaders
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2.py", line 56, in <module>
    learn()
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2.py", line 48, in learn
    trainer = pl.Trainer(max_epochs=2, accelerator="mps", devices="1")
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/utilities/argparse.py", line 70, in insert_env_defaults
    return fn(self, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 401, in __init__
    self._accelerator_connector = _AcceleratorConnector(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 155, in __init__
    self._set_parallel_devices_and_init_accelerator()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 387, in _set_parallel_devices_and_init_accelerator
    raise MisconfigurationException(
lightning_fabric.utilities.exceptions.MisconfigurationException: `MPSAccelerator` can not run on your system since the accelerator is not available. The following accelerator(s) is available and can be passed into `accelerator` argument of `Trainer`: ['cpu', 'cuda'].
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
```

## カスタムデータローダーを使わない方針に戻す

重すぎて一生終わらなかったりcustomdataloaderだと途中で急に止まって動かなくなったりしたので元のdatloaderに戻す。そして重すぎるのでepoch数やbatchsizeを減らしてちゃんと動くか確認してみた。動いた、20分ぐらいだった。

```
    loader = DataLoader(dataset, batch_size=512, num_workers=15)  # shuffle=True
    print("loaded dataloaders")
    trainer = pl.Trainer(max_epochs=2, accelerator="cuda", devices="1")
```

```
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning2.py
loading data
loaded obs
loaded end
/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2.py:42: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
  dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
loaded dataset
loaded dataloaders
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
training start
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Missing logger folder: /mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/lightning_logs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name        | Type             | Params
-------------------------------------------------
0 | net         | Sequential       | 109 K
1 | loss_module | CrossEntropyLoss | 0
-------------------------------------------------
109 K     Trainable params
0         Non-trainable params
109 K     Total params
0.438     Total estimated model params size (MB)
Epoch 1: 100%|██████████████████████████████████████████████████████████| 85625/85625 [06:55<00:00, 205.85it/s, v_num=0]`Trainer.fit` stopped: `max_epochs=2` reached.
Epoch 1: 100%|██████████████████████████████████████████████████████████| 85625/85625 [06:55<00:00, 205.84it/s, v_num=0]
training end
saved model
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#

```

batchsize1024にして10epoch回してみたら、2時間ぐらいで終わった。

```
[0, 90, 45, -135, 0, -135, 45, -135, 45, 0, 90, 90, 0, 90, 45, 90, 90, 0, 90, 90, 90, 0, 90, 0, 90, 45, 90, 90, 90, 90, -135, 0, 90, 90, 90, 45, 90, 45, -135, 90, -135, 90, 45, 0, 90, 90, -135, 90, -135, 45, 45, 45, 90, 45, 90, -135, -135, 45, 90, 0, 90, -135, 90, 45, 90, -135, 90, 90, 90, -135, 90, 90, 90, 90, 0, -135, 90, 90, 90, 90, 90, -135, 90, 90, -135, 0, 45, -135, 45, 45, 90, -135, -135, 90, 90, 90, 45, 45, -135, 0]
1位 (90点): 48回
2位 (45点): 19回
3位 (0点): 13回
4位 (-135点): 20回 ​
```
確実に強くなっててよかった。

1000半荘回したらこんな感じ
平均順位: 2.045
一位: 459
二位: 195
三位: 188
四位: 158



https://chat.openai.com/share/4641d974-d75e-46f4-9160-b76b33867fec


full_full.pthとmodel_shanten_100.pthを使って100半荘回したらこんな感じ
```
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python mlp_agent_playground.py
[0, 90, -135, 90, -135, 90, 45, 90, 90, 0, 90, 0, 90, -135, 0, 90, 90, 0, 90, 90, 90, 0, 0, 0, 0, 90, 90, 90, 90, -135, 45, 45, 90, 90, 45, 90, 90, -135, -135, 90, 90, 90, 90, 90, 90, 45, 45, 90, 90, 90, 45, 90, 90, -135, 90, 90, 0, 90, 90, 0, 90, 90, 45, 90, 90, 90, 90, 0, 90, 90, -135, -135, -135, 45, 0, 90, 90, 0, 90, 0, 45, 90, 90, 90, 90, 90, 90, -135, 45, 90, 45, 45, -135, 45, -135, 90, 90, 90, 90, 0]

平均順位: 1.85
一位: 57
二位: 14
三位: 16
四位: 13
```
圧倒的に強くなってることがわかる


full_full.pthとfull(read_from_paifu).pthを使って1000半荘回したらこんな感じ。勝ち越していることがわかってよかった。
```
平均順位: 2.172
一位: 354
二位: 287
三位: 192
四位: 167
```

## train_from_paifu.py

* train_from_paifu.old.py・・・1牌譜だけ読み込んで学習させてみたやつ。オリジナルのコピー
* train_from_paifu.py・・・1天鳳位だけの全ての牌譜をconvert&読み込めるファイルだけに除外したディレクトリから読み込んで学習させるように変換した。200ファイル(半荘)ほど。

* train_from_paifu.new.py・・・全ての天鳳位の牌譜を読み込めるようにした。65909ファイル(半荘)ほど。100GB超えのnpyファイルを作る。1日ぐらいかかるし、めっちゃリソース食う(128GB+swap)。chatgptに聞きながら並列処理や読み込みのバッチ分割などを行った(ログがchatgptのどこかに残ってるはず)
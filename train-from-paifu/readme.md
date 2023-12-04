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



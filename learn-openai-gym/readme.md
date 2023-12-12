# learn openai gym
https://qiita.com/ishizakiiii/items/75bc2176a1e0b65bdd16
openaiが作った強化学習のためのライブラリ

強化学習とはある環境においてエージェントが自律的に行動を選択し、その結果として得られる報酬を最大化するように学習すること
**教師あり学習と強化学習は違う**

## pytoarchのdataloaderについて
https://zenn.dev/megane_otoko/articles/074_simple_pytorch

https://qiita.com/tetsuro731/items/d64b9bbb8de6874b7064

# memo
landarは動いた。(画面に描写して学習しているところまで見えた)

mountaincarもcatepoleも動いたがgym wrapperとかが変わりすぎたのでgymnasiumを使おうと思う。

# オンライン強化学習とオフライン強化学習について


# 強化学習のライブラリ一覧
https://docs.agilerl.com/en/latest/
agalierl
```
poetry add agilerl
```

あ、はい
```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# poetry add agilerl
Using version ^0.1.19 for agilerl

Updating dependencies
Resolving dependencies... (10.7s)

Because no versions of agilerl match >0.1.19,<0.2.0
 and agilerl (0.1.19) depends on gymnasium (>=0.28.1,<0.29.0), agilerl (>=0.1.19,<0.2.0) requires gymnasium (>=0.28.1,<0.29.0).
And because gymnasium (0.29.1) depends on gymnasium (0.29.1)
 and no versions of gymnasium match >0.29.1,<0.30.0, agilerl (>=0.1.19,<0.2.0) is incompatible with gymnasium (>=0.29.1,<0.30.0).
So, because mjx-playground depends on both gymnasium (^0.29.1) and agilerl (^0.1.19), version solving failed.
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# 
```
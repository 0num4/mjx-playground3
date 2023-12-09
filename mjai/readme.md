# mjaiにbotを投稿する

https://note.com/oshizo/n/n3764a42af7d2

MLP()とMLPAgent()を同じファイルに実装してmodelファイルも同梱してbot.pyを固めてzipとして投稿する。

# mjaiのioの簡易検証

https://mjai.app/docs/highlevel-api
ここのquick checkを参考にして、mjaiのioを簡易的に検証する。

```
cd mjai
python examples/rulebase/bot.py 1 < tests/mjai/bot/data_base_akadora.log
```

## mjai.appからmjai(python module)を持ってくる

```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/mjai# python examples/rulebase/bot.py 1 < tests/mjai/bot/data_base_akadora.log
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/mjai/examples/rulebase/bot.py", line 8, in <module>
    from mjai import Bot
ModuleNotFoundError: No module named 'mjai'
```

```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# poetry add ../mjai.app/

```
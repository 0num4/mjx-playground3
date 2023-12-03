# shantenBotを使った教師あり学習
https://note.com/oshizo/n/nbbcfb4e24908
https://note.com/oshizo/n/n4eae69dbeb23

https://github.com/mjx-project/mjx/blob/master/mjx/agents.py#L50
普通にclass作って実装しているだけ

## poetry
```
poetry source add torch_cu118 --priority=explicit https://download.pytorch.org/whl/cu118
poetry add torch torchvision torchaudio --source torch_cu118
```

作ったmodelをインスタンス化してoptを渡して学習

## gym
openai gymのことっぽい

GymEnvはmjx.envのreset()とstep()を上書きしてるっぽい
mjx.env()
https://vscode.dev/github/0num4/mjx-playground3/blob/mainpda-py3.9/lib/python3.9/site-packages/mjx/env.py#L11

実行結果
上がrandombotで下がshantenbot

```
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/supervised_learning_from_shantenBot# python battle_vs_shantenbots.py 
[90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/supervised_learning_from_shantenBot# python battle_vs_shantenbots.py 
[45, 90, -135, -135, 90, -135, 0, 0, -135, -135, 45, 0, 0, 0, -135, -135, -135, -135, -135, -135, 0, 45, 0, -135, 0, 90, -135, -135, -135, 45, -135, -135, -135, -135, -135, 0, -135, 0, -135, -135, 0, 0, 0, 45, 0, 90, -135, -135, -135, -135, -135, -135, -135, 0, -135, 0, -135, 45, 0, 0, -135, -135, 45, 0, 0, 0, 0, 0, 0, 0, -135, -135, -135, -135, -135, -135, 0, 0, 0, 0, 0, 0, -135, 45, 0, -135, -135, -135, -135, -135, 90, -135, 0, 0, -135, 90, 0, 45, 45, 90]
```
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

# 状態を可視化 && もっと!!mjx-playground

## 可視化
```
            features = obs.to_features(feature_name="mjx-small-v0")
            # ndarrayを画像に変換
            featurelist = features.tolist()  # dump(f"svg/{file_id}.dmp")
            json_data = json.dumps(featurelist)
            with open(f"svg/{file_id}.json", 'w') as f:
                json.dump(json_data, f)

            obs.save_svg(f"svg/{file_id}.svg")
```
obs.save_svg()でsvgファイルを保存できる。



obs.show_svg()はJupyter Notebookでしか使えない。
```
This function only works in Jupyter Notebook.Traceback (most recent call last):
```

## もっと!!mjx-playground

obsに大体の情報が入っている。

```
player_id player_0
len(legal_actions) 14
obs.curr_hand().to_json() {"closedTiles":[20,27,36,44,51,52,55,58,70,73,100,107,127,129]}
obs.curr_hand().shanten_number() 3
obs.who()0
obs.dealer()0
obs.doras()[10]
obs.draws()[<mjx.tile.Tile object at 0x7f108ece00a0>]
draw.id() 73 draw.type() TileType.S1 draw.is_red() False draw.num() 1
```

## env, obs, actionの関係
**env.reset()とenv.step()で帰ってくる型は同じ。**
["player_2", obs]みたいな。
基本1つしか入ってないんですが念のためforでまわしてるっぽい
`for player_id, obs in obs_dict.items(): `

env.step()の返り値で帰ってくる1つめのstrとobs.who()は一致してそう


## obsの中身
大体使えそうなのは上のメソッドあたり。
河の情報などを見ることはできない？
obs.draws()で引いてきた牌の情報の履歴が見える

## state
stateにもsave_svg()がある。stateで保存したsvgの場合全てのプレイヤーの牌が見える。
```
state = env.state()
state.save_svg("svg/test.svg")
```

**state.past_decisions()を使うとなぜかセグフォで落ちたりめちゃくちゃ重い。なんだこのメソッド・・・**

save_svgの山の残りはwallの中にある。

## その他ゲームの情報など

svgの中で使われているMahjongTableを使うと早そう。例えばこれは山の残り枚数。
```
            proto_data = obs.to_proto()
            sample_data = MahjongTable.decode_observation(proto_data)
            print("sample_data.wall_num: " + str(sample_data.wall_num))
```

## ゲームの進行
env.reset()からenv.reset()の間は1ゲーム。次の局に行ったときの処理とかはいらずにforの中ですっと始まる。sample.pyのように。

ゲームごとの処理を入れたいときはa = obs.round()でifで次の局に行ったかどうかなどを判定するしかなさそう。
```python
    while not env.done():
        for player_id, obs in obs_dict.items():
            print("player_id "+player_id)
            print(str(obs.round()) + "局") #ここの値がすっと変わる。env.done()は局ごとではないため。
```
**done("game")の引数でgameかroundか選べるっぽい**

```python
for game in range(2):  # 100半荘回す
    env.reset()  # ゲーム開始
    round = 0
    while not env.done():
        for player_id, obs in obs_dict.items():
            if round != obs.round():
                print(f"round {round} -> {obs.round()}")
                print("次の局にいきました "+str(obs.round()))
                round = obs.round()
```

## 放銃率、ツモ率などの計算

legal_actions()で行動できるactionsが取れるがdiscardがたくさんある。これは捨てる牌ごとにあるので13個ぐらいあると思う。
```
for action in legal_actions:
    print(action.tile().type())
    print(action.type().name)
```
```
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
DISCARD
<bound method Tile.type of <mjx.tile.Tile object at 0x7f90546e6b20>>
TSUMOGIRI
```

# for player_id, obs in obs_dict.items():について。
とんなんしゃーぺーの順でまぁ回ってくと思うんですがポンとかチーとかできる場合例えば
player 0123012303みたいにぐるぐるならない。
行動が出来る人が先にforで回ってくるのでaction.type().name == "RON"の場合historyを取っておいて一番最後のplayer_idを持ってこれば放銃者がわかり、そこから放銃率、ツモ率が計算できる。

# envについて

run()はserver, client通信で使う？普通の対局で使えるのか

普通の対局ではreset()とstep()を使うが。

reset()のcppの処理
https://github.com/mjx-project/mjx/blob/master/include/mjx/env.cpp#L15

seed_gen_()でseedを作り
playerをシャッフルし、
```
  // initialize state
  state_ = internal::State(
      mjx::internal::State::ScoreInfo{shuffled_player_ids, seed.value()});
```
してobsを返す


step()・・・内部のstateをガチャガチャして返す.
state.next()で次のゲームに行ってるっぽい。大事なのはstate_.Update(std::move(actions));かなぁ
```cpp
std::unordered_map<PlayerId, Observation> MjxEnv::Step(
    const std::unordered_map<PlayerId, mjx::Action>& action_dict) noexcept {
  std::unordered_map<PlayerId, Observation> observations;

  if (state_.IsRoundOver() && !state_.IsGameOver()) {
    auto next_state_info = state_.Next();
    state_ = mjx::internal::State(next_state_info);
    return Observe();
  }

  std::vector<mjxproto::Action> actions;
  actions.reserve(action_dict.size());
  for (const auto& [player_id, action] : action_dict)
    actions.push_back(action.proto());
  state_.Update(std::move(actions));
  return Observe();
}
```

EnvRunner::EnvRunnerは内部でMjxEnv(player_ids_);を呼び出している
env.resetとかもやってスレッド間で通信してるっぽいね。
https://github.com/mjx-project/mjx/blob/master/include/mjx/env.cpp#L179


pythonのmjx.env.runはagentを4つ取るが、その引数にはGrpcAgentしか受け取らないので、普通にshantenAgentとかは渡せないっぽい。
```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization# python speed_eval.py 
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/speed_eval.py", line 28, in <module>
    run(num_games=1000, agent_addresses={
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/mjx/env.py", line 69, in run
    agents = {k: _mjx.GrpcAgent(addr) for k, addr in agent_addresses.items()}  # type: ignore
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/mjx/env.py", line 69, in <dictcomp>
    agents = {k: _mjx.GrpcAgent(addr) for k, addr in agent_addresses.items()}  # type: ignore
TypeError: __init__(): incompatible constructor arguments. The following argument types are supported:
    1. _mjx.GrpcAgent(arg0: str)

Invoked with: <mjx.agents.RandomAgent object at 0x7ff8bd046d10>
```

嘘、python.mjx.env.runの中のagentの受け取り変えたら行けてそうな気がする。でも1000で回したからか一切反応ないから死んでそう・・・
```
    agents = {k: addr for k, addr in agent_addresses.items()}  # type: ignore
```

# ゲームの評価と処理速度

```
time python speed_eval.py
[array of 1000 game ranks]
real    1m25.208s
user    1m25.328s
sys     0m0.930s
```

100半荘だとshantenAgentで20秒ほど。

処理速度についてはmjx projectの別のリポジトリにもある。(ruby-mjaiとの比較だからあんま意味ないけど・・・)
https://github.com/mjx-project/speed_benchmark/tree/master

## 並列処理を使った高速化

2分が10秒になった。gpu使えればもっと早くなるかもしれない。
また、cpu使用率もちゃんと100%になっていた。
```
time python chatgpt-multiprocessing_speed_eval.py
[array of 1000 game ranks]
real    0m10.678s
user    2m37.529s
sys     0m2.652s
```

1万局のオーダーも30分が1分ちょいになって大分現実味を帯びた
```
time python chatgpt-multiprocessing_speed_eval.py
[array of 10000 game ranks]
real    1m41.313s
user    26m8.795s
sys     0m3.563s
```

# codonを使った高速化
codonもgpu使えるしpythonのjitコンパイラなので特に手を入れる必要がないはず。

https://github.com/exaloop/codon
wsl上もこれでインストール出来た。
```
/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

初期状態だとpoetryの環境は認識されず、mjxが無いといわれる。
----

うまくpoetryの環境が認識出来ず、別環境で一回poetry newしてそこでcodon run hoge.pyした。python3.11と比べても引けを取らない速さ。
```
(3.11.4) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/test/codontest# python helloworld.py 
ackermann(3, 11) = 16381
elapsed time: 14128 [ms]
(3.11.4) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/test/codontest# codon run helloworld.py 
helloworld.py:5:1-33: error: cannot import name 'setrecursionlimit' from 'sys'
(3.11.4) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/test/codontest# codon run helloworld.py 
ackermann(3, 11) = 16381
elapsed time: 318 [ms]
(3.11.4) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/test/codontest# python --version
Python 3.11.4
(3.11.4) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/test/codontest# 
```

## なんかガチャガチャやったら動いた
ただmjx.agents import RandomAgentみたいなインポート方法がわからない
time codon run visualization/speed_eval.py 
```python
# https://note.com/oshizo/n/nbbcfb4e24908
from python import mjx
# from python import mjx.agents.RandomAgent
# from mjx.env import run
env = mjx.MjxEnv()
agent = mjx.agents.RandomAgent()
obs_dict = env.reset()  # game start
# print(obs_dict)
rank_hist = []

for game in range(100):  # 100半荘回す
    env.reset()  # ゲーム開始
    round = 0
    while not env.done():
        actions = {player_id: agent.act(obs) for player_id, obs in obs_dict.items()}
        # print(actions)
        obs_dict = env.step(actions)
    # env.state()
    rank_hist.append(env.rewards())
print(rank_hist)

```

速度を比較してみたんですが100半荘だとcodonが12sでpythonが8sだった。草。高速化とは。

とりあえずまた1000半荘で比較。

python:
cpuは10-20%前後。
real    1m21.140s
user    1m21.195s
sys     0m1.030s

codon:
time codon run visualization/speed_eval_codon.py
こちらもcpuは10-20%前後。
real    1m39.720s
user    1m36.845s
sys     0m4.834s

はい。草。

___
アッカーマン関数で比較してみる。
```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# python visualization/akkaman.py 
ackermann(3, 11) = 16381
elapsed time: 30936 [ms]
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# codon run visualization/akkaman.py 
ackermann(3, 11) = 16381
elapsed time: 336 [ms]
```

やっぱ生のpythonだと効果出てそうだけどね。

# numbaを使った高速化
まぁアッカーマン関数なら一瞬。
```
(base) root ➜ /workspaces/vscode-devcontainer-test (feat/cppdebug) $ python sample.py 
ackermann(3, 11) = 16381
elapsed time: 15355 [ms]
(base) root ➜ /workspaces/vscode-devcontainer-test (feat/cppdebug) $ python sample.py 
/workspaces/vscode-devcontainer-test/sample.py:9: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  @jit
ackermann(3, 11) = 16381
elapsed time: 1038 [ms]
```

wsl上のpoetryでも特に問題なく一瞬でインストール&実行できた。
```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# python visualization/akkaman.py 
/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/akkaman.py:10: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  def ackermann(m: int, n: int) -> int:
ackermann(3, 11) = 16381
elapsed time: 1546 [ms]
```

1000半荘回す奴でもimport numba出来てまぁwarnは出てるが普通に動いてそう。
```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# python visualization/speed_eval_numba.py 
/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/speed_eval_numba.py:8: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
  def main():
/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/speed_eval_numba.py:7: NumbaWarning: 
Compilation is falling back to object mode WITH looplifting enabled because Function "main" failed type inference due to: Unknown attribute 'MjxEnv' of type Module(<module 'mjx' from '/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/mjx/__init__.py'>)

File "visualization/speed_eval_numba.py", line 9:
def main():
    env = mjx.MjxEnv()
    ^

During: typing of get attribute at /mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/speed_eval_numba.py (9)

File "visualization/speed_eval_numba.py", line 9:
def main():
    env = mjx.MjxEnv()
    ^

  @jit
/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/speed_eval_numba.py:7: NumbaWarning: 
Compilation is falling back to object mode WITHOUT looplifting enabled because Function "main" failed type inference due to: Cannot determine Numba type of <class 'numba.core.dispatcher.LiftedLoop'>

File "visualization/speed_eval_numba.py", line 15:
def main():
    <source elided>

    for game in range(1000):  # 100半荘回す
    ^

```

time python visualization/speed_eval_numba.py
特に変わらないねぇ
real    1m22.179s
user    1m22.581s
sys     0m1.801s

https://qiita.com/tariaki/items/83a9115d672fe9c3becb

numbaが遅い原因は大体理解した。
```
(mjx-playground-S0ozRpda-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground# time python visualization/speed_eval_numba.py 
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/speed_eval_numba.py", line 27, in <module>
    main()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/numba/core/dispatcher.py", line 468, in _compile_for_args
    error_rewrite(e, 'typing')
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/numba/core/dispatcher.py", line 409, in error_rewrite
    raise e.with_traceback(None)
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
Unknown attribute 'MjxEnv' of type Module(<module 'mjx' from '/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/mjx/__init__.py'>)

File "visualization/speed_eval_numba.py", line 9:
def main():
    env = mjx.MjxEnv()
    ^

During: typing of get attribute at /mnt/c/Users/Owner/work/private/mahjong/mjx-playground/visualization/speed_eval_numba.py (9)

File "visualization/speed_eval_numba.py", line 9:
def main():
    env = mjx.MjxEnv()
    ^

ChatGPT

エラーメッセージは、NumbaがmjxモジュールのMjxEnvクラスを認識できないことを示しています。これは、Numbaのnopythonモードでは、特定の外部モジュールやその属性が認識されないため発生する可能性があります。Numbaのnopythonモードは、PythonのオブジェクトやPython C APIに依存しないコードのコンパイルを試みますが、すべてのPythonモジュールや関数がnopythonモードで動作するわけではありません。

この問題を解決するためには、MjxEnvがNumbaと互換性のある方法で実装されているかを確認するか、MjxEnvを使用する部分をNumbaのデコレータから除外する必要があります。また、Numbaのドキュメントで互換性に関する詳細情報を確認することも有効です。


```

# codonをビルドしてみる
codon build -o spdevl visualization/speed_eval_codon.py
real    1m39.123s
user    1m36.869s
sys     0m4.899s

コンパイルしても速度変わらないが。

**codonをgpu上で動かすにはソースからコンパイルする必要があるが、それにはllvmなどいろいろ必要があるらしく、普通に終わった(vs-code-dev-container-testにある)**
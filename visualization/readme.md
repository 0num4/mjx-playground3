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

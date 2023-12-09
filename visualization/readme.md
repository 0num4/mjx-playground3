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

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
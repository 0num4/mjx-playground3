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
print("player_id "+player_id)
# print(legal_actions)
print("legal_actions"+str(len(legal_actions)))
print("obs.curr_hand().to_json()")
print(obs.curr_hand().to_json())
# print(obs.curr_hand().to_json())
print("obs.who()" + str(obs.who()))
print("obs.dealer()" + str(obs.dealer()))
print("obs.doras()" + str(obs.doras()))
```

## env, obs, actionの関係
**env.reset()とenv.step()で帰ってくる型は同じ。**
["player_2", obs]みたいな。
基本1つしか入ってないんですが念のためforでまわしてるっぽい
`for player_id, obs in obs_dict.items(): `

env.step()の返り値で帰ってくる1つめのstrとobs.who()は一致してそう
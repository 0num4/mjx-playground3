from time import sleep
import mjx
from mjx.agents import ShantenAgent
import json
import numpy as np
import tqdm

env = mjx.MjxEnv()
obs_dict = env.reset()  # state
state = env.state()
state.save_svg("svg/test.svg")
# Agentの入力になるObservation（観測）には、プレイヤーの席順、点数、手配、捨て牌など様々な情報が含まれています。
# https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/observation.py#L19
# Mjxには、観測を機械学習モデルで扱いやすい行列にするメソッドto_featuresが用意されています。
agent = ShantenAgent()
# obj_str, obs_tmp = obs_dict  # state

obs_hist = []
action_hist = []

for game in range(2):  # 100半荘回す
    env.reset()  # ゲーム開始
    while not env.done():
        counter = 0
        actions = {}  # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/action.py#L14
        for player_id, obs in obs_dict.items():  # 1巡目ということだと思う
            # obsはcppのオブジェクトで中は何も見えない
            legal_actions = obs.legal_actions()  # アクションのリストをobsから取ってくる？
            # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/observation.py#L57
            print("player_id "+player_id)
            # print(legal_actions)
            print("len(legal_actions) "+str(len(legal_actions)))
            print("obs.curr_hand().to_json() "+str(obs.curr_hand().to_json()))
            print("obs.curr_hand().shanten_number() " + str(obs.curr_hand().shanten_number()))
            # print(obs.curr_hand().to_json())
            print("obs.who()" + str(obs.who()))
            print("obs.dealer()" + str(obs.dealer()))
            print("obs.doras()" + str(obs.doras()))
            print("obs.draws()" + str(obs.draws()))  # drawsが何故か複数ある
            for draw in obs.draws():
                print("     draw.id()) "+str(draw.id()) + " draw.type() "+str(draw.type()) + " draw.is_red() "+str(draw.is_red()) + " draw.num() "+str(draw.num()))
            print("obs._repr_html_() ")  # svgが出てくる
            # ファイルに保存
            # with open(f"svg/{counter}_{player_id}.svg", mode='w') as file:
            #     file.write(obs._repr_html_())

            action = agent.act(obs)  # actionもcpp objなので見えないけどagentが取ったアクションだと思う
            actions[player_id] = action
            # obs.show_svg()
            file_id = f"{counter}_{player_id}_{action.to_idx()}_{obs.who()}_{action.type().name}"
            features = obs.to_features(feature_name="mjx-small-v0")
            # ndarrayを画像に変換
            featurelist = features.tolist()  # dump(f"svg/{file_id}.dmp")
            json_data = json.dumps(featurelist)
            with open(f"svg/{file_id}.json", 'w') as f:
                json.dump(json_data, f)

            obs.save_svg(f"svg/{file_id}.svg")
            # 選択できるアクションが複数ある場合、obsとactionを保存する
            if len(legal_actions) > 1:
                obs_hist.append(obs.to_features(feature_name="mjx-small-v0").ravel())  # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/observation.py#L111
                action_hist.append(action.to_idx())
        # print(actions)  # player_0しか入ってない・・・actionは1行動でstep()を刻むことで次の打牌とかツモになるっぽい
        obs_dict = env.step(actions)  # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/env.py#L25
        # print(obs["player_3"].save_svg(f"svg/{obs['player_3'].}.svg"))
        # print(obs_dict[1].curr_hand().shanten_number())
        # print(anystr)
        print("------------for end------------")
        sleep(1)
    env.reset()

# ファイルに書き出し
np.save("shanten_obs.npy", np.stack(obs_hist))
np.save("shanten_actions.npy", np.array(action_hist, dtype=np.int32))
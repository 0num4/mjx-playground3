import mjx
from mjx.agents import ShantenAgent
import json
import numpy as np
import tqdm

env = mjx.MjxEnv()
obs_dict = env.reset()  # state
# Agentの入力になるObservation（観測）には、プレイヤーの席順、点数、手配、捨て牌など様々な情報が含まれています。
# https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/observation.py#L19
# Mjxには、観測を機械学習モデルで扱いやすい行列にするメソッドto_featuresが用意されています。
agent = ShantenAgent()

obs_hist = []
action_hist = []

for game in range(1):  # 100半荘回す
    env.reset()  # ゲーム開始
    while not env.done():
        actions = {}  # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/action.py#L14
        for player_id, obs in obs_dict.items():  # 1巡目ということだと思う
            # obsはcppのオブジェクトで中は何も見えない
            legal_actions = obs.legal_actions()  # アクションのリストをobsから取ってくる？
            # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/observation.py#L57
            # print(player_id)
            # print(legal_actions)
            # print(obs.curr_hand().to_json())
            # print(obs.who())
            # print(obs.dealer())
            # print(obs.doras())
            action = agent.act(obs)  # actionもcpp objなので見えないけどagentが取ったアクションだと思う
            actions[player_id] = action

            # 選択できるアクションが複数ある場合、obsとactionを保存する
            if len(legal_actions) > 1:
                obs_hist.append(obs.to_features(feature_name="mjx-small-v0").ravel())  # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/observation.py#L111
                action_hist.append(action.to_idx())
        # print(actions)  # player_0しか入ってない・・・actionは1行動でstep()を刻むことで次の打牌とかツモになるっぽい
        obs_dict = env.step(actions)  # https://github.com/mjx-project/mjx/blob/fcdac0eabf854c2a530168eda989479f41681ef9/mjx/env.py#L25
    env.reset()

# ファイルに書き出し
# np.save("shanten_obs.npy", np.stack(obs_hist))
# np.save("shanten_actions.npy", np.array(action_hist, dtype=np.int32))
print(np.stack(obs_hist).shape)
print(np.array(action_hist, dtype=np.int32).shape)
# (102496, 544)
# (102496,)

print(len(obs_hist))
print(len(action_hist))
# len(obs_hist) == len(action_hist)
# true and 103399
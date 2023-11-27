import mjx
from mjx.agents import ShantenAgent
import json

env = mjx.MjxEnv()
obs_dict = env.reset()  # state
# Agentの入力になるObservation（観測）には、プレイヤーの席順、点数、手配、捨て牌など様々な情報が含まれています。

agent = ShantenAgent()

for game in range(100): # 100半荘回す
    env.reset() # ゲーム開始
    while not env.done():
        for player_id, obs in obs_dict.items():
            legal_actions = obs.legal_actions()
            print(player_id)
            print(legal_actions)

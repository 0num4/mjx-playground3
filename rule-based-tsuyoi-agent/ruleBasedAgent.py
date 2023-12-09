# https://note.com/oshizo/n/nbbcfb4e24908
import mjx
from mjx.agents import RandomAgent
env = mjx.MjxEnv()
agent = RandomAgent()
obs_dict = env.reset()  # game start

rank_hist = []

for game in range(1000):  # 100半荘回す
    env.reset()  # ゲーム開始
    round = 0
    while not env.done():
        actions = {player_id: agent.act(obs) for player_id, obs in obs_dict.items()}
        # print(actions)
        obs_dict = env.step(actions)
    # env.state()
    rank_hist.append(env.rewards())
print(rank_hist)
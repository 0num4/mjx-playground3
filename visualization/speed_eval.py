# https://note.com/oshizo/n/nbbcfb4e24908
import mjx
# from mjx.agents import RandomAgent
env = mjx.MjxEnv()
agent = mjx.agents.RandomAgent()
obs_dict = env.reset()  # game start
# print(obs_dict)
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

# agents = {
#     "player_0": agent,
#     "player_1": agent,
#     "player_2": agent,
#     "player_3": "127.0.0.1:9090",
# }
# run(num_games=10, agent_addresses={
#     "player_0": agent,
#     "player_1": agent,
#     "player_2": agent,
#     "player_3": agent})
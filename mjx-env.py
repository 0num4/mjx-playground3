# https://note.com/oshizo/n/nbbcfb4e24908
import mjx
from mjx.agents import RandomAgent
env = mjx.MjxEnv()
agent = RandomAgent()
obs_dict = env.reset()  # game start

while not env.done():
    actions = {player_id: agent.act(obs) for player_id, obs in obs_dict.items()}
    print(actions)
    obs_dict = env.step(actions)
env.state()
print(env.rewards())
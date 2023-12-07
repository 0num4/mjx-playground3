import gym
from gym import envs

env = gym.make("MountainCar-v0", render=True)
obs = env.reset()
# obsはenvの初期状態
env.render()
print(obs)

# 学習リスト
envids = [spec.id for spec in envs.registry.values()]
print(envids)

# for _ in range(1000):
#     env.render()
#     obs, info = env.step(env.action_space.sample())
# import gym
# from gym import wrappers
# openai gymは引数が毎秒変わるレベルでえぐいし、deprecatedになったのでgymnasiumを使う
# https://gymnasium.farama.org/content/basic_usage/
import gymnasium as gym
env = gym.make('CartPole-v1')

"""
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/learn-openai-gym# python cartpole-v0.py 
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/gym/envs/registration.py:555: UserWarning: WARN: The environment CartPole-v0 is out of date. You should consider upgrading to version `v1`.
"""
# cartpole-v0は非推奨になっている
env=gym.make("CartPole-v1")

# gym.wrapperでrenderで視覚化周りのことができる
recordedEnv = wrappers.record_video.RecordVideo(env, "./")
for episode in range(20):
    obs = recordedEnv.reset()
    for t in range(100):
        recordedEnv.render()
        action = recordedEnv.action_space.sample()
        observation, reward, terminated, truncated, info = recordedEnv.step(action)
        if terminated:
            print("Episode finished after {} timesteps".format(t+1))
            break
        # env.render()

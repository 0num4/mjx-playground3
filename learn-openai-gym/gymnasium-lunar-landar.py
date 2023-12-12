# landarとは着陸するという意味
# https://note.com/kikaben/n/n57584c49d5c2
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    # print(action) # 0, 1, 2, 3が入る。それぞれロケットから火が出る方向みたい。
    # 0: 何もしない 1: 右から火が出る(左に回転する) 2: 真下から火が出る(当然上に行く) 3: 左から火が出る(右に回転する)
    # action = 2
    observation, reward, terminated, truncated, info = env.step(action)
    # obsの値は、船体の位置、速度、角度、角速度、脚の接触情報、旗の位置、速度、角度、角速度、旗と船体の距離、旗と船体の角度、旗と船体の角速度など
    print(reward)
    if truncated:
        print("旗の間に着地出来た")  # 嘘っぽい、このゲームでは仕様上常にfalseらしい。
        observation, info = env.reset()
    if terminated:
        observation, info = env.reset()
        print("(それ以外の場所に)着地した")

env.close()
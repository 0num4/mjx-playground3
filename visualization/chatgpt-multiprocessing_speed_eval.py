import mjx
from mjx.agents import RandomAgent
import multiprocessing
from itertools import repeat


def play_game(_):
    env = mjx.MjxEnv()
    agent = RandomAgent()
    obs_dict = env.reset()

    while not env.done():
        actions = {player_id: agent.act(obs) for player_id, obs in obs_dict.items()}
        obs_dict = env.step(actions)

    return env.rewards()


if __name__ == "__main__":
    pool = multiprocessing.Pool()  # プールを使用してプロセスを管理
    games_to_play = 10000
    results = pool.map(play_game, range(games_to_play))  # 並列処理でゲームを実行
    pool.close()
    pool.join()
    print(results)

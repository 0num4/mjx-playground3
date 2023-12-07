from concurrent.futures import ProcessPoolExecutor
from mjx import Observation, State, Action
import numpy as np
import glob
from tqdm import tqdm
# mjxproto_dirから全ての牌譜を読み込む

batch_size = 3000


def process_file(file):
    obs_hist = []
    action_hist = []

    with open(file) as f:
        lines = f.readlines()

        for line in lines:
            state = State(line)
            for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
                obs = Observation._from_cpp_obj(cpp_obs)
                feature = obs.to_features(feature_name="mjx-small-v0")

                action = Action._from_cpp_obj(cpp_act)
                action_idx = action.to_idx()

                obs_hist.append(feature.ravel())
                action_hist.append(action_idx)

    return np.stack(obs_hist), np.array(action_hist, dtype=np.int32)


# ファイルのリストを取得
files = glob.glob("paifu_mjxproto/*.json")

num_processes = 15  # 使用するプロセス数

# 並列処理
with ProcessPoolExecutor(max_workers=num_processes) as executor:
    results = list(tqdm(executor.map(process_file, files), total=len(files)))
print("end")
# 結果の結合
all_obs = np.concatenate([obs for obs, _ in results], axis=0)
print("all_obs concated")
all_actions = np.concatenate([actions for _, actions in results], axis=0)

print("all_acts concated")
# 結果を保存
np.save("shanten_obs_full_full.npy", all_obs)
print("saved obs")
np.save("shanten_actions_full_full.npy", all_actions)

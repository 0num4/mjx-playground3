# 作ったMLPのagentと、ShantenAgentを対戦させる

import mjx
from mjx.agents import ShantenAgent, RandomAgent
import json
import numpy as np
import tqdm
import torch
import torch.nn as nn
# import nn_samplecode
import rl_gym
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

# env = mjx.MjxEnv()

# random_agent = RandomAgent()
shanten_agent = ShantenAgent()
env = rl_gym.GymEnv(
    opponent_agents=[shanten_agent, shanten_agent, shanten_agent],
    reward_type="game_tenhou_7dan",
    done_type="game",
    feature_type="mjx-small-v0",
)
obs, info = env.reset()  # state


# Agentの入力になるObservation（観測）には、プレイヤーの席順、点数、手配、捨て牌など様々な情報が含まれています。
#  クラスを作る
class MLP(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),  # 入力層
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # 隠れ層
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),  # 出力層
        )
        self.loss_module = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_module(preds, y)
        self.log("train_loss", loss)
        return loss

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer

    def forward(self, x):
        return self.net(x.float())


class ModifiedCNN(pl.LightningModule):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=128):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(32 * (obs_size // 4), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.loss_module = nn.CrossEntropyLoss()

    # def forward(self, x):
    #     print("Input shape:", x.shape)  # 入力データの形状を出力
    #     x = x.unsqueeze(1)  # データを1次元の畳み込み層に適した形状に変更
    #     x = self.conv_net(x)
    #     # x = self.fc_net(x)
    #     print("Output shape after conv_net:", x.shape)  # 畳み込み層の出力形状を出力
    #     return x

    def forward(self, x):
        print("Input shape:", x.shape)
        x = x.unsqueeze(1)
        x = self.conv_net(x)
        x = self.fc_net(x)  # 完全連結層を適用
        print("Output shape after fc_net:", x.shape)
        return x

    def training_step(self, batch, batch_idx):
        # トレーニングステップのロジックをここに記述
        x, y = batch
        logits = self(x)
        loss = self.loss_module(logits, y)
        return loss

    def train_dataloader(self):
        # トレーニングデータローダーの定義
        inps = np.load("./shanten_obs_full_full.npy", mmap_mode="r")
        tgts = np.load("./shanten_actions_full_full.npy", mmap_mode="r")
        print("Input shape:", inps.shape)  # 入力データの形状を出力
        print("Output shape after conv_net:", tgts.shape)
        dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
        return DataLoader(dataset, batch_size=1024, num_workers=15)

    def configure_optimizers(self):
        # オプティマイザーの設定
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


modified_cnn_model = ModifiedCNN()  # クラスを作ってインスタンス化させただけだとまだmodel
# mlp_model.load_state_dict(torch.load("./model_paifu_1_bactch_size1024_10epochs.pth"))
modified_cnn_model.load_state_dict(torch.load("./model_paifu_1_batch_size1024_10epochs_full_enhanced_cnn.pth"))
# mlp_model.parameters()でmodelのすべてのパラメータを取得できる
# torch.optim.Adam: これは、オプティマイザの一種です。オプティマイザは、ニューラルネットワークの学習過程でモデルの重みを更新するアルゴリズムを指します。Adamはその中でも特に人気のあるオプティマイザで、確率的勾配降下法（SGD）の改良版として広く使われています。
# optimizerやパラメータチューニングは学習前に行う、つまり今は学習前
# optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
# mlp_agent = rl_gym.REINFORCE(mlp_model, optimizer)


class ModifiedCNNAgent(mjx.Agent):

    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # 予測
        feature = observation.to_features(feature_name="mjx-small-v0")
        with torch.no_grad():
            action_logit = modified_cnn_model(torch.Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()
        
        # アクション決定
        mask = observation.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)


modified_cnn_agent = ModifiedCNNAgent()
env = rl_gym.GymEnv(
    opponent_agents=[shanten_agent, shanten_agent, shanten_agent],
    reward_type="game_tenhou_7dan",
    done_type="game",
    feature_type="mjx-small-v0",
)
obs, info = env.reset()  # state

modified_cnn_model_me = ModifiedCNN()
modified_cnn_model_me.load_state_dict(torch.load("./model_paifu_1_batch_size1024_10epochs_full_enhanced_cnn.pth"))


obs_hist = []
action_hist = []
rank_hist = []


for game in range(100):  # 100半荘回す
    obs, info = env.reset()  # ゲーム開始
    done = False
    while not done:
        print("Input shape:", obs.shape)
        print(info["action_mask"])
        actions = modified_cnn_agent.act(obs, info["action_mask"])
        obs, r, done, info = env.step(actions)
    rank_hist.append(r)

print(rank_hist)

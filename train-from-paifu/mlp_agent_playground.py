import mjx
from mjx.agents import ShantenAgent, RandomAgent
import json
import numpy as np
import tqdm
import torch
from torch import optim
import torch.nn as nn
# import nn_samplecode
import rl_gym
import pytorch_lightning as pl
from mjx import Agent, Observation, Action

# env = mjx.MjxEnv()


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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        return self.net(x.float())


old_mlp_model = MLP()
old_mlp_model.load_state_dict(torch.load('./model_paifu_1_bactch_size1024_10epochs.pth'))

# loadstatedictだけでよい？optチューニングとかはできない？
# opt = optim.Adam(old_mlp_model.parameters(), lr=1e-3)
# old_mlp_rainforcement_agent = rl_gym.REINFORCE(old_mlp_model, opt)

class MLPAgent(mjx.Agent):

    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # 予測
        feature = observation.to_features(feature_name="mjx-small-v0")
        with torch.no_grad():
            action_logit = old_mlp_model(torch.Tensor(feature.ravel()))
        action_proba = torch.sigmoid(action_logit).numpy()
        
        # アクション決定
        mask = observation.action_mask()
        action_idx = (mask * action_proba).argmax()
        return mjx.Action.select_from(action_idx, legal_actions)
    
# random_agent = RandomAgent()
# shanten_agent = ShantenAgent()
# shanten_mlp_model = MLP()
# shanten_mlp_model.load_state_dict(torch.load("./model_paifu_1_bactch_size1024_10epochs.pth"))
# optimizer = torch.optim.Adam(shanten_mlp_model.parameters(), lr=1e-3)
# shanten_mlp_agent = rl_gym.REINFORCE(shanten_mlp_model, optimizer)


old_mlp_agent = MLPAgent()
env = rl_gym.GymEnv(
    opponent_agents=[old_mlp_agent, old_mlp_agent, old_mlp_agent],
    reward_type="game_tenhou_7dan",
    done_type="game",
    feature_type="mjx-small-v0",
)
obs, info = env.reset()  # state

mlp_model = MLP()  # クラスを作ってインスタンス化させただけだとまだmodel
# mlp_model.load_state_dict(torch.load("./model_paifu_1_bactch_size1024_10epochs.pth"))
mlp_model.load_state_dict(torch.load("./model_paifu_1_batch_size1024_10epochs_noCustomDataloader_full_full.pth"))
# mlp_model.parameters()でmodelのすべてのパラメータを取得できる
# torch.optim.Adam: これは、オプティマイザの一種です。オプティマイザは、ニューラルネットワークの学習過程でモデルの重みを更新するアルゴリズムを指します。Adamはその中でも特に人気のあるオプティマイザで、確率的勾配降下法（SGD）の改良版として広く使われています。
# optimizerやパラメータチューニングは学習前に行う、つまり今は学習前
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
mlp_agent = rl_gym.REINFORCE(mlp_model, optimizer)

obs_hist = []
action_hist = []
rank_hist = []

for game in range(1000):  # 100半荘回す
    obs, info = env.reset()  # ゲーム開始
    done = False
    while not done:
        actions = mlp_agent.act(obs, info["action_mask"])
        obs, r, done, info = env.step(actions)
    rank_hist.append(r)    
    
print(rank_hist)
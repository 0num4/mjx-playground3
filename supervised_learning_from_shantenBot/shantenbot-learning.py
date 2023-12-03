import random
from mjx import Agent, Observation, Action
import torch

class MaltiLayerPerceptronAgent(Agent):
    def __init__(self, model):
        self.model = model

    # mjxのagentつくるにはとりあえずclass実装してact()を実装すればいい
    def act(self, obs: Observation):
        # obsはcppのオブジェクトで中は何も見えない
        legal_actions = obs.legal_actions()  # アクションのリストをobsから取ってくる？
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # 予測
        feature = obs.to_features(feature_name="mjx-small-v0")  # obsをfeature化したものを読み込んでるっぽい
        with torch.no_grad():
            pred = self.model(torch.Tensor(feature).float().unsqueeze(0))
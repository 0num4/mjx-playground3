import json
import sys
import random
import torch
from torch import nn, Tensor
import mjx
from mjx import Agent, Observation, Action
from mjx.agents import ShantenAgent
from gateway import MjxGateway, to_mjai_tile


class MLP(nn.Module):
    def __init__(self, obs_size=544, n_actions=181, hidden_size=544):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())
    

class MLPAgent(Agent):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        try:
            if len(legal_actions) == 1:
                return legal_actions[0]

            # 予測
            feature = observation.to_features(feature_name="mjx-small-v0")
            with torch.no_grad():
                action_logit = self.model(Tensor(feature.ravel()))
            action_proba = torch.sigmoid(action_logit).numpy()

            # アクション決定
            mask = observation.action_mask()
            action_idx = (mask * action_proba).argmax()
            return mjx.Action.select_from(action_idx, legal_actions)
        except:
            return random.choice(legal_actions)


def main():
    model = MLP()
    model.load_state_dict(torch.load('./model_shanten_100.pth'))
    agent = MLPAgent(model)

    player_id = int(sys.argv[1])
    assert player_id in range(4)
    bot = MjxGateway(player_id, agent)

    while True:
        line = sys.stdin.readline().strip()
        resp = bot.react(line)
        sys.stdout.write(resp + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
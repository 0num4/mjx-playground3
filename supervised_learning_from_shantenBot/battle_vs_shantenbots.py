# 作ったMLPのagentと、ShantenAgentを対戦させる

import mjx
from mjx.agents import ShantenAgent
import json
import numpy as np
import tqdm
import torch
import nn_samplecode

env = mjx.MjxEnv()
obs_dict = env.reset()  # state
# Agentの入力になるObservation（観測）には、プレイヤーの席順、点数、手配、捨て牌など様々な情報が含まれています。

random_agent = mjx.agents.RandomAgent()
mlp_agent = nn_samplecode.MLP()

# https://blog.deepblue-ts.co.jp/rule-based/mahjong_rule_7pair/

from mjx.agents import Agent
from mjx import ActionType, Observation, Action
import numpy as np
import random


class ToitsuAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: Observation) -> Action:
        legal_actions = observation.legal_actions()
        # 選びようがないとき、それをする
        if len(legal_actions) == 1:
            return legal_actions[0]

        # あがれるとき、あがる
        win_actions = [a for a in legal_actions if a.type() in [ActionType.TSUMO, ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        # リーチできるとき、リーチする
        riichi_actions = [a for a in legal_actions if a.type() == ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]

        # 鳴きができるとき、パスを選択する
        steal_actions = [
            a for a in legal_actions
            if a.type() in [ActionType.CHI, ActionType.PON, ActionType, ActionType.OPEN_KAN]
        ]
        if len(steal_actions) >= 1:
            pass_action = [a for a in legal_actions if a.type() == ActionType.PASS][0]
            return pass_action

        # 切る
        legal_discards = [
            a for a in legal_actions if a.type() in [ActionType.DISCARD, ActionType.TSUMOGIRI]
        ]

        # 手牌に2枚の牌があるとき、それを切らない
        feature_tehai_2 = observation.to_features("han22-v0")[1]
        feature_tehai_3 = observation.to_features("han22-v0")[2]
        feature_tehai_toitsu = [all(i) for i in zip(*[feature_tehai_2, [not i for i in feature_tehai_3]])]
        feature_tehai_toitsu_true = np.flatnonzero(feature_tehai_toitsu)
        space_discards = [i for i in legal_discards if i.tile().id()//4 not in set(feature_tehai_toitsu_true)]
        space_discards_id = [i.tile().id()//4 for i in space_discards]

        # 手牌に3枚以上の牌があるとき、それを切る
        feature_tehai_morethan3_true = np.flatnonzero(feature_tehai_3)
        space_discards_3or4 = [i for i in space_discards if i.tile().id()//4 in set(feature_tehai_morethan3_true)]
        if len(space_discards_3or4) >= 1:
            return random.choice(space_discards_3or4) 

        # 手牌に1枚の牌があるとき、河に出ている数をカウントして最もたくさん出ている牌を切る
        obs_index = [30+i*10+j for i in range(4) for j in range(4)]
        feature_kawa = np.array(observation.to_features("han22-v0")[obs_index], dtype=int)
        feature_kawa = np.sum(feature_kawa, axis=0)
        kawa_maisu_list = feature_kawa[space_discards_id]
        kawa_maisu_max = np.flatnonzero(kawa_maisu_list == max(kawa_maisu_list))
        space_discards_kawa = [j for i, j in enumerate(legal_discards) if i in set(kawa_maisu_max)]
        return random.choice(space_discards_kawa)
from mjx import Observation, State, Action
with open("2021080221gm-00a9-0000-85adf647&tw=0.json") as f:
    lines = f.readlines()

    for line in lines:
        state = State(line)

        for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
            obs = Observation._from_cpp_obj(cpp_obs)
            feature = obs.to_features(feature_name="mjx-small-v0")

            action = Action._from_cpp_obj(cpp_act)
            action_idx = action.to_idx()
            print(action.type())

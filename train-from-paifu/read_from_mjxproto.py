from mjx import Observation, State, Action 
with open("mjxproto_dir/2022040323gm-00a9-0000-84bff308_tw=0.json") as f:
  lines = f.readlines()
  

  for line in lines:
    state = State(line)

    for cpp_obs, cpp_act in state._cpp_obj.past_decisions():
      obs = Observation._from_cpp_obj(cpp_obs)
      feature = obs.to_features(feature_name="mjx-small-v0")

      action = Action._from_cpp_obj(cpp_act)
      action_idx = action.to_idx()
import numpy as np
import torch

a = [[1,2], [2,3], [3,4]]
# 型が違うとstackしたときにエラーが出る
# a = [[1,2], [2,3], [3]
# (mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/learn-openai-gym# python sample.py 
# [[1, 2], [2, 3], [3]]
# Traceback (most recent call last):
#   File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/learn-openai-gym/sample.py", line 7, in <module>
#     print(np.stack(a))
#   File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/numpy/core/shape_base.py", line 449, in stack
#     raise ValueError('all input arrays must have the same shape')
# ValueError: all input arrays must have the same shape


# (mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/learn-openai-gym# python sample.py 
# [[1, 2], [2, 3], [3, 4]]
# [[1 2]
#  [2 3]
#  [3 4]]

print(a)
print(np.stack(a))
# 普通にnpstackに配列を入れるとNDArrayになって帰ってくる。これをtorchに入れるとTensorになる。
print(np.stack(a).shape)
# shapeは(3, 2)



# # np.save("shanten_obs.npy", np.stack(obs_hist))
# # np.save("shanten_actions.npy", np.array(action_hist, dtype=np.int32))
# print(np.stack(obs_hist).shape)
# print(np.array(action_hist, dtype=np.int32).shape)
# # (102496, 544)
# # (102496,)
# この544は何？

# 1半荘で回しても必ず544だった
# (1549, 544)
# (1549,)
# 1549
# 1549
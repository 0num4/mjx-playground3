Administrator in ~
❯ cd .\work\private\mahjong\mjx-playground\train-from-paifu\

Administrator in mjx-playground\train-from-paifu on  feat/training-from-paifu [!?] via  via 🐍 v3.11.4
❯ ls


    ディレクトリ: C:\Users\Owner\work\private\mahjong\mjx-playground\train-from-paifu


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        2023/12/06     11:00                lightning_logs
d-----        2023/12/04     19:11                mjxproto_dir
d-----        2023/12/05      3:10                paifu_mjxproto
d-----        2023/12/04     20:52                __pycache__
-a----        2023/12/06     11:57           2878 battle_vs_shantenbots.py
-a----        2023/12/06     12:32           3150 battle_vs_shantenbots_oldagent.py
-a----        2023/12/06     12:05            985 count_angles.js
-a----        2023/12/06      1:14             50 cuda_print.py
-a----        2023/12/06     16:08           4229 mlp_agent_playground.py
-a----        2023/12/04     20:43         441220 model_paifu_1.pth
-a----        2023/12/04     21:03         441534 model_paifu_1_bactch_size1024_10epochs.pth
-a----        2023/12/06      1:30         441688 model_paifu_1_batch_size1024_10epochs_full_full.pth
                                                  h
-a----        2023/12/06     10:36         441986 model_paifu_1_batch_size512_2epochs_noCustomDataloader_full_full.pth
-a----        2023/12/03     20:26         441260 model_shanten_100.pth
-a----        2023/12/06     16:16          10187 readme.md
-a----        2023/12/05     23:40           1513 read_from_mjxproto.new.py
-a----        2023/12/05     21:52           2628 read_from_mjxproto.old.py
-a----        2023/12/05     22:12           2914 read_from_mjxproto.py
-a----        2023/12/04     20:51           5737 rl_gym.py
-a----        2023/12/04     20:15           2631 sample.py
-a----        2023/12/04     20:43           2092 shanten_actions.npy
-a----        2023/12/05     22:11        5373272 shanten_actions_batch_2000.npy
-a----        2023/12/04     21:02         328484 shanten_actions_full.npy
-a----        2023/12/04     20:43        1068544 shanten_obs.npy
-a----        2023/12/05     22:07     2922862080 shanten_obs_batch_0.npy
-a----        2023/12/05     22:11     2922990464 shanten_obs_batch_2000.npy
-a----        2023/12/04     21:02      178625792 shanten_obs_full.npy
-a----        2023/12/06      0:23    95395361408 shanten_obs_full_full.npy
-a----        2023/12/06     12:40           2177 supervised_leaning.py
-a----        2023/12/06     16:57           3914 supervised_learning2-cnn copy.py
-a----        2023/12/06     16:24           2673 supervised_learning2-cnn.py
-a----        2023/12/06     16:24           2661 supervised_learning2.py
-a----        2023/12/06     10:04           2266 supervised_learning3.py



Administrator in mjx-playground\train-from-paifu on  feat/training-from-paifu [!?] via  via 🐍 v3.11.4
❯ poetry shell
Spawning shell within C:\Users\Owner\AppData\Local\pypoetry\Cache\virtualenvs\mjx-playground-9Trep71b-py3.10
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

新しいクロスプラットフォームの PowerShell をお試しください https://aka.ms/pscore6

パーソナル プロファイルとシステム プロファイルの読み込みにかかった時間は 549 ミリ秒です。

Administrator in mjx-playground\train-from-paifu on  feat/training-from-paifu [!?] via  via 🐍 v3.10.6 (mjx-playground-py3.10)
❯ exit

Administrator in mjx-playground\train-from-paifu on  feat/training-from-paifu [!?] via  via 🐍 v3.11.4 took 9s
❯ wsl
root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning2-cnn2.py
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2-cnn2.py", line 1, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# poetry shell
Spawning shell within /root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9
root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# . /root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/bin/activate
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# pyt
hon supervised_learning2-cnn2.py
loading data
loaded obs
loaded end
/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2-cnn2.py:98: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
  dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
loaded dataset
loaded dataloaders
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
training start
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2-cnn2.py", line 112, in <module>
    learn()
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2-cnn2.py", line 106, in learn
    trainer.fit(model=model, train_dataloaders=loader)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 544, in fit
    call._call_and_handle_interrupt(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 44, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 580, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 937, in _run
    _verify_loop_configurations(self)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/configuration_validator.py", line 37, in _verify_loop_configurations
    __verify_train_val_loop_configuration(trainer, model)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/configuration_validator.py", line 57, in __verify_train_val_loop_configuration
    raise MisconfigurationException(
lightning_fabric.utilities.exceptions.MisconfigurationException: No `training_step()` method defined. Lightning `Trainer` expects as minimum a `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning2-cnn.py
loading data
loaded obs
loaded end

/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2-cnn.py:68: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
  dataset = TensorDataset(torch.Tensor(inps), torch.LongTensor(tgts))
loaded dataset
loaded dataloaders
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
training start
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2-cnn.py", line 82, in <module>
    learn()
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning2-cnn.py", line 76, in learn
    trainer.fit(model=model, train_dataloaders=loader)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 544, in fit
    call._call_and_handle_interrupt(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 44, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 580, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 937, in _run
    _verify_loop_configurations(self)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/configuration_validator.py", line 37, in _verify_loop_configurations
    __verify_train_val_loop_configuration(trainer, model)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/configuration_validator.py", line 57, in __verify_train_val_loop_configuration
    raise MisconfigurationException(
lightning_fabric.utilities.exceptions.MisconfigurationException: No `training_step()` method defined. Lightning `Trainer` expects as minimum a `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined.
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning3
supervised_learning3-cnn2.py  supervised_learning3.py
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning3
supervised_learning3-cnn2.py  supervised_learning3.py
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning3-cnn2.py
Setting up dataset and dataloader
Initializing model
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
Starting training
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name        | Type             | Params
-------------------------------------------------
0 | conv_net    | Sequential       | 1.6 K
1 | rnn         | LSTM             | 2.3 M
2 | fc_net      | Sequential       | 187 K
3 | loss_module | CrossEntropyLoss | 0
-------------------------------------------------
2.5 M     Trainable params
0         Non-trainable params
2.5 M     Total params
9.935     Total estimated model params size (MB)
Epoch 0:   0%|                                                                                   | 0/81 [00:00<?, ?it/s]/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py:306: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv1d(input, weight, bias, self.stride,
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 120, in <module>
    learn()
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 113, in learn
    trainer.fit(model=model, train_dataloaders=loader)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 544, in fit
    call._call_and_handle_interrupt(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 44, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 580, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 989, in _run
    results = self._run_stage()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1035, in _run_stage
    self.fit_loop.run()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 202, in run
    self.advance()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 359, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 136, in run
    self.advance(data_fetcher)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 240, in advance
    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 187, in run
    self._optimizer_step(batch_idx, closure)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 265, in _optimizer_step
    call._call_lightning_module_hook(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 157, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/core/module.py", line 1282, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/core/optimizer.py", line 151, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/strategies/strategy.py", line 230, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/plugins/precision/precision.py", line 117, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/optimizer.py", line 373, in wrapper
    out = func(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/optimizer.py", line 76, in _use_grad
    ret = func(self, *args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/adam.py", line 143, in step
    loss = closure()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/plugins/precision/precision.py", line 104, in _wrap_closure
    closure_result = closure()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 140, in __call__
    self._result = self.closure(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 126, in closure
    step_output = self._step_fn()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 315, in _training_step
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 309, in _call_strategy_hook
    output = fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/strategies/strategy.py", line 382, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 43, in training_step
    logits = self(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 34, in forward
    x = x.view(batch_size, self.seq_length, -1)
RuntimeError: shape '[1024, 10, -1]' is invalid for input of size 4456448
Epoch 0:   0%|          | 0/81 [00:02<?, ?it/s]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning3-cnn2.py
Setting up dataset and dataloader
Initializing model
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
Starting training
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name        | Type             | Params
-------------------------------------------------
0 | conv_net    | Sequential       | 1.6 K
1 | rnn         | LSTM             | 2.3 M
2 | fc_net      | Sequential       | 187 K
3 | loss_module | CrossEntropyLoss | 0
-------------------------------------------------
2.5 M     Trainable params
0         Non-trainable params
2.5 M     Total params
9.935     Total estimated model params size (MB)
Epoch 0:   0%|                                                                                   | 0/81 [00:00<?, ?it/s]/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py:306: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv1d(input, weight, bias, self.stride,
Conv net output shape: torch.Size([1024, 4352])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 134, in <module>
    learn()
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 127, in learn
    trainer.fit(model=model, train_dataloaders=loader)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 544, in fit
    call._call_and_handle_interrupt(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 44, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 580, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 989, in _run
    results = self._run_stage()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1035, in _run_stage
    self.fit_loop.run()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 202, in run
    self.advance()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 359, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 136, in run
    self.advance(data_fetcher)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 240, in advance
    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 187, in run
    self._optimizer_step(batch_idx, closure)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 265, in _optimizer_step
    call._call_lightning_module_hook(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 157, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/core/module.py", line 1282, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/core/optimizer.py", line 151, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/strategies/strategy.py", line 230, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/plugins/precision/precision.py", line 117, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/optimizer.py", line 373, in wrapper
    out = func(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/optimizer.py", line 76, in _use_grad
    ret = func(self, *args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/adam.py", line 143, in step
    loss = closure()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/plugins/precision/precision.py", line 104, in _wrap_closure
    closure_result = closure()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 140, in __call__
    self._result = self.closure(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 126, in closure
    step_output = self._step_fn()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 315, in _training_step
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 309, in _call_strategy_hook
    output = fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/strategies/strategy.py", line 382, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 57, in training_step
    logits = self(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 48, in forward
    x = x.view(batch_size, self.seq_length, -1)
RuntimeError: shape '[1024, 10, -1]' is invalid for input of size 4456448
Epoch 0:   0%|          | 0/81 [00:01<?, ?it/s]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning3-cnn2.py
Setting up dataset and dataloader
Initializing model
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
Starting training
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name        | Type             | Params
-------------------------------------------------
0 | conv_net    | Sequential       | 1.6 K
1 | rnn         | LSTM             | 289 K
2 | fc_net      | Sequential       | 187 K
3 | loss_module | CrossEntropyLoss | 0
-------------------------------------------------
478 K     Trainable params
0         Non-trainable params
478 K     Total params
1.913     Total estimated model params size (MB)
Epoch 0:   0%|                                                                                   | 0/81 [00:00<?, ?it/s]/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py:306: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv1d(input, weight, bias, self.stride,
Conv net output shape: torch.Size([1024, 4352])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 124, in <module>
    learn()
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 117, in learn
    trainer.fit(model=model, train_dataloaders=loader)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 544, in fit
    call._call_and_handle_interrupt(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 44, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 580, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 989, in _run
    results = self._run_stage()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1035, in _run_stage
    self.fit_loop.run()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 202, in run
    self.advance()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 359, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 136, in run
    self.advance(data_fetcher)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 240, in advance
    batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 187, in run
    self._optimizer_step(batch_idx, closure)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 265, in _optimizer_step
    call._call_lightning_module_hook(
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 157, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/core/module.py", line 1282, in optimizer_step
    optimizer.step(closure=optimizer_closure)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/core/optimizer.py", line 151, in step
    step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/strategies/strategy.py", line 230, in optimizer_step
    return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/plugins/precision/precision.py", line 117, in optimizer_step
    return optimizer.step(closure=closure, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/optimizer.py", line 373, in wrapper
    out = func(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/optimizer.py", line 76, in _use_grad
    ret = func(self, *args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/optim/adam.py", line 143, in step
    loss = closure()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/plugins/precision/precision.py", line 104, in _wrap_closure
    closure_result = closure()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 140, in __call__
    self._result = self.closure(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 126, in closure
    step_output = self._step_fn()
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/automatic.py", line 315, in _training_step
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 309, in _call_strategy_hook
    output = fn(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/strategies/strategy.py", line 382, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 47, in training_step
    logits = self(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/supervised_learning3-cnn2.py", line 38, in forward
    x = x.view(batch_size, self.seq_length, -1)
RuntimeError: shape '[1024, 10, -1]' is invalid for input of size 4456448
Epoch 0:   0%|          | 0/81 [00:01<?, ?it/s]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu#
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python supervised_learning3-cnn2.py
Setting up dataset and dataloader
Initializing model
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:67: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
Starting training
You are using a CUDA device ('NVIDIA GeForce RTX 4090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name        | Type             | Params
-------------------------------------------------
0 | conv_net    | Sequential       | 1.6 K
1 | fc_net      | Sequential       | 580 K
2 | loss_module | CrossEntropyLoss | 0
-------------------------------------------------
582 K     Trainable params
0         Non-trainable params
582 K     Total params
2.329     Total estimated model params size (MB)
Epoch 0:   0%|                                                                                   | 0/81 [00:00<?, ?it/s]/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py:306: UserWarning: Applied workaround for CuDNN issue, install nvrtc.so (Triggered internally at ../aten/src/ATen/native/cudnn/Conv_v8.cpp:80.)
  return F.conv1d(input, weight, bias, self.stride,
Epoch 9: 100%|████████████████████████████████████████████████████████████████| 81/81 [00:00<00:00, 102.56it/s, v_num=5]`Trainer.fit` stopped: `max_epochs=10` reached.
Epoch 9: 100%|█████████████████████████████████████████████████████████████████| 81/81 [00:00<00:00, 97.52it/s, v_num=5]
Training complete, saving model
Model saved
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots
battle_vs_shantenbots.py           battle_vs_shantenbots_cnn.py       battle_vs_shantenbots_oldagent.py
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_
battle_vs_shantenbots_cnn.py       battle_vs_shantenbots_oldagent.py
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 119, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 121, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 78, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Given groups=1, weight of size [16, 1, 3], expected input[1, 544, 1] to have 1 channels, but got 544 channels instead
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 119, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 121, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 78, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Given groups=1, weight of size [16, 1, 3], expected input[1, 544, 1] to have 1 channels, but got 544 channels instead
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
mask tensor([0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1.,
        0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([544])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 121, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 121, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 79, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Given groups=1, weight of size [16, 1, 3], expected input[1, 544, 1] to have 1 channels, but got 544 channels instead
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([1, 1, 16, 34])
mask tensor([0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0.,
        1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([1, 1, 16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 121, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 79, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 1, 16, 34]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([1, 16, 34])
mask tensor([1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([1, 16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 121, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 79, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 16, 34]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([1, 16, 34])
mask tensor([0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([1, 16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 121, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 79, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 16, 34]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([1, 16, 34])
mask tensor([0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0.,
        1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([1, 16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 121, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 79, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 16, 34]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([1, 16, 34])
mask tensor([0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([1, 16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 123, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 79, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 16, 34]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([1, 16, 34])
mask tensor([1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1.,
        0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([1, 16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 131, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 87, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 16, 34]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([1, 16, 34])
mask tensor([0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([1, 16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 131, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 87, in forward
    x = self.conv_net(x)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 310, in forward
    return self._conv_forward(input, self.weight, self.bias)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 306, in _conv_forward
    return F.conv1d(input, weight, bias, self.stride,
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: [1, 1, 16, 34]
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Modified observation shape: torch.Size([16, 34])
mask tensor([0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
        1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0.])
Input shape: torch.Size([16, 34])
Traceback (most recent call last):
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 131, in <module>
    actions = mlp_agent.act(obs, info["action_mask"])
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/rl_gym.py", line 126, in act
    logits = self.model(observation)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 88, in forward
    x = self.fc_net(x)  # 完全連結層を適用
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/container.py", line 215, in forward
    input = module(input)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/.cache/pypoetry/virtualenvs/mjx-playground-S0ozRpda-py3.9/lib/python3.9/site-packages/torch/nn/modules/linear.py", line 114, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x256 and 4352x128)
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu# python battle_vs_shantenbots_cnn.py
Traceback (most recent call last
  File "/mnt/c/Users/Owner/work/private/mahjong/mjx-playground/train-from-paifu/battle_vs_shantenbots_cnn.py", line 168, in <module>
    actions = modified_cnn_agent.act(obs, info["action_mask"])
TypeError: act() takes 2 positional arguments but 3 were given
(mjx-playground-py3.9) root@DESKTOP-2TQ96U5:/mnt/c/Users/Owner/wo


###################################

https://note.nkmk.me/python-pytorch-tensor-dim-size-numel/
import torch

x = torch.tensor(2.0, requires_grad=True)

z = (x + 2) ** 2
z.backward()
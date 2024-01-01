import torch

tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)

torch.cat((tensor1, tensor2), dim=0)

# sample = torch.Tensor([4,5,6])

# tensor([[-0.7515, -0.3427,  1.7045],
#         [-1.0117, -1.5237, -0.6474],
#         [-0.9683,  1.6535, -0.9635],
#         [-0.6191,  0.0717,  0.3665]])
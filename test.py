import torch
import torch_directml

print("Torch version:", torch.__version__)

dml = torch_directml.device()  # DirectML device
x = torch.randn(2, 3).to(dml)
print("Tensor device:", x.device)
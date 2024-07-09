import torch
from torch import nn

from gnr8 import generate


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(100, 36)

    def forward(self, x):
        return self.wte(x)


model = M()

#input_ids = [random.randint(0,10) for _ in range(10)]
input_ids = torch.randint(low=0, high=100, size = (1, 10), device="cpu")
res = generate(model, input_ids)

print(input_ids)
print(res)


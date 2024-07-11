import torch
from torch import nn

from gnr8 import generate

device = "cuda" if torch.cuda.is_available() else "cpu"


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(100, 36)

    def forward(self, x):
        return self.wte(x)


model = M().to(device)

#input_ids = [random.randint(0,10) for _ in range(10)]
input_ids = torch.randint(low=0, high=100, size = (1, 10), device=device)
res = generate(model, input_ids)

print(input_ids)
print(res)


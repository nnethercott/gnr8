import torch
from torch import nn
from transformers import AutoTokenizer

from gnr8 import generate

device = "cuda" if torch.cuda.is_available() else "cpu"


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(36, 36)

    def forward(self, x):
        return self.wte(x)


model = M().to(device)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

#input_ids = [random.randint(0,10) for _ in range(10)]
input_ids = torch.randint(low=0, high=10, size = (1, 10), device=device)

kwargs = {"tokenizer": tokenizer}
res = generate(model, input_ids, kwargs)


print(input_ids)
print(res)


import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from gnr8 import generate
device = "cuda" if torch.cuda.is_available() else "cpu"


class M(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)['logits']
        return x 


tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tinyllama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = M(tinyllama).to(device)

prompt = "tell me a story about a cat"
input_ids = tok(prompt, return_tensors = 'pt')['input_ids'].to(device)

kwargs = {"tokenizer": tok}
#res = generate(model, input_ids, kwargs)



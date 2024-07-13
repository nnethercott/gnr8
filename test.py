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

messages = [{"role": "user", "content": "tell me a story about a cat."}]
prompt = tok.apply_chat_template(messages, tokenizer=False)
input_ids = torch.tensor(prompt).unsqueeze(0).to(device)

kwargs = {"tokenizer": tok}
res = generate(model, input_ids, kwargs)



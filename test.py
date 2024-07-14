import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse 
import gnr8
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
model = model.eval()


parser = argparse.ArgumentParser()
parser.add_argument("--prompt")
args = parser.parse_args()
prompt = args.prompt

messages = [{"role":"user", "content": prompt}]
input_ids = torch.tensor(tok.apply_chat_template(messages)).unsqueeze(0)

kwargs = {"tokenizer": tok}

#ids = model.model.generate(input_ids = input_ids, temperature=1.3, top_k = 32, do_sample = True, min_new_tokens=384, max_new_tokens = 385)

res = gnr8.generate(model, input_ids.to(device), kwargs)

#with torch.no_grad():
#    gnr8.foo(model, input_ids)

print("\x1B[2J\x1B[1;1H")
print(tok.decode(res[0][input_ids.shape[1]:]).strip())

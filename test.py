import argparse
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import gnr8

device = "cuda" if torch.cuda.is_available() else "cpu"


class M(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)["logits"]
        return x


tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tinyllama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = M(tinyllama).to(device)
model = model.eval()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt", "-p", type=str, default="tell me a funny story please."
)
args = parser.parse_args()
prompt = args.prompt

messages = [{"role": "user", "content": prompt}]
input_ids = torch.tensor(tok.apply_chat_template(messages)).unsqueeze(0)

gc = gnr8.GenerationConfig(
    max_new_tokens=256,
    temperature=1.3,
    do_sample=True,
    topk=48,
    stream=False,
    tokenizer=tok,
)


now = time.time()
new_tokens = gnr8.generate(model, input_ids.to(device), gc)
elapsed = time.time()-now
print(f"generated in {elapsed} s")
print(f"tok/s {len(new_tokens)/elapsed}")

print(tok.batch_decode(input_ids))
print(tok.decode(new_tokens))


# if gc.stream:
#     print("\x1B[2J\x1B[1;1H")
#     print(tok.decode(res[0][input_ids.shape[1] :]).strip())

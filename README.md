# gnr8

using [tch-rs](https://github.com/LaurentMazare/tch-rs) for llm generate util in rust


# (intended) usage

import `gnr8` directly in your torch application:

```python
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import gnr8

device = "cuda" if torch.cuda.is_available() else "cpu"


# wrapper for hf model 
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

messages = [{"role":"user", "content": "tell me a funny story."}]
input_ids = torch.tensor(tok.apply_chat_template(messages)).unsqueeze(0)

kwargs = {"tokenizer": tok}
generated = gnr8.generate(model, input_ids.to(device), kwargs)

print(tok.batch_decode(generated))
```

# planned features
- [x] topk sampling
- [x] greedy search
- [x] temperature
- [ ] beam search
- [ ] length penalty
- [ ] streaming

# misc todos 
* solve cuda memory leak when calling model from rust-side 
* split single thread stream into dual-thread with `std::sync::mpsc::channel`
* expose `crate::gen::GenerationConfig` as a pyclass instead of hardcoding in `src/gen.rs` 

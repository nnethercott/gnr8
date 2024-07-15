# gnr8

using [tch-rs](https://github.com/LaurentMazare/tch-rs) for llm generate util in rust

![alt text](https://github.com/nnethercott/gnr8/blob/main/media/demo.gif?raw=true)

# install 
`gnr8` not yet on pypi so right now building is done with [cargo](https://www.rust-lang.org/tools/install). Additionally, make sure you have `libtorch` properly installed on your system at version 2.2.0 and numpy<2. For installing and linking the C++ api for torch needed by tch-rs follow the steps included on the [project's readme](https://github.com/LaurentMazare/tch-rs).

## running with docker 
You can use the pre-built `libgnr8.so` shared object by running the `nnethercott/gnr8` docker image. If you have [nvidia container runtime](https://developer.nvidia.com/container-runtime) installed then the container can access your gpus. 

```bash 
docker run --runtime=nvidia --gpus all -it -v path-to-your-pyscript:/app nnethercott/gnr8:latest /bin/bash
```

With the above steps done you can install the project with:
```bash 
git clone git@github.com:nnethercott/gnr8.git && cd gnr8
cargo build && cp target/debug/libgnr8.so gnr8.so
echo "testing the package..."
python test.py
```

# (intended) usage

import `gnr8` directly in your torch application:

```python
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import gnr8

device = "cuda" if torch.cuda.is_available() else "cpu"

# get any AutoModelForCausalLM from huggingface 
model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(model_id)
tinyllama = AutoModelForCausalLM.from_pretrained(model_id)

# wrapper for hf model as `gnr8.generate` expects model.__call__ to yield logits not dict 
class M(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)['logits']
        return x 

model = M(tinyllama).to(device)
model = model.eval()

messages = [{"role":"user", "content": "tell me a funny story."}]
input_ids = torch.tensor(tok.apply_chat_template(messages)).unsqueeze(0)

generation_config = gnr8.GenerationConfig(
    max_new_tokens = 128, 
    temperature=1.3,
    do_sample=True, 
    topk = 48,
    stream = False,
    tokenizer=tok,
)

generated = gnr8.generate(model, input_ids.to(device), generation_config)
print(tok.batch_decode(generated))
```

# planned features
- [x] topk sampling
- [x] greedy search
- [x] temperature
- [ ] beam search
- [ ] length penalty
- [x] streaming

# misc todos 
* solve cuda memory leak when calling model from rust-side 
* split single thread stream into dual-thread with `std::sync::mpsc::channel`

# gnr8

using [tch-rs](https://github.com/LaurentMazare/tch-rs) for llm generate util in rust

# usage

import `gnr8` directly in your torch application:

```python
from gnr8 import generate
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Embedding(10, 36).to(device)
input_ids = torch.randint(low=0, high=9, size = (1, 16), device=device)

kwargs = {
    'max_new_tokens': 64,
    'do_sample': True,
    'topk': 32,
    'temperature': 1.3,
}

res = generate(model, input_ids, **kwargs)
```

- [x] topk sampling
- [x] greedy search
- [x] temperature
- [ ] beam search
- [ ] length penalty

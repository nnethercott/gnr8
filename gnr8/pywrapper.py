import torch

from gnr8 import gnr8


def generate(model, input_ids, gc):
    new_tokens = []
    with torch.no_grad():
        while len(new_tokens) < gc.max_new_tokens:
            # respect ctx_size
            input_ids = input_ids[
                :, max(0, input_ids.shape[1] - gc.ctx_size) : input_ids.shape[1]
            ]

            # twd in no_grad context manager
            logits = model.forward(input_ids)

            done, next_token = gnr8.generate_token(logits, gc)
            new_tokens.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token), dim=-1)

            if done:
                return new_tokens

        return new_tokens

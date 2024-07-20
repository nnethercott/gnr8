import torch

from gnr8 import gnr8


def generate(model, input_ids, gc):
    new_tokens = []
    with torch.no_grad():
        while len(new_tokens) < gc.max_new_tokens:
            done, input_ids = gnr8.generate_token(model, input_ids, gc)
            next_token = input_ids[:, -1]
            new_tokens.append(next_token.item())

            if done:
                return new_tokens

        return new_tokens

import torch as t
from torch.nn import functional as F
import LayerNorm
from LayerNorm import Config, cfg

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = LayerNorm.reference_gpt2.to_tokens(reference_text).to(LayerNorm.device)
print(tokens.shape)

class Embed(t.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = t.empty((cfg.d_vocab, cfg.d_model))

    def one_hot(self, tokens):
        vectors = []
        for token in tokens.squeeze():
            list = [0]
            list = list * 50257
            list[token] = 1
            vector = t.tensor(list)
            vectors += vector
        tensor = t.tensor(vectors)
        return tensor

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        vocab_size = self.W_E.shape[0]

        one_hot = t.zeros(batch_size, seq_len, vocab_size, device=tokens.device)
        one_hot.scatter_(2, tokens.unsqueeze(-1), 1)

        print("tokens shape before einsum:", one_hot.shape)
        print("self.W_E shape:", self.W_E.shape)

        output = t.einsum("bsv, ve->bse", one_hot, self.W_E)

        print("output shape after einsum:", output.shape)
        return output

       
       
        # tokens = self.one_hot(tokens)
        # print("self.W_E shape:", self.W_E.shape)
        # output = t.einsum("ji,jk->ik", self.W_E, tokens.T)
        # print("tokens shape after einsum:", tokens.shape)
        # return output.T

# [seq, dim]
embed = Embed(cfg)

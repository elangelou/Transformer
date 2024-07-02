import torch as t 
import torch.nn as nn
import LayerNorm as LayerNorm


class PosEmbed(nn.Module):
    def __init__(self, cfg: LayerNorm.Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        batch, seq_len = tokens.shape
        output = t.einsum('n d -> n d', self.W_pos[:seq_len]).expand(batch, -1, -1)
        return output.squeeze()
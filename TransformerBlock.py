import torch.nn as nn
import LayerNorm
import attention
import MLP

class TransformerBlock(nn.Module):
    def __init__(self, cfg: LayerNorm.Config):
        super().__init__()
        self.cfg = cfg 
        self.ln1 = LayerNorm.LayerNorm(cfg)
        self.attn = attention.Attention(cfg)
        self.ln2 = LayerNorm.LayerNorm(cfg)
        self.mlp = MLP.MLP(cfg)

    def forward(self, resid_pre): 
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post

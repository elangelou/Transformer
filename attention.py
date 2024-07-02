import torch as t
import torch.nn as nn
import einops
import LayerNorm as LayerNorm

class Attention(nn.Module):
        
    def __init__(self, cfg: LayerNorm.Config):
                super().__init__()
                self.cfg = cfg
                self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
                self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
                self.W_V = nn.Parameter(t.empty(((cfg.n_heads, cfg.d_model, cfg.d_head))))
                self.W_O = nn.Parameter(t.empty(((cfg.n_heads, cfg.d_head, cfg.d_model))))
                self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
                self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
                self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
                self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
                nn.init.normal_(self.W_Q, std=self.cfg.init_range)
                nn.init.normal_(self.W_K, std=self.cfg.init_range)
                nn.init.normal_(self.W_V, std=self.cfg.init_range)
                nn.init.normal_(self.W_O, std=self.cfg.init_range)
                self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=LayerNorm.device))

    def forward(self, normalized_resid_pre):
        print("normalized_resid_pre shape:", normalized_resid_pre.shape)
        print("W_Q shape:", self.W_Q.shape)
        print("W_K shape:", self.W_K.shape)
        print("W_V shape:", self.W_V.shape)
        print("W_O shape:", self.W_O.shape)
        
        q = einops.einsum(normalized_resid_pre, self.W_Q, "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head") + self.b_Q
        k = einops.einsum(normalized_resid_pre, self.W_K, "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head") + self.b_K
        v = einops.einsum(normalized_resid_pre, self.W_V, "batch seq_len d_model, n_heads d_model d_head -> batch seq_len n_heads d_head") + self.b_V

        attn_scores = einops.einsum(q, k, "batch seq_len n_heads d_head, batch seq_len2 n_heads d_head -> batch n_heads seq_len seq_len2")
        
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head ** 0.5)
        attn_pattern = attn_scores_masked.softmax(dim=-1)
       
    
        z = einops.einsum(v, attn_pattern, 'batch posn_K nheads d_head , batch nheads posn_Q posn_K -> batch posn_Q nheads d_head')

        attn_out = einops.einsum(z, self.W_O, 'batch posn_Q nheads d_head, heads d_head d_model -> batch posn_Q d_model') + self.b_O
        return attn_out

    def apply_causal_mask(self, attn_scores):
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores
        
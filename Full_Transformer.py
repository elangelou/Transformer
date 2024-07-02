import torch as t
import torch.nn as nn
import math
import einops
from fancy_einsum import einsum
from tqdm.notebook import tqdm
from LayerNorm import LayerNorm, device, reference_gpt2
from Unembedding import Unembed
from TransformerBlock import TransformerBlock
from LayerNorm import Config, cfg
from embedding import Embed, tokens
from positionalembeding import PosEmbed

class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens):
        print(f"Input tokens shape: {tokens.shape}")
        
        embed = self.embed(tokens)
        print(f"Embedded shape: {embed.shape}")

        pos_embed = self.pos_embed(tokens)
        print(f"Positional embedding shape: {pos_embed.shape}")

        residual = embed + pos_embed
        print(f"Residual shape after addition: {residual.shape}")
        
        for block in self.blocks:
            residual = block(residual)
        normalized_resid_final = self.ln_final(residual)
        logits = self.unembed(self.ln_final(residual))
        print(f"Output logits shape: {logits.shape}")
        
        return logits
    
transformer = DemoTransformer(cfg)

demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)

def get_log_probs(logits, tokens):
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probs_for_tokens

pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

test_string = '''The Total Perspective Vortex derives its picture'''
for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    print(test_tokens.shape)

    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)

demo_gpt2 = DemoTransformer(Config(debug=False))
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

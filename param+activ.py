from dataclasses import dataclass
import transformer_modules.LayerNorm as LayerNorm
import DemoTransformer

model_cfg = LayerNorm.Config(
    debug=False, 
    d_model=256, 
    n_heads=4, 
    d_head=64, 
    d_mlp=1024, 
    n_layers=2, 
    n_ctx=256, 
    d_vocab=LayerNorm.reference_gpt2.cfg.d_vocab
)
model = DemoTransformer(model_cfg)


batch = 1
position = 35
d_model = 768
n_heads = 12
n_layers = 12
d_mlp = 3072 
d_head = 64

for activation_name, activation in LayerNorm.cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")

for name, param in LayerNorm.reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

print(LayerNorm.reference_gpt2.cfg)

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()
print(cfg)
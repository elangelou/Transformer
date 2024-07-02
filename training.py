import torch
import torch as nn
import Full_Transformer
import wandb
import numpy as np
import datasets
import math
from torch import Tensor
from typing import Dict
import tqdm.auto as tqdm
from Full_Transformer import Config, reference_gpt2
from Full_Transformer import DemoTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader
from LayerNorm import cfg

##Config 

batch_size = 8

model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)

## Create Data

dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
print(dataset)
print(dataset[0]['text'][:100])
tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", num_proc=4 )
data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

## Create Model

model = DemoTransformer(model_cfg)
model.cuda()

## Create Optimizer 

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

## Run training loop

loses = []
print("number of batches:", len(data_loader))
for epoch in range(num_epochs):
    for c, batch in tqdm.tqdm(enumerate(data_loader)):
        tokens = batch['tokens'].cuda()
        logits = model(tokens)
        loss = lm_cross_entropy_loss(logits, tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if c % log_every == 0:
            print(f"Step: {c}, Loss: {loss.item():.4f}")
        if c > max_steps:
            break
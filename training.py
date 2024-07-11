import torch
import torch as nn
import torch.nn.functional as F
import Full_Transformer
import wandb
import numpy as np
import datasets
import math
from torch import Tensor
from typing import Dict
import tqdm.auto as tqdm
from Full_Transformer import Config, reference_gpt2, DemoTransformer, lm_cross_entropy_loss
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import DataLoader
from LayerNorm import cfg

##Config 

batch_size = 8
num_epochs = 200
lr = 1e-4
weight_decay = 1e-2
log_every = 100
max_steps = 100
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

losses = []
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

def generate_text(model, prompt, max_length=100, temp=0.7, top_k=50):
    try:
        model.eval()

        input_ids = reference_gpt2.tokenizer.encode(prompt, return_tensors="pt").to('cuda')

        generated = input_ids[0].tolist()

        for i in range(max_length):
            inputs = torch.tensor([generated]).to('cuda')

            with torch.no_grad():
                outputs = model(inputs)

            next_token_logits = outputs[0, -1, :] / temp
        
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
        
            next_token_index = torch.multinomial(probs, num_samples=1).item()
            next_token = top_k_indices[next_token_index].item()
        
            generated.append(next_token)

            if next_token == reference_gpt2.tokenizer.eos_token_id:
                break
        
            if i % 10 == 0:
                print(f"Generated token {i} tokens")
    
        return reference_gpt2.tokenizer.decode(generated, skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

model.eval()
prompt = "Once upon a time"
generated_text = generate_text(model, prompt, temp=0.7, top_k=50, max_length=100)
print("Generated text:")
print(generated_text)
        
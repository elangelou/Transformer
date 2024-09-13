# Transformer
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Implementing a Transformer model (GPT-2) from scratch.

## Layers

- **Attention**: Core attention mechanism
- **Embedding**: Input embedding layer
- **Full Transformer**: Complete transformer architecture
- **Layer Normalization**: Normalization technique
- **MLP**: Feed-forward network component
- **Positional Embedding**: Position information encoding
- **Transformer Block**: Individual transformer layer
- **Unembedding**: Output layer for token prediction

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/transformer-implementation.git
   cd transformer-implementation
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```

## Usage

This project provides both Python scripts and Jupyter notebooks for each component, allowing for flexible use and experimentation.

To run a Python script:
```
python script_name.py
```

To run a Jupyter notebook:
```
jupyter notebook
```
Then navigate to the desired `.ipynb` file.

## Training

Use `training.py` or `training.ipynb` to train your transformer model. The `transformer_arena.py` and `transformer_arena.ipynb` files provide an environment for testing and evaluating your implementation.

## Acknowledgements

This implementation is based on instructions found in the ARENA curriculum: https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch
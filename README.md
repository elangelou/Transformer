# Transformer
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

This project contains an implementation of a Transformer model (GPT-2) from scratch.

## Structure

- `attention.py` & `attention.ipynb`: Attention mechanism implementation
- `embedding.py` & `embedding.ipynb`: Embedding layer implementation
- `Full_Transformer.py` & `Full_Transformer.ipynb`: Complete transformer model
- `LayerNorm.py` & `LayerNorm.ipynb`: Layer normalization implementation
- `MLP.py` & `MLP.ipynb`: Multi-layer perceptron implementation
- `param+activ.py` & `param+activ.ipynb`: Parameter and activation functions
- `positionalembeding.py` & `positionalembeding.ipynb`: Positional embedding implementation
- `setup_arena.py` & `setup_arena.ipynb`: Setup for training environment
- `training.py` & `training.ipynb`: Training routines
- `transformer_arena.py` & `transformer_arena.ipynb`: Transformer testing environment
- `TransformerBlock.py` & `TransformerBlock.ipynb`: Individual transformer block implementation
- `Unembedding.py` & `Unembedding.ipynb`: Unembedding layer implementation
- `setup.py`: Project setup script
- `LICENSE`: License information for the project

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

## Layers

- **Attention**: Core attention mechanism
- **Embedding**: Input embedding layer
- **Full Transformer**: Complete transformer architecture
- **Layer Normalization**: Normalization technique
- **MLP**: Feed-forward network component
- **Positional Embedding**: Position information encoding
- **Transformer Block**: Individual transformer layer
- **Unembedding**: Output layer for token prediction

## Training

Use `training.py` or `training.ipynb` to train your transformer model. The `transformer_arena.py` and `transformer_arena.ipynb` files provide an environment for testing and evaluating your implementation.

## Acknowledgements

This project was based on instructions found in the ARENA curriculum: https://arena3-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch
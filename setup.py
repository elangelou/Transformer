from setuptools import setup, find_packages

setup(
    name='Transformer',
    author='elangelou',
    version='0.0.1',
    packages=find_packages(),
    install_requires=["einops", "requests", "pytest", "ipykernel", "notebook", "datasets", "ipywidgets==7.7.1", "jupyterlab-widgets==1.1.1", "jupyter", "matplotlib", "numpy-stl", "mediapy", "pandas", "scikit-learn", "wandb==0.13.10", "ftfy", "tensorboard", "pygame", "plotly", "sympy", "huggingface_hub", "accelerate", "pre-commit", "diffusers", "gdown", "jaxtyping", "tiktoken", "typeguard", "torchinfo", "torchvision", "frozendict", "openai==0.28", "transformer_lens", "gym==0.23.1", "pygame", "autorom[accept-rom-license]", "ale-py", "protobuf==3.20.3", "mujoco", "imageio-ffmpeg", "streamlit", "streamlit-antd-components==0.2.5", "streamlit-image-select", "streamlit-on-Hover-tabs", "nnsight", "opencv-python", "torchtext", "portalocker>=2.0.0", "git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python", "git+https://github.com/callummcdougall/eindex.git", "git+https://github.com/neelnanda-io/neel-plotly"]
)
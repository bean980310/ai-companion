#!/bin/zsh
# Make sure you have uv installed and activated virtual environment before running this script.

echo "Installing requirements..."
uv pip install -r requirements/common.txt
uv pip install jupyter jupyterlab jupyterlab_widgets ipython ipykernel ipywidgets

echo "Installing pytorch..."
uv pip install torch torchvision torchaudio

echo "Installing tensorflow..."
uv pip install tensorflow tensorboard keras tf-keras tensorflow-metal

echo "Installing requirements..."
uv pip install -r requirements/macos_arm64/requirements.txt --no-build-isolation

echo "Installing requirements..."
uv pip install "transformers[audio]"

echo "Installing requirements..."
uv pip install -r requirements/macos_arm64/requirements_ai_models.txt --no-build-isolation

echo "Installing requirements..."
uv pip install outetts
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal --upgrade
uv pip install protobuf transformers --upgrade
uv pip install xai-sdk
uv pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
# uv pip install mlx_image
uv pip install fsspec numpy psutil pandas --upgrade

echo "Checking python package"
uv pip check
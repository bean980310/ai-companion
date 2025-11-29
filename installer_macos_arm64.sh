#!/bin/zsh
# Make sure you have uv installed and activated virtual environment before running this script.

echo "Installing requirements..."
uv pip install jupyter jupyterlab jupyterlab_widgets ipython ipykernel ipywidgets

echo "Installing pytorch..."
uv pip install torch torchvision torchaudio

echo "Installing requirements..."
uv pip install -r requirements/macos_arm64/requirements.txt --no-build-isolation

echo "Installing requirements..."
uv pip install "transformers[audio]"

echo "Installing LLM Backend for AI Companion..."
uv pip install ai-companion-llm-backend

echo "Installing requirements..."
uv pip install outetts
uv pip install https://github.com/bean980310/llama-cpp-python/releases/download/v0.3.17-metal/llama_cpp_python-0.3.17-cp312-cp312-macosx_26_0_arm64.whl
uv pip install protobuf transformers --upgrade
uv pip install xai-sdk
uv pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
# uv pip install mlx_image
uv pip install fsspec numpy psutil pandas --upgrade

echo "Checking python package"
uv pip check
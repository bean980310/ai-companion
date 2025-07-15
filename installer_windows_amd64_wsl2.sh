#!/bin/bash
# Make sure you have uv installed and activated virtual environment before running this script.

echo "Installing requirements..."
uv pip install -r requirements/common.txt
uv pip install jupyter jupyterlab jupyterlab_widgets ipython ipykernel ipywidgets

echo "Installing pytorch..."
uv pip install torch torchvision torchaudio

echo "Installing requirements..."
uv pip install -r requirements/windows_amd64_wsl2.txt --no-build-isolation

echo "Installing requirements..."
uv pip install "transformers[audio]"

echo "Installing requirements..."
uv pip install -r requirements/ai_models.txt --no-build-isolation

echo "Installing requirements..."
uv pip install outetts
uv pip install llama-cpp-pytho protobuf==5.29.5 transformers --upgrade
uv pip install xai-sdk
uv pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
uv pip install packaging==24.2 fsspec==2025.3.0 numpy==2.1.3

echo "Checking python package"
uv pip check
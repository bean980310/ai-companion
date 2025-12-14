#!/bin/bash
# Make sure you have uv installed and activated virtual environment before running this script.

echo "Installing requirements..."
uv pip install jupyter jupyterlab jupyterlab_widgets ipython ipykernel ipywidgets

echo "Installing pytorch..."
uv pip install torch torchvision torchaudio

echo "Installing requirements..."
uv pip install -r requirements/linux_amd64_cuda_cu128/requirements.txt --no-build-isolation

echo "Installing requirements..."
uv pip install "transformers[audio]"

echo "Installing LLM Backend for AI Companion..."
uv pip install "ai-companion-llm-backend @ https://github.com/bean980310/ai-companion-llm-backend/releases/download/v0.1.0/ai_companion_llm_backend-0.1.0-py3-none-any.whl"

echo "Installing requirements..."
uv pip install outetts
uv pip install https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.17-cu128-AVX2-linux-20251120/llama_cpp_python-0.3.17-cp312-cp312-linux_x86_64.whl
uv pip install protobuf transformers --upgrade
uv pip install xai-sdk
uv pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
uv pip install fsspec numpy psutil pandas --upgrade

echo "Checking python package"
uv pip check
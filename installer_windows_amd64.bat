@echo off

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/common.txt
%HomePath%\.local\bin\uv.exe pip install jupyter jupyterlab jupyterlab_widgets ipython ipykernel ipywidgets

echo "Installing pytorch..."
%HomePath%\.local\bin\uv.exe pip install torch torchvision torchaudio

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/windows_amd64_native_cu128/requirements.txt --no-build-isolation

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install "transformers[audio]"

echo "Installing LLM Backend for AI Companion..."
%HomePath%\.local\bin\uv.exe pip install "ai-companion-llm-backend @ https://github.com/bean980310/ai-companion-llm-backend/releases/download/v0.1.0/ai_companion_llm_backend-0.1.0-py3-none-any.whl"

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install outetts
%HomePath%\.local\bin\uv.exe pip install https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.17-cu128-AVX2-win-20251121/llama_cpp_python-0.3.17-cp312-cp312-win_amd64.whl
%HomePath%\.local\bin\uv.exe pip install protobuf transformers --upgrade
%HomePath%\.local\bin\uv.exe pip install xai-sdk
%HomePath%\.local\bin\uv.exe pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
%HomePath%\.local\bin\uv.exe pip install fsspec numpy psutil pandas --upgrade

echo "Checking python package"
%HomePath%\.local\bin\uv.exe pip check

pause
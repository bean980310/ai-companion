@echo off

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/common.txt
%HomePath%\.local\bin\uv.exe pip install jupyter jupyterlab jupyterlab_widgets ipython ipykernel ipywidgets

echo "Installing pytorch..."
%HomePath%\.local\bin\uv.exe pip install torch torchvision torchaudio

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/windows_amd64_native/requirements.txt --no-build-isolation

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install "transformers[audio]"

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/windows_amd64_native/requirements_ai_models.txt --no-build-isolation

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install outetts
%HomePath%\.local\bin\uv.exe pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --upgrade
%HomePath%\.local\bin\uv.exe pip install protobuf==5.29.5 transformers --upgrade
%HomePath%\.local\bin\uv.exe pip install xai-sdk
%HomePath%\.local\bin\uv.exe pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
%HomePath%\.local\bin\uv.exe pip install fsspec==2025.3.0 numpy==2.1.3 psutil==7.0.0

echo "Checking python package"
%HomePath%\.local\bin\uv.exe pip check

pause
# Make sure you have uv installed and activated virtual environment before running this script.

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/common.txt
%HomePath%\.local\bin\uv.exe pip install jupyter jupyterlab jupyterlab_widgets ipython ipykernel ipywidgets

echo "Installing pytorch..."
%HomePath%\.local\bin\uv.exe pip install torch torchvision torchaudio

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/windows_amd64_native_cu128/requirements.txt --no-build-isolation

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install "transformers[audio]"

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/windows_amd64_native_cu128/requirements_ai_models.txt --no-build-isolation

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install outetts
%HomePath%\.local\bin\uv.exe pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --upgrade
%HomePath%\.local\bin\uv.exe pip install protobuf transformers --upgrade
%HomePath%\.local\bin\uv.exe pip install xai-sdk
%HomePath%\.local\bin\uv.exe pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
%HomePath%\.local\bin\uv.exe pip install fsspec numpy psutil pandas --upgrade

echo "Checking python package"
%HomePath%\.local\bin\uv.exe pip check

pause
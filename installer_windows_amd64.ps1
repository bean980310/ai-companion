# Make sure you have uv installed and activated virtual environment before running this script.

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/common.txt

echo "Installing pytorch..."
%HomePath%\.local\bin\uv.exe pip install torch torchvision torchaudio

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/windows_amd64.txt --no-build-isolation

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install "transformers[audio]"

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install -r requirements/ai_models.txt --no-build-isolation

echo "Installing requirements..."
%HomePath%\.local\bin\uv.exe pip install outetts
%HomePath%\.local\bin\uv.exe pip install llama-cpp-python protobuf==5.29.5 transformers --upgrade
%HomePath%\.local\bin\uv.exe pip install xai-sdk
%HomePath%\.local\bin\uv.exe pip install "langchain-chroma>=0.1.2" "langchain-neo4j>=0.4.0"
%HomePath%\.local\bin\uv.exe pip install packaging==24.2 fsspec==2025.3.0 numpy==2.1.3

echo "Checking python package"
%HomePath%\.local\bin\uv.exe pip check

pause
# 인공지능 컴패니언 프로젝트

## 프로젝트 소개

생성형 AI 모델들을 활용한 인공지능 컴패니언 서비스 개발

챗봇 서비스와 AI를 활용한 그림, 비디오, 오디오 생성 서비스를 제공하며 유저 페르소나와 캐릭터 페르소나 시스템을 활용한 AI 캐릭터 기능을 통한 페르소나 챗봇 기능을 도입하여 AI를 단순 도구를 넘어서서 친구 및 파트너로서 교류 및 협력하여 대화를 넘어선 여러가지 작업 및 놀이를 제공하는 서비스

유저 페르소나는 주로 일반 사용자를 대상으로 삼으며, 그림, 비디오, 음악, 오디오 생성 서비스는 프로 사용자 역시 대상에 포함.

## 주요 기능
### Chatbot
LLM을 활용하여 AI와 대화.

**지원 모델**
* **API**

|모델 제공|모델명|
|-------|----|
|OpenAI|gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o-mini, gpt-4o|
|Anthropic|claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-latest, claude-3-5-sonnet-latest, claude-3-5-haiku-latest, claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.5-pro, gemini-2.0-flash|

* **Local**: Transformers, GGUF, MLX(Apple Silicon Mac 한정)
Local Model의 경우 아래의 모델에 대해 Download Center에서 모델 다운로드를 사전 제공.
|모델 제공|모델명|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B, meta-llama/Llama-3.1-8B-Instruct, meta-llama/Llama-3.2-11B-Vision, meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b, google/gemma-2-9b-it, google/gemma-3-12b-pt, google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-14B, Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3, mistralai/Mistral-7B-Instruct-v0.3, mistralai/Mistral-Small-24B-Base-2501, mistralai/Mistral-Small-24B-Instruct-2501|

### Image Generation
Stable Diffusion, Flux등 이미지 생성 모델을 활용한 이미지 생성. ComfyUI를 백엔드 서버로 활용하여 이미지를 생성함.

### Storyteller
LLM을 활용하여 텍스트를 생성. Chatbot과는 달리 소설 등의 글작성에 UI가 최적화됨.

### Video Generation
Coming Soon

### Audio Generation
Coming Soon

### Translator
번역 모델을 활용한 다국어 번역. 이미지 파일 또는 pdf 파일을 업로드 하여 텍스트 추출후 해당 텍스트를 번역에 활용하는것도 가능.

## 자신의 컴퓨터 환경에 맞는 requirements를 설치하기

- **가상 환경 설정**

```shell
# conda (권장!)
# Python 3.10
conda create -n ai-companion python=3.10
# Python 3.11
conda create -n ai-companion python=3.11
# Python 3.12
conda create -n ai-companion python=3.12
conda activate ai-companion
# venv
python3 -m venv venv
# MacOS, Linux 환경
source venv/bin/activate 
# Windows 환경
source venv/Scripts/activate 
```

- **Windows 환경**
```shell
pip install -r requirements_windows_amd64.txt
```

- **macOS 환경(Apple Silicon을 탑재한 Mac)**
```zsh
pip install -r requirements_macos_arm64.txt
```
(Intel Mac은 requirements.txt로 설치!)

- **Linux 환경**
```bash
pip install -r requirements_linux.txt
```

### 맥에서 xformers 설치시 주의사항!

```zsh
brew update
brew install gcc cmake llvm
```

```zsh
export PATH="/opt/homebrew/opt/llvm/bin:/$PATH"
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"
```

```zsh
pip install xformers
```

## 실행

```zsh
python app.py
```

### Intel Mac 지원에 관하여

Intel Mac에서는 정상적인 동작을 보장하지 않으며, 대부분의 기능을 지원하지 않습니다.
따라서 최종 버전 배포 시점 및 그 이전에 Intel Mac에 대한 지원이 제거될 수 있습니다.
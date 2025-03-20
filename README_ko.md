# 로컬 머신을 위한 AI 컴패니언
로컬 환경에서 실행되는 생성형 AI 모델(LLM, Diffusion 등)을 활용한 Gradio 기반 인공지능 컴패니언 앱입니다.

[English](README.md) [한국어](README_ko.md) [日本語](README_ja.md)

## 개요

챗봇을 비롯해 그림, 비디오, 오디오 등 다양한 콘텐츠 생성 서비스를 제공합니다. 특히 유저 페르소나와 캐릭터 페르소나 시스템을 도입하여 AI를 단순한 도구가 아닌 친구나 파트너로서 상호작용하며, 일상적인 대화부터 협업과 여가 활동까지 다양한 활동을 함께할 수 있는 서비스입니다.

유저 페르소나는 주로 일반 사용자를 대상으로 하며, 그림·비디오·음악·오디오 생성 서비스는 전문가 사용자도 포함됩니다.

## 주요 기능
### 챗봇(Chatbot)
LLM을 활용해 AI와 자연스럽게 대화할 수 있습니다.

**지원 모델**
* **API**

|제공사|모델명|
|-------|----|
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.5-preview|
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **Local**: Transformers, GGUF, MLX(Apple Silicon 탑재 Mac 한정)<br>Transformers 모델은 다운로드 센터에서 미리 제공됩니다.

|제공사|모델명|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**캐릭터 페르소나 설정**
* **시스템 프롬프트(시스템 메시지)**: 챗봇의 역할과 행동을 정의하는 기본적인 지시사항입니다. (기본값: “당신은 유용한 AI 비서입니다.”) 캐릭터 또는 프리셋 변경 시 자동으로 업데이트됩니다.
* **캐릭터**: 원하는 AI 대화 상대를 선택합니다.
* **프리셋**: 사용자가 직접 정의한 시스템 프롬프트를 적용합니다. 캐릭터 변경 시 대응하는 프리셋으로 자동 전환됩니다.

**하이퍼파라미터 설정**<br>하이퍼파라미터에 익숙하지 않다면 기본값 사용을 권장합니다.

* **시드 값(Seed)**: 난수 생성 초기값 (기본값: 42)
* **온도(temperature)**: 답변의 창의성 및 랜덤성 조절 (기본값: 0.6)
* **Top K**: 생성 시 고려할 후보 단어의 최대 개수 설정 (기본값: 20)
* **Top P**: 누적 확률 기반으로 상위 토큰을 샘플링하여 출력 (기본값: 0.9)
* **반복 패널티(repetition penalty)**: 단어 반복 빈도 조절 (기본값: 1.1)

### 이미지 생성(Image Generation)
Stable Diffusion, Flux 등의 이미지 생성 모델을 사용하여 이미지를 생성합니다. ComfyUI를 백엔드 서버로 사용합니다.

**지원 모델**
* **API**
현재는 제한적인 모델만 지원됩니다.

|제공사|모델명|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **Local**: Diffusers, Checkpoints
 - **Diffusers**: 현재 모델 선택 및 스캔은 가능하지만 실제 이미지 생성은 지원하지 않습니다. (추후 지원 예정)
 - **Checkpoints**: ComfyUI를 통해 이미지 생성 가능. 모델 파일은 models/checkpoints에 위치해야 합니다.

지원되는 베이스 모델:
- Stable Diffusion 1.5
- Stable Diffusion 2.x
- Stable Diffusion XL 1.0
- Stable Diffusion 3 Medium
- FLUX.1 Schnell
- FLUX.1 Dev
- Stable Diffusion 3.5 Large
- Stable Diffusion 3.5 Medium
- Illustrious XL 1.0

**세부 옵션**

* **LoRA**: 베이스 모델당 최대 10개까지 적용 가능.
* **VAE**: 기본적으로 내장된 VAE를 사용하며 사용자 지정도 가능합니다.
* **Embedding**: embedding:name 형태로 입력하여 사용합니다.
* **ControlNet**: 인터페이스는 현재 미구현(추후 지원 예정).
* **Refiner**: SDXL 1.0 리파이너의 시작 단계를 설정할 수 있습니다.

**이미지 생성 옵션**

* **Positive Prompt**: 생성하려는 이미지를 설명하는 단어를 입력.
* **Negative Prompt**: 제외하고 싶은 특징을 입력.
* **Width, Height**: 이미지 크기 지정.
* **Recommended Resolution**

|Base Model|Recommended Resolution|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|
|Illustrious XL 1.0|1536x1536 <br> 1248x1824 <br> 1824x1248|

* **generation Steps**: 높을수록 고품질이지만 처리 시간이 늘어납니다.
* **Denoise Strength**: 노이즈 강도 설정.

**고급 설정(Advanced Settings)**

* **Sampler**: 샘플링 방식 선택
* **Scheduler**: 같은 샘플러와 프롬프트에서도 결과가 달라질 수 있습니다.
* **CFG Scale**: 값이 클수록 프롬프트를 정확히 따르고, 작을수록 창의적인 결과를 생성합니다.
* **Seed**: 난수 초기값 설정
* **Clip Skip**: 이미지 생성 과정 일부를 생략하는 기능
* **Batch Size**: 한번에 생성할 이미지 개수 지정

**Image to Image**<br>기존 이미지 변경 및 Inpaint 기능 지원.

### Storyteller
LLM을 활용한 글쓰기 전용 UI 제공(현재 개발 중).

**지원 모델**<br>Chatbot 문단내 지원 모델 참조.

**하이퍼파라미터 설정**<br>Chatbot 문단내 하이퍼파라미터 설정 참조.<br>하이퍼파라미터의 작동원리에 대해 잘 알지 못할경우 기본값으로 두는걸 권장합니다.

### 비디오 생성(Video Generation)
곧 지원 예정입니다.

### 오디오 생성(Audio Generation)
곧 지원 예정입니다.

### Translator
번역 모델을 활용한 다국어 번역 서비스. 이미지나 PDF 파일에서 텍스트를 추출하여 번역할 수도 있습니다.

## 설치 방법

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

**의존성 설치**

- **Windows 환경**
```shell
pip install -r requirements_windows_amd64.txt
```

- **macOS 환경(Apple Silicon을 탑재한 Mac)**
```zsh
pip install -r requirements_macos_arm64.txt
```
(Intel Mac 사용자는 requirements.txt 사용!)

- **Linux 환경**
```bash
pip install -r requirements_linux.txt
```

### 맥에서 xformers 설치 시 주의사항

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

## 실행 방법

```zsh
python app.py
```

### Intel Mac 지원 안내

Intel Mac에서는 일부 기능이 정상적으로 동작하지 않을 수 있으며, 추후 지원이 중단될 수 있습니다.
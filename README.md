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
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.5-preview|
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **Local**: Transformers, GGUF, MLX(Apple Silicon Mac 한정)<br>Transformers Model의 경우 아래의 모델에 대해 Download Center에서 모델 다운로드를 사전 제공.

|모델 제공|모델명|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**캐릭터 페르소나 설정**
* **시스템 프롬프트(시스템 메시지)**: 챗봇에게 역할을 부여하거나 유저의 요구에 맞도록 시스템에 지시하는 프롬프트. (기본값: 당신은 유용한 AI 비서입니다.) 프리셋 혹은 캐릭터 변경시 이에 맞춰 사전정의된 시스템 프롬프트로 자동 설정됨.
* **캐릭터**: 대화할 상대를 선택가능.
* **프리셋**: 사용자 정의된 시스템 프롬프트를 적용. 캐릭터 변경시 이에 대응하는 프리셋으로 자동 설정됨.

**하이퍼파라미터 설정**<br>하이퍼파라미터의 작동원리에 대해 잘 알지 못할경우 기본값으로 두는걸 권장합니다.

* **시드 값**: 생성 과정에 사용되는 난수의 초기값. (기본값: 42)
* **온도(temperature)**: 답변의 창의성과 무작위성을 조절하는 하이퍼파라미터. 높을수록 예측하기 어렵고 창의적인 답변을 생성. 낮을수록 결정적이고 보수적인 답변을 생성. (기본값: 0.6)
* **Top K**: 고려하는 옵션의 수를 제한하고 가능성이 가장 높은 단어를 선택하여 고품질 출력을 보장하는 하이퍼파라미터. (기본값: 20)
* **Top P**: 답변의 창의성과 무작위성을 조절하는 하이퍼파라미터. 임계 확률을 설정하고 누적 확률이 임계치를 초과하는 상위 토큰을 선택한 뒤 모델이 토큰 세트에서 무작위로 샘플링하여 출력을 생성. (기본값: 0.9)
* **반복 패널티(repetition penalty)**: 중복 단어의 수를 조절하는 하이퍼파라미터. 높을수록 중복되는 단어가 적어짐. (기본값: 1.1)

### Image Generation
Stable Diffusion, Flux등 이미지 생성 모델을 활용한 이미지 생성. ComfyUI를 백엔드 서버로 활용하여 이미지를 생성함.

**지원 모델**
* **API**
현재 이미지 생성 API 모델은 제한적으로 지원중.

|모델 제공|모델명|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **Local**: Diffusers, Checkpoints
 - **Diffusers**: 현재, Diffusers 모델의 경우 스캔도 가능하며 선택도 가능하지만 Diffusers 모델을 사용한 실제 이미지 생성은 사용할수 없음. (추후 기능 구현 예정.)
 - **Checkpoints**: ComfyUI를 경유하여 이미지를 생성함. ComfyUI 디렉토리의 models/checkpoints에 모델 파일이 들어있어야함.

Local Model의 경우 다음과 같은 Base Model을 지원.
- Stable Diffusion 1.5
- Stable Diffusion 2.x
- Stable Diffusion XL 1.0
- Stable Diffusion 3 Medium
- FLUX.1 Schnell
- FLUX.1 Dev
- Stable Diffusion 3.5 Large
- Stable Diffusion 3.5 Medium

* **LoRA**: Local Model의 경우 LoRA를 최대 10개 까지 선택가능. 단, 베이스 모델에 맞는 LoRA를 적용해야 함.
* **VAE**: VAE를 사용자 지정. Default로 둘 경우, Checkpoints에 내장된 VAE를 사용.
* **Embedding**: 사용시 embedding:name과 같은 방식으로 입력하여 적용.
* **ControlNet**: 현재 ai-companion에서는 인터페이스가 구현되지 않았으며, 추후 구현 예정.
* **Refiner**: Stable Diffusion XL 1.0 Refiner 모델로, Refiner Start Step로 리파이너 샘플링이 시작되는 단계를 지정할수 있음.

**생성 옵션**

* **Positive Prompt**: 입력한 단어에 해당되는 이미지를 생성.
* **Negative Prompt**: 결과 이미지에서 보고 싶지 않은 것들을 제외.
* **Width, Height**: 이미지의 너비, 높이를 조정.
* **Recommended Resolution**

|Base Model|Recommended Resolution|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|

* **generation Steps**: 인공지능이 노이징된 이미지를 복구할 때 몇 단계의 스텝을 걸쳐 이미지를 복구 시킬지 여부를 결정하는 값으로, 값이 높을수록 생성까지 걸리는 시간이 길어짐.
* **Denoise Strength**: 노이즈의 강도를 조정.

**Advanced Settings**

* **Sampler**: 샘플링 방식이 다르면 같은 프롬프트에서도 다른 결과를 얻을 수 있음.
* **Scheduler**: 스케쥴러가 다를경우 같은 샘플러와 같은 프롬프트에서도 다른 결과를 얻을 수 있음.
* **CFG Scale**: CFG 값이 높을수록 프롬프트의 설명을 잘 따르고, 낮을수록 창의적으로 이미지를 생성함.
* **Seed**: 생성 과정에 사용되는 난수의 초기값.
* **Clip Skip**: 이미지 생성 과정의 일부를 건너뛰는 기능.
* **Batch Size**: 한번의 실행으로 생성할 이미지의 갯수.

**Image to Image**<br>이미지에 변화를 줄수 있음. Inpaint를 사용하여 Mask 구간에 대해서만 변화를 줄수 있음.

### Storyteller
LLM을 활용하여 텍스트를 생성. Chatbot과는 달리 소설 등의 글작성에 UI가 최적화됨.(현재 미완성.)

**지원 모델**<br>Chatbot 문단내 지원 모델 참조.

**하이퍼파라미터 설정**<br>Chatbot 문단내 하이퍼파라미터 설정 참조.<br>하이퍼파라미터의 작동원리에 대해 잘 알지 못할경우 기본값으로 두는걸 권장합니다.

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
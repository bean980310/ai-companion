# 인공지능 컴패니언 프로젝트

## 프로젝트 소개

생성형 AI 모델들을 활용한 인공지능 컴패니언 서비스 개발

챗봇 서비스와 AI를 활용한 그림, 비디오, 오디오 생성 서비스를 제공하며 유저 페르소나와 캐릭터 페르소나 시스템을 활용한 AI 캐릭터 기능을 통한 페르소나 챗봇 기능을 도입하여 AI를 단순 도구를 넘어서서 친구 및 파트너로서 교류 및 협력하여 대화를 넘어선 여러가지 작업 및 놀이를 제공하는 서비스

유저 페르소나는 주로 일반 사용자를 대상으로 삼으며, 그림, 비디오, 음악, 오디오 생성 서비스는 프로 사용자 역시 대상에 포함.

## 프로젝트 목표

- 언어모델을 활용한 AI챗봇 개발
- 스테이블 디퓨전, FLUX 등 그림 생성 AI를 활용하여 그림 생성 서비스를 개발
- ComfyUI를 백엔드로 활용하여 개발기간 단축
- MusicGen, AudioCraft 등 오디오, 음악 생성 AI를 활용하여 오디오 생성 서비스를 개발
- ToonCrafter, Mochi-1 등 비디오 생성 AI를 활용하어 비디오 생성 서비스를 개발
- 유저 페르소나, 캐릭터 페르소나를 도입하여 AI챗봇에 인격을 부여한 대화
- 사용자 층은 주로 일반 사용자 대상
- VisionAI, OCR, GANs를 활용하여 이미지 인식 시스템 구축
- Langchain을 활용하여 채팅 및 이미지 인식을 통해 프롬프트를 추출하고 이를 바로 그림, 비디오, 오디오 AI 서비스와 연계하여 연동성 있는 서비스 제공
- LLM, 번역 모델을 활용한 실시간 번역 서비스 구축
- Gradio UI, Streamlit등을 이용하여 웹 인터페이스를 통한 사용자 친화적 인터페이스 제공
- 각 캐릭터마다 재능을 부여하여 캐릭터를 통한 생성 AI 서비스를 제공하는 시스템 구축
- 음성 대화 서비스를 통해 실제로 캐릭터와 대화하는 듯한 분위기를 조성하여 실시간으로 대화하는 서비스 제공
- 로컬에서 구동이 가능하도록 llama.cpp, ollama, mlx, onnx 등과 같은 추론 라이브러리를 이용하여 개인용 컴퓨터에서의 구동에 부담이 없도록 최적화
- 성능 최적화시 GGUF, MLX, ONNX형식으로 양자화를 햐여 개인용 컴퓨터에서의 구동에 부담이 없도록 함

## 프로젝트 개발에 사용되는 자원

- Python - 개발 언어
- pytorch - 머신러닝 라이브러리
- transformers, datasets, diffusers, safetensors - 허깅페이스 라이브러리
- langchain - LLM을 사용한 애플리케이션 개발 프레임워크
- llama.cpp, ollama, mlx, onnx - 추론 라이브러리
- gradio, streamlit - 사용자 인터페이스 라이브러리
- Tesseract, EasyOCR, pdf2image - 광학 인식 처리 라이브러리
- SQL - 데이터베이스
- Llama3.1, Qwen2.5 - 대형 언어 모델
- Stable Diffusion, FLUX - 그림 생성 모델
- MusicGen, AudioGen, AudioCraft - 오디오, 음악 생성 모델
- ToonCrafter, Mochi-1 - 비디오 생성 모델
- T5, BERT - 자연어 처리 모델
- M2M100 - 다국어 번역 모델
- CLIP, Llama3.2 Vision, Qwen2VL - 비전 언어 모델
- OpenAI GPT4o, DALL E3, Google Gemini 1.5 Pro – API
- ComfyUI - 백엔드
- Colab Pro+ - 클라우드 컴퓨팅 시스템
- Ngrok - 네트워크 터널링 프로그램


## 역할 분담

**공통 역할**

- 매일 정기적으로 회의 참여
- 팀원들과 디스코드 혹은 Zoom을 통해 소통하기
- 애플리케이션 테스트

**장현빈(팀장)**:

- git 저장소와 소스 코드 관리
- 프로젝트 진행을 위한 데이터 수집
- Google Colab과 ngrok 연동
- OCR, 생성AI 모델을 활용한 서비스 구현
- gradio, Streamlit등을 이용한 사용자 친화적 인터페이스 제공
- 프롬프트 작성을 통한 페르소나 도입
- langchain을 활용한 생성 AI 자동화
- llama.cpp, ollama, mlx를 통한 최적화 작업
- 필요한 경우 파인튜닝 및 양자화하여 모델 최적화 후 모델 평기 지표를 활용한 성능 평가
- 데이터 수집후 전처리 작업 수행
- 강의 혹은 커리큘럼을 만들고 팀원들 가르치기
- Notion 에다가 프로젝트 진행 상황 정리

**윤재선**: 제가 잘 몰라서 뭘할지를 모르겠고 굳은일 있으면 맡겨주세요.

- 강의 혹은 커리큘럼을 통해 배워가면서 진행
- git 활용하는 방법 배우기
- Notion 에다가 배운 내용들 정리하기
- 최적화 모델 제공 받을시 자연어 처리 및 모델 평가 지표를 활용한 성능 평가

**이용석**: 저도 하나씩 시켜주시면 쫒아가볼게요…

- 강의 혹은 커리큘럼을 통해 배워가면서 진행
- Notion 에다가 배운 내용들 정리하기
- 최적화 모델 제공 받을시 자연어 처리 및 모델 평가 지표를 활용한 성능 평가
- 데이터 제공 받을시 데이터 전처리 작업 수행

## 기대효과

- 사람들이 인공지능을 도구를 넘어서서 친구, 동료로써 유대감 형성
- 프로그래밍, 프롬프트에 대한 지식이 없어도 생성형 AI를 다룰수 있도록 하는 환경 조성
- 개인용 컴퓨터에서도 부담이 없는 구동을 통한 생성형 AI 대중화
- 비즈니스, 교육, 엔터테인먼트 등에서 AI가 단순 도구를 넘어 동료로써 서로 협력하면서 업무에 활용

## 자신의 컴퓨터 환경에 맞는 requirements를 설치하기

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

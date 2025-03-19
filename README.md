# AI Companion for Local Machines
An AI companion that uses generative AI models (LLM, Diffusion, etc.) that run in a local machines with gradio app.

[English](README.md) [한국어](README_ko.md) [日本語](README_ja.md)

## Introduction

A service that provides chatbot services and AI-based image, video, and audio generation services, and introduces persona chatbot functions through AI character functions that utilize user persona and character persona systems, thereby enabling AI to go beyond simple tools and interact and cooperate as friends and partners, providing various tasks and play beyond conversation.

User personas are primarily targeted at regular users, and image, video, music, and audio creation services also target professional users.

## Features
### Chatbot
Conversation with AI using your LLM.

**Supported Models**
* **API**

|Developer|Model repo id|
|-------|----|
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.5-preview|
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **Local**: Transformers, GGUF, MLX (Apple Silicon Mac only)<br>For Transformers models, model downloads are provided in advance in the Download Center for the models below.

|Developer|Model repo id|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**Character Persona Setting**
* **System Prompt(System Message)**: A prompt that assigns a role to the chatbot or instructs the system to meet the user's needs. (Default: You are an useful AI assistant.) When changing presets or characters, the system prompt is automatically set to a predefined prompt.
* **Character**: You can choose who you want to conversation.
* **Preset**: Apply a customized system prompt. When changing characters, the corresponding preset is automatically set.

**Hyperparameters**<br>If you are not familiar with how hyperparameters work, we recommend leaving them at default values.

* **Seed**: The initial random number used in the generation process. (Default: 42)
* **Temperature**: Hyperparameter that controls the creativity and randomness of the answers. The higher the value, the more difficult to predict and creative the answers are generated. The lower the value, the more deterministic and conservative the answers are generated. (Default: 0.6)
* **Top K**: A hyperparameter that limits the number of options considered and selects the most probable word, ensuring high quality output. (Default: 20)
* **Top P**: Hyperparameter that controls the creativity and randomness of the answers. Set a threshold probability and select the top tokens whose cumulative probability exceeds the threshold, then the model randomly samples from the token set to generate output. (Default: 0.9)
* **Repetition Penalty**: Hyperparameter that controls the number of duplicate words. The higher the value, the fewer duplicate words there are. (Default: 1.1)

### Image Generation
Image generation using image generation models such as Stable Diffusion and Flux. Images are generated using ComfyUI as a backend server.

**Supported Models**
* **API**
Currently, the image creation API model has limited support.

|Developer|Model repo id|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **Local**: Diffusers, Checkpoints
 - **Diffusers**: Currently, scanning and selection are possible for the Diffusers model, but actual image creation using the Diffusers model is not available. (This feature will be implemented in the future.)
 - **Checkpoints**: Generate images via ComfyUI. Model files must be in models/checkpoints in the ComfyUI directory.

For Local Model, the following Base Models are supported.
- Stable Diffusion 1.5
- Stable Diffusion 2.x
- Stable Diffusion XL 1.0
- Stable Diffusion 3 Medium
- FLUX.1 Schnell
- FLUX.1 Dev
- Stable Diffusion 3.5 Large
- Stable Diffusion 3.5 Medium
- Illustrious XL 1.0

* **LoRA**: For Local Model, you can select up to 10 LoRA. However, you must apply LoRA that matches the base model.
* **VAE**: Customize the VAE. If left as default, the VAE built into Checkpoints will be used.
* **Embedding**: When using, input it in the same way as embedding:name and apply it.
* **ControlNet**: Currently, the interface is not implemented in ai-companion, but will be implemented in the future.
* **Refiner**: With the Stable Diffusion XL 1.0 Refiner model, you can specify the step at which refiner sampling begins with the Refiner Start Step.

**Generation Option**

* **Positive Prompt**: Generate an image corresponding to the entered word.
* **Negative Prompt**: Exclude things you don't want to see in the resulting image.
* **Width, Height**: Adjust the width and height of the image.
* **Recommended Resolution**

|Base Model|Recommended Resolution|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|
|Illustrious XL 1.0|1536x1536 <br> 1248x1824 <br> 1824x1248|

* **generation Steps**: This is a value that determines how many steps the AI ​​will take to restore a noisy image. The higher the value, the longer it takes to create the image.
* **Denoise Strength**: Adjust the intensity of the noise.

**Advanced Settings**

* **Sampler**: Different sampling methods can lead to different results even from the same prompt.
* **Scheduler**: Different schedulers can lead to different results even with the same sampler and prompt.
* **CFG Scale**: The higher the CFG value, the better the image is at following the prompt's instructions, while the lower the value, the more creative the image is.
* **Seed**: The initial value of the random number used in the generation process.
* **Clip Skip**: Ability to skip parts of the image generation process.
* **Batch Size**: The number of images to generate in one run.

**Image to Image**<br>Edit the image. When using Inpaint, edit only the mask area.

### Storyteller
Generate text using LLM. Unlike Chatbot, the UI is optimized for writing novels and other texts. (Currently unfinished.)

**Supported Models**<br>Chatbot 문단내 지원 모델 참조.

**Hyperparameters**<br>See Hyperparameter settings in the Chatbot paragraph.<br>If you are not familiar with how hyperparameters work, we recommend leaving them at the default values.

### Video Generation
Coming Soon

### Audio Generation
Coming Soon

### Translator
Multilingual translation using translation models. You can also upload image files or PDF files, extract text, and use that text for translation.

## Get Started

- **Python venv Setup**

```shell
# conda (Recommaned!)
# Python 3.10
conda create -n ai-companion python=3.10
# Python 3.11
conda create -n ai-companion python=3.11
# Python 3.12
conda create -n ai-companion python=3.12
conda activate ai-companion
# venv
python3 -m venv venv
# for MacOS, Linux
source venv/bin/activate 
# for Windows
source venv/Scripts/activate 
```

- **for Windows**
```shell
pip install -r requirements_windows_amd64.txt
```

- **for macOS(Apple Silicon Mac)**
```zsh
pip install -r requirements_macos_arm64.txt
```
(When Intel CPU Mac, install requirements.txt)

- **for Linux**
```bash
pip install -r requirements_linux.txt
```

### Note: Install xformers from Apple Silicon Mac!

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

## Run

```zsh
python app.py
```

### About Intel Mac support.

It may not work properly on Intel Macs and most features are not supported.
Therefore, we may discontinue support for Intel Macs at or before the final release date.
# AI Companion for Local Machines

A Gradio-based AI companion app leveraging generative AI models (LLMs, Diffusion models, etc.) running in a local environment.

[English](README.md) [한국어](README_ko.md) [日本語](README_ja.md) [简体中文](README_zh_cn.md) [繁體中文](README_zh_tw.md)

## Overview

The AI Companion offers chatbot services along with AI-powered generation of images, videos, and audio content. Utilizing user personas and character personas, the app introduces persona-based chatbot interactions, transforming AI beyond mere tools into genuine friends and partners. Users can engage in conversations, collaborative tasks, and recreational activities with AI characters.

While the user persona mainly targets general users, the image, video, music, and audio generation services also cater to professional users.

## Main Features

### Chatbot

Interact with AI using Large Language Models (LLMs).

**Supported Models**  

* **API**

|Provider|Model Name|
|-------|----|
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.1 <br> gpt-4.1-mini <br> gpt-4.1-nano |
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **Local**: Transformers, GGUF, MLX (Apple Silicon Macs only)  
Pre-downloaded Transformer models are available via the Download Center.

|Provider|Model Name|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**Character Persona Settings**  

* **System Prompt(System Message)**: Instructions that define the chatbot’s role or behavior based on user needs. (Default: You are an useful AI assistant.) Presets or characters automatically apply predefined prompts.
* **Character**: Choose your AI conversation partner.
* **Preset**: Applies user-defined system prompts. Presets automatically update when switching characters.

**Hyperparameter Settings**  
If you aren’t familiar with hyperparameters, it’s recommended to stick to the default values.

* **Seed**: Initial random number seed used in generation (default: 42).
* **Temperature**: Controls creativity/randomness of responses. Higher values yield more creative answers; lower values result in deterministic responses (default: 0.6).
* **Top K**: Limits the number of word options, choosing from the most probable words for high-quality outputs (default: 20).
* **Top P**: Adjusts randomness by setting a probability threshold, sampling from top tokens until the threshold is reached (default: 0.9).
* **Repetition Penalty**: Controls repetition of words; higher values reduce repetition (default: 1.1).

### Image Generation

Create images using models like Stable Diffusion and Flux. Images are generated via ComfyUI backend.

**Supported Models**  

* **API**
Currently, API image generation models have limited support.

|Developer|Model repo id|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **Local**: Diffusers, Checkpoints
  * **Diffusers**: Currently, Diffusers models can be scanned and selected but cannot yet generate images (feature coming soon).
  * **Checkpoints**: Image generation via ComfyUI. Model files should be placed in models/checkpoints within the ComfyUI directory.

Supported Local Base Models:

* Stable Diffusion 1.5
* Stable Diffusion 2.x
* Stable Diffusion XL 1.0
* Stable Diffusion 3 Medium
* FLUX.1 Schnell
* FLUX.1 Dev
* Stable Diffusion 3.5 Large
* Stable Diffusion 3.5 Medium
* Illustrious XL 1.0

**Detailed options**  

* **LoRA**: Supports up to 10 LoRA models per base model. Compatible LoRAs must be applied.
* **VAE**: User-defined VAE. Defaults to embedded VAE within the Checkpoints.
* **Embedding**: Use via syntax like embedding:name.
* **ControlNet**: Interface not yet implemented; coming soon.
* **Refiner**: For Stable Diffusion XL 1.0, select the start step for refiner sampling.

**Generation Options**  

* **Positive Prompt**: Generate images based on the input text.
* **Negative Prompt**: Exclude unwanted features from generated images.
* **Width, Height**: Adjust image dimensions.
* **Recommended Resolutions**

|Base Model|Recommended Resolution|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|
|Illustrious XL 1.0|1536x1536 <br> 1248x1824 <br> 1824x1248|

* **Generation Steps**: Higher values result in longer generation times but potentially better quality.
* **Denoise Strength**: Adjusts the level of noise.

**Advanced Settings**  

* **Sampler**: Different samplers yield varying outputs.
* **Scheduler**: Affects output even with the same sampler/prompt.
* **CFG Scale**: Higher CFG values closely follow prompts; lower values are more creative.
* **Seed**: Initial random number seed.
* **Clip Skip**: Skips certain steps during image creation.
* **Batch Size**: Number of images generated per run.

**Image to Image**  
Modify existing images or use Inpaint for selective alterations.

### Storyteller

Utilizes LLMs optimized for text creation, especially for storytelling. (Currently under development.)

**Supported Models**  
Same as Chatbot.

**Hyperparameters**  
Same as Chatbot.  
If you are not familiar with how hyperparameters work, we recommend leaving them at the default values.

### Video Generation

Coming Soon

### Audio Generation

Coming Soon

### Translator

Multi-language translation using translation models. Supports text extraction from uploaded images or PDF files for translation.

## Installation

**Clone a repository**  

```shell
# Clone a repository only
git clone https://github.com/bean980310/ai-companion.git
# Clone a repository with submodules
git clone --recursive https://github.com/bean980310/ai-companion.git
# Init and Update a submodules
git submodule init
git submodule update
```

* **Virtual Environment Setup**

```shell
# conda (Recommended!)
# Python 3.10
conda create -n ai-companion python=3.10
# Python 3.11
conda create -n ai-companion python=3.11
# Python 3.12
conda create -n ai-companion python=3.12
conda activate ai-companion
# venv
python3 -m venv venv
source venv/bin/activate # macOS/Linux
source venv/Scripts/activate # Windows
```

**Install dependencies**  

* **Pytorch**

```shell
# for Apple Silicon Mac and Windows with CPU and Linux with CUDA 12.4
pip install torch torchvision torchaudio
# for Windows with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

* **Common**

```shell
pip install -r requirements/common.txt
pip install -r requirements/common_2.txt
pip install -r requirements/ai_models.txt
pip install -r requirements/ai_models_gguf.txt
```

* **Windows**

```shell
# on Windows
pip install -r requirements/windows_amd64.txt
# on Windows Subsystem for Linux 2
pip install -r requirements/windows_amd64_wsl2.txt
```

* **macOS(Apple Silicon Mac)**

```zsh
pip install -r requirements/macos_arm64.txt
pip install -r requirements/macos_arm64_mlx.txt
```

* **Linux**

```bash
# on AMD64 with NVIDIA GPU
pip install -r requirements/linux_amd64_cuda.txt
# on AMD64 with AMD GPU
pip install -r requirements/linux_amd64_rocm.txt
# on ARM64 (NVIDIA GPU only)
pip install -r requirements/linux_arm64.txt
# on Google Colab TPU
pip install -r requirements/linux_colab_tpu.txt
```

* **MeloTTS(Optional)**

```shell
pip install git+https://github.com/myshell-ai/MeloTTS.git --no-deps
```

### Installing xformers on Mac

```zsh
brew update
brew install gcc cmake llvm@16 libomp
```

```zsh
export PATH="/opt/homebrew/opt/llvm@16/bin:$PATH"

export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export CXXFLAGS="-Xpreprocessor -fopenmp"

export CC=/opt/homebrew/opt/llvm@16/bin/clang
export CXX=/opt/homebrew/opt/llvm@16/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/llvm@16/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm@16/include"
```

```zsh
pip install xformers
```

## Run

```zsh
python app.py
```

### Support for Intel Macs

AI Companion for Local Machines no longer supports Intel CPU-based Macs.  
If you are using an Intel CPU-based Macs, we recommend that you consider migrating to an Apple Silicon Based Macs or a Windows PC or Linux machine with an Nvidia GPU environment. If you have difficulty migrating from an Intel CPU-based Macs, you can use a companion application that supports Intel CPU-based Macs instead.

### Guidance for Windows Systems with GPUs That Do Not Support CUDA 12.4 or Higher

AI Companion for Local Machines provides support for dependency installation to ensure compatibility between PyTorch and xformers.  
On Windows, versions of CUDA lower than 12.4 cannot install the latest xformers. Therefore, for CUDA 11.8 and 12.1 environments, the PyTorch version is fixed at **2.4.0** to ensure compatibility with the latest supported xformers version, **xformers 0.0.27.post2**.  
Please note that future support for Windows systems with CUDA versions below 12.4 will be discontinued.  
If you are using Windows and have a CUDA version below 12.4, we recommend upgrading to CUDA 12.4 or higher and reinstalling PyTorch and xformers accordingly. If your GPU does not support CUDA 12.4 or higher, we recommend upgrading to a compatible GPU or considering migration to a newer PC.  

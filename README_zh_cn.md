# 面向本地环境的AI伴侣

一款基于Gradio、运行于本地环境的生成式AI（LLM、Diffusion等）伴侣应用程序。

[English](README.md) [한국어](README_ko.md) [日本語](README_ja.md) [简体中文](README_zh_cn.md) [繁體中文](README_zh_tw.md)

## 概述

提供包括聊天机器人在内的图像、视频、音频等多种内容生成服务。尤其引入了用户人格（User Persona）与角色人格（Character Persona）系统，使AI不再只是工具，更成为用户的朋友和伙伴，用户不仅可以与AI进行日常对话，还能共同协作完成任务或一起享受娱乐活动。

用户人格主要面向一般用户，而图像、视频、音乐和音频生成服务也可满足专业用户的需求。

## 主要功能

### 聊天机器人（Chatbot）

利用LLM与AI进行自然对话。

**支持的模型**  

* **API方式**

|提供商|模型名称|
|-------|----|
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.1 <br> gpt-4.1-mini <br> gpt-4.1-nano |
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **本地方式**：Transformers、GGUF、MLX（仅支持Apple Silicon Mac）<br>预先在下载中心提供Transformers模型。

|提供商|模型名称|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**角色人格设置**  

* **系统提示（System Prompt）**: 定义聊天机器人的基本角色和行为指令（默认值：“你是一个乐于助人的AI助手。”）更改角色或预设时会自动更新。
* **角色（Character）**: 选择你希望互动的AI角色。
* **预设（Preset）**: 用户自定义的系统提示，更改角色时自动切换对应的预设。

**超参数设置**  
如不熟悉超参数，建议保持默认值。

* **种子值（Seed）**: 随机数生成初始值（默认：42）
* **温度（Temperature）**: 控制回复的创意性和随机性（默认：0.6）
* **Top K**: 限制候选词汇的最大数量（默认：20）
* **Top P**: 根据累计概率从顶层token中采样（默认：0.9）
* **重复惩罚（Repetition Penalty）**: 控制词汇重复频率（默认：1.1）

### 图像生成（Image Generation）

通过Stable Diffusion、Flux等模型生成图像。以ComfyUI作为后端服务。

**支持的模型**  

* **API方式**  
目前仅支持有限的模型。

|提供商|模型名称|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **本地方式**：Diffusers、Checkpoints
  * **Diffusers**：目前仅支持模型扫描和选择，尚无法实际生成图像（后续支持）。
  * **Checkpoints**：通过ComfyUI生成图像，模型文件需置于models/checkpoints目录下。

支持的基础模型：

* Stable Diffusion 1.5
* Stable Diffusion 2.x
* Stable Diffusion XL 1.0
* Stable Diffusion 3 Medium
* FLUX.1 Schnell
* FLUX.1 Dev
* Stable Diffusion 3.5 Large
* Stable Diffusion 3.5 Medium
* Illustrious XL 1.0

**详细选项**  

* **LoRA**: 每个基础模型最多支持加载10个LoRA。
* **VAE**: 默认使用内置VAE，也可自定义。
* **Embedding**: 以embedding:name的形式使用。
* **ControlNet**: 当前界面未实现，后续支持。
* **Refiner**: 针对SDXL 1.0，可设定Refiner采样起始步骤。

**图像生成选项**  

* **Positive Prompt**: 描述想要生成图像的特征。
* **Negative Prompt**: 描述排除的特征。
* **宽度、高度（Width, Height）**: 指定图像尺寸。
* **推荐分辨率**

|基础模型|推荐分辨率|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|
|Illustrious XL 1.0|1536x1536 <br> 1248x1824 <br> 1824x1248|

* **生成步骤（Generation Steps）**: 值越高质量越佳，但耗时越长。
* **去噪强度（Denoise Strength）**: 调整噪点程度。

**高级设置**  

* **采样器（Sampler）**: 选择不同的采样方法。
* **调度器（Scheduler）**: 即使相同条件，结果也可能不同。
* **CFG比例（CFG Scale）**: 值越高越严格遵循提示词，值越低生成结果越具创意性。
* **种子值（Seed）**: 随机数种子。
* **Clip Skip**: 跳过部分图像生成步骤。
* **批量大小（Batch Size）**: 每次生成的图像数量。

**图像到图像（Image to Image）**  
支持对现有图像进行修改和局部重绘（Inpaint）。

### 故事创作（Storyteller）

基于LLM的专门文本创作界面（开发中）。

### 视频生成（Video Generation）

即将支持。

### 音频生成（Audio Generation）

即将支持。

### 翻译器（Translator）

利用翻译模型实现多语言翻译，支持从图像或PDF文件中提取文本进行翻译。

## 安装方法

**克隆存储库**  

```shell
# 仅克隆一个存储库
git clone https://github.com/bean980310/ai-companion.git
# 克隆带有子模块的存储库
git clone --recursive https://github.com/bean980310/ai-companion.git
# 初始化并更新子模块
git submodule init
git submodule update
```

* **虚拟环境设置**

```shell
# 使用conda（推荐）
conda create -n ai-companion python=3.10
conda activate ai-companion

# 使用其他python版本
cd ai-companion
# 使用venv
python3 -m venv venv
# 使用uv
uv venv --python 3.10 
uv venv --python 3.11
uv venv --python 3.12
# MacOS/Linux/Windows WSL2
source venv/bin/activate # venv
source .venv/bin/activate # uv
# Windows
.\venv\Scripts\activate.bat # venv
.\.venv\Scripts\activate.bat # uv
```

**安装依赖**  

* **Windows 环境**：

```shell
# on Windows
.\installer_windows_amd64.bat
```

```pwsh
# on Windows (Powershell)
.\installer_windows_amd64.ps1
```

```bash
# on Windows Subsystem for Linux 2
bash installer_windows_amd64_wsl2.sh
# or
./installer_windows_amd64_wsl2.sh
```

* **macOS 环境（Apple Silicon Mac）**：

```zsh
zsh installer_macos_arm64.sh
# or
./installer_macos_arm64.sh
```

* **Linux 环境**：

```bash
bash installer_linux_amd64_cuda.sh
# or
./installer_linux_amd64_cuda.sh
```

* **MeloTTS(Optional)**

```shell
pip install git+https://github.com/myshell-ai/MeloTTS.git --no-deps
```

### macOS下安装xformers注意事项

```zsh
brew update
brew install gcc cmake llvm@18 libomp
```

```zsh
export PATH="/opt/homebrew/opt/llvm@18/bin:$PATH"

export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
export CXXFLAGS="-Xpreprocessor -fopenmp"

export CC=/opt/homebrew/opt/llvm@18/bin/clang
export CXX=/opt/homebrew/opt/llvm@18/bin/clang++
export LDFLAGS="-L/opt/homebrew/opt/llvm@18/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm@18/include"
```

```zsh
pip install --no-build-isolation --upgrade xformers
```

## 运行方法

```zsh
python app.py
```

### Intel Mac支持说明

面向本地环境的AI伴侣 不再支持基于 Intel CPU 的 Mac。  
如果您使用的是基于 Intel CPU 的 Mac，我们建议您考虑迁移到基于 Apple Silicon 的 Mac，或使用 Nvidia GPU 环境的 Windows PC 或 Linux 机器。如果您从基于 Intel CPU 的 Mac 迁移遇到困难，可以使用支持基于 Intel CPU 的 Mac 的配套应用。

### 关于不支持 CUDA 12.4 及以上版本的 GPU 在 Windows 上的支持说明

本地版 AI Companion 支持安装兼容的 PyTorch 与 xformers 依赖项。  
在 Windows 环境下，低于 CUDA 12.4 的版本无法安装最新版本的 xformers。因此，在 CUDA 11.8 与 12.1 环境下，为兼容最后支持的 xformers **0.0.27.post2**，我们将 PyTorch 固定为 **2.4.0** 版本。  
未来我们将停止支持安装了 CUDA 12.4 以下版本的 Windows 系统。  
如果您正在使用 Windows 且 CUDA 版本低于 12.4，建议您升级到 12.4 或更高版本，并重新安装与之对应的 PyTorch 与 xformers。如果您的 GPU 不支持 CUDA 12.4 或更高版本，建议更换为支持 CUDA 12.4 的 GPU，或考虑迁移至更新的计算机。  

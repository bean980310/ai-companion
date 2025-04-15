# 面向本地環境的AI伴侶

一款基於Gradio、運行於本地環境的生成式AI（LLM、Diffusion等）伴侶應用程序。

[English](README.md) [한국어](README_ko.md) [日本語](README_ja.md) [简体中文](README_zh_cn.md) [繁體中文](README_zh_tw.md)

## 概述

提供包括聊天機器人在內的圖像、視頻、音頻等多種內容生成服務。尤其引入了用戶人格（User Persona）與角色人格（Character Persona）系統，使AI不再只是工具，更成爲用戶的朋友和夥伴，用戶不僅可以與AI進行日常對話，還能共同協作完成任務或一起享受娛樂活動。

用戶人格主要面向一般用戶，而圖像、視頻、音樂和音頻生成服務也可滿足專業用戶的需求。

## 主要功能
### 聊天機器人（Chatbot）
利用LLM與AI進行自然對話。

**支持的模型**
* **API方式**

|提供商|模型名稱|
|-------|----|
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.5-preview|
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **本地方式**：Transformers、GGUF、MLX（僅支持Apple Silicon Mac）<br>預先在下載中心提供Transformers模型。

|提供商|模型名稱|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**角色人格設置**
* **系統提示（System Prompt）**: 定義聊天機器人的基本角色和行爲指令（默認值：“你是一個樂於助人的AI助手。”）更改角色或預設時會自動更新。
* **角色（Character）**: 選擇你希望互動的AI角色。
* **預設（Preset）**: 用戶自定義的系統提示，更改角色時自動切換對應的預設。

**超參數設置**<br>如不熟悉超參數，建議保持默認值。

* **種子值（Seed）**: 隨機數生成初始值（默認：42）
* **溫度（Temperature）**: 控制回覆的創意性和隨機性（默認：0.6）
* **Top K**: 限制候選詞彙的最大數量（默認：20）
* **Top P**: 根據累計概率從頂層token中採樣（默認：0.9）
* **重複懲罰（Repetition Penalty）**: 控制詞彙重複頻率（默認：1.1）

### 圖像生成（Image Generation）
通過Stable Diffusion、Flux等模型生成圖像。以ComfyUI作爲後端服務。

**支持的模型**
* **API方式**<br>目前僅支持有限的模型。

|提供商|模型名稱|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **本地方式**：Diffusers、Checkpoints
 - **Diffusers**：目前僅支持模型掃描和選擇，尚無法實際生成圖像（後續支持）。
 - **Checkpoints**：通過ComfyUI生成圖像，模型文件需置於models/checkpoints目錄下。

 支持的基礎模型：
- Stable Diffusion 1.5
- Stable Diffusion 2.x
- Stable Diffusion XL 1.0
- Stable Diffusion 3 Medium
- FLUX.1 Schnell
- FLUX.1 Dev
- Stable Diffusion 3.5 Large
- Stable Diffusion 3.5 Medium
- Illustrious XL 1.0

**詳細選項**

* **LoRA**: 每個基礎模型最多支持加載10個LoRA。
* **VAE**: 默認使用內置VAE，也可自定義。
* **Embedding**: 以embedding:name的形式使用。
* **ControlNet**: 當前界面未實現，後續支持。
* **Refiner**: 針對SDXL 1.0，可設定Refiner採樣起始步驟。

**圖像生成選項**

* **Positive Prompt**: 描述想要生成圖像的特徵。
* **Negative Prompt**: 描述排除的特徵。
* **寬度、高度（Width, Height）**: 指定圖像尺寸。
* **推薦分辨率**

|基礎模型|推薦分辨率|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|
|Illustrious XL 1.0|1536x1536 <br> 1248x1824 <br> 1824x1248|

* **生成步驟（Generation Steps）**: 值越高質量越佳，但耗時越長。
* **去噪強度（Denoise Strength）**: 調整噪點程度。

**高級設置**

* **採樣器（Sampler）**: 選擇不同的採樣方法。
* **調度器（Scheduler）**: 即使相同條件，結果也可能不同。
* **CFG比例（CFG Scale）**: 值越高越嚴格遵循提示詞，值越低生成結果越具創意性。
* **種子值（Seed）**: 隨機數種子。
* **Clip Skip**: 跳過部分圖像生成步驟。
* **批量大小（Batch Size）**: 每次生成的圖像數量。

**圖像到圖像（Image to Image）**<br>支持對現有圖像進行修改和局部重繪（Inpaint）。

### 故事創作（Storyteller）
基於LLM的專門文本創作界面（開發中）。

### 視頻生成（Video Generation）
即將支持。

### 音頻生成（Audio Generation）
即將支持。

### 翻譯器（Translator）
利用翻譯模型實現多語言翻譯，支持從圖像或PDF文件中提取文本進行翻譯。

## 安裝方法

**克隆儲存庫**

```shell
# 僅克隆一個儲存庫
git clone https://github.com/bean980310/ai-companion.git
# 克隆帶有子模組的儲存庫
git clone --recursive https://github.com/bean980310/ai-companion.git
# 初始化並更新子模組
git submodule init
git submodule update
```

- **虛擬環境設置**

```shell
# 使用conda（推薦）
conda create -n ai-companion python=3.10
conda activate ai-companion

# 使用venv
python3 -m venv venv
source venv/bin/activate  # MacOS/Linux
source venv/Scripts/activate  # Windows
```

**安裝依賴**

- **Pytorch**
```shell
# for Apple Silicon Mac and Windows with CPU and Linux with CUDA 12.4
pip install torch torchvision torchaudio
# for Windows with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- **Windows 環境**：
```shell
pip install -r requirements_windows_amd64.txt
```

- **macOS 環境（Apple Silicon Mac）**：
```zsh
pip install -r requirements_macos_arm64.txt
```
（Intel Mac 請使用requirements.txt）

- **Linux 環境**：
```bash
pip install -r requirements_linux.txt
```

- **MeloTTS(Optional)**
```shell
pip install git+https://github.com/myshell-ai/MeloTTS.git --no-deps
```

### macOS下安裝xformers注意事項

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

## 運行方法

```zsh
python app.py
```

### Intel Mac支持說明

對基於 Intel CPU 的 Mac 的支持已停止。
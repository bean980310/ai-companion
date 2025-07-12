# ローカル環境向けAIコンパニオン

ローカル環境で動作する生成系AIモデル（LLM、Diffusionモデル等）を活用した、GradioベースのAIコンパニオンアプリです。

[English](README.md) [한국어](README_ko.md) [日本語](README_ja.md) [简体中文](README_zh_cn.md) [繁體中文](README_zh_tw.md)

## 概要

チャットボット機能に加え、AIを用いた画像、動画、音声生成サービスを提供します。
ユーザーペルソナとキャラクターペルソナの仕組みを導入することで、AIを単なるツールとしてだけでなく、友人やパートナーとして交流し、会話だけでなく様々なタスクや遊びも共に楽しめるサービスを提供します。

ユーザーペルソナは一般ユーザーを主な対象とし、画像・動画・音楽・音声生成サービスはプロユーザーも対象に含みます。

## 主な機能

### チャットボット(Chatbot)

LLMを利用してAIと対話します。

**対応モデル**  

* **API**

|提供元|モデル名|
|-------|----|
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.1 <br> gpt-4.1-mini <br> gpt-4.1-nano |
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **ローカル(Local)**: Transformers、GGUF、MLX（Apple Silicon搭載Macのみ）<br>Transformersモデルは事前にダウンロードセンターから取得可能です。

|提供元|モデル名|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**キャラクターペルソナの設定**  

* **システムプロンプト（システムメッセージ）**: AIに役割を与えたり、ユーザーの要望に沿った指示を与えるプロンプトです。（デフォルト：「あなたは役に立つAIアシスタントです。」）キャラクターやプリセット変更時に自動的に設定されます。
* **キャラクター**: 対話相手のAIキャラクターを選択できます。
* **プリセット**: ユーザーが作成したプロンプトの設定です。キャラクター変更時には対応するプリセットに自動的に切り替わります。

**ハイパーパラメータ設定**  
よく分からない場合はデフォルトのまま使用することを推奨します。

* **シード値(Seed)**: 生成時の乱数初期値（デフォルト: 42）
* **温度(Temperature)**: 回答の創造性やランダム性を制御します。高いほど創造的、低いほど保守的な回答になります（デフォルト: 0.6）
* **Top K**: 候補として考慮するトークンの数を制限し、品質の高い単語を選択します（デフォルト: 20）
* **Top P**: 累積確率で上位トークンを選択し、ランダムにサンプリングします（デフォルト: 0.9）
* **反復ペナルティ(repetition penalty)**: 単語の重複を抑えます。値が高いほど単語の繰り返しが減ります（デフォルト: 1.1）

### 画像生成(Image Generation)

Stable DiffusionやFluxなどの画像生成モデルを利用します。バックエンドにComfyUIを使っています。

**対応モデル**  

* **API**
APIの画像生成モデルは現在限定的に対応しています。

|提供元|モデル名|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **ローカル**: Diffusers, Checkpoints
  * **Diffusers**: 現在モデル選択のみ可能、画像生成は未対応（後日対応予定）。
  * **Checkpoints**: ComfyUIを通して画像を生成。モデルファイルは models/checkpoints ディレクトリに配置。

ローカルでのベースモデル対応:

* Stable Diffusion 1.5
* Stable Diffusion 2.x
* Stable Diffusion XL 1.0
* Stable Diffusion 3 Medium
* FLUX.1 Schnell
* FLUX.1 Dev
* Stable Diffusion 3.5 Large
* Stable Diffusion 3.5 Medium
* Illustrious XL 1.0

**詳細オプション**  

* **LoRA**: ベースモデルに合わせて最大10個まで適用可能。
* **VAE**: ユーザー指定が可能。未指定の場合、Checkpoint内蔵のVAEを使用。
* **Embedding**: embedding:nameの形式で使用可能。
* **ControlNet**: 現在未実装、後日実装予定。
* **Refiner**: SDXL1.0モデルで使用可能、サンプリング開始ステップを指定可能。

**生成オプション**  

* **Positive Prompt**: 含めたい要素を指定。
* **Negative Prompt**: 除外したい要素を指定。
* **Width, Height**: 画像サイズを調整。
* **推奨解像度**: モデルに応じた推奨値

|Base Model|Recommended Resolution|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|
|Illustrious XL 1.0|1536x1536 <br> 1248x1824 <br> 1824x1248|

* **生成ステップ(Generation Steps)**: 高いほど生成時間が長く高品質になる傾向があります。
* **ノイズ除去強度(Denoise Strength)**: ノイズの強度調整。

**高度な設定(Advanced Settings)**  

* **サンプラー(Sampler)**: 出力結果に影響。
* **スケジューラー(Scheduler)**: 同じ条件でも結果が変化。
* **CFGスケール(CFG Scale)**: プロンプトへの追従度を調整。
* **シード(Seed)**: 乱数初期値。
* **Clip Skip**: 画像生成の途中工程をスキップ。
* **バッチサイズ(Batch Size)**: 一度に生成する画像数。

**Image to Image**  
イメージに変化を与えることができる。 Inpaintを使用してマスク区間に対してのみ変化を与えることができます。

### Storyteller

LLMを用いて物語などのテキスト生成を支援する機能（現在開発中）。

**対応モデル**  
Chatbot段落内の対応モデルを参照。

**ハイパーパラメータ設定**  
Chatbot段落内のハイパーパラメータ設定を参照。  
ハイパーパラメータの作動原理についてよく分からない場合は、デフォルトにしておくことをお勧めします。

### 動画生成(Video Generation)

近日対応予定

### 音声生成(Audio Generation)

近日対応予定

### 翻訳(Translator)

翻訳モデルを活用し、多言語翻訳に対応。画像やPDFのテキスト抽出による翻訳も可能。

## インストール方法

**リポジトリをクローンする**  

```shell
# リポジトリのみをクローンする
git clone https://github.com/bean980310/ai-companion.git
# サブモジュールを含むリポジトリをクローンする
git clone --recursive https://github.com/bean980310/ai-companion.git
# サブモジュールの初期化と更新
git submodule init
git submodule update
```

* **仮想環境設定**

```shell
# conda (おすすめ!)
# Python 3.10
conda create -n ai-companion python=3.10
# Python 3.11
conda create -n ai-companion python=3.11
# Python 3.12
conda create -n ai-companion python=3.12
conda activate ai-companion
# その他の仮想環境
cd ai-companion
# venv
python3 -m venv venv
# uv
uv venv --python 3.10 
uv venv --python 3.11
uv venv --python 3.12
# MacOS、Linux環境
source venv/bin/activate 
# Windows環境
source venv/Scripts/activate 
# Windows Powershell環境
.\venv\Scripts\activate
```

**依存関係のインストール**  

* **一般環境**

```shell
pip install -r requirements/common.txt
```

* **Windows環境**

```shell
# on Windows
pip install -r requirements/windows_amd64.txt
# on Windows Subsystem for Linux 2
pip install -r requirements/windows_amd64_wsl2.txt
# Common
pip install -r requirements/ai_models.txt
```

* **macOS環境(Apple Siliconを搭載したMac)**

```zsh
pip install -r requirements/macos_arm64.txt
pip install -r requirements/ai_models.txt
pip install -r requirements/macos_arm64_mlx.txt
```

* **Linux環境**

```bash
# on AMD64 with NVIDIA GPU
pip install -r requirements/linux_amd64_cuda.txt
# on AMD64 with AMD GPU
pip install -r requirements/linux_amd64_rocm.txt
# on ARM64 (NVIDIA GPU only)
pip install -r requirements/linux_arm64.txt
# on Google Colab TPU
pip install -r requirements/linux_colab_tpu.txt
# Common
pip install -r requirements/ai_models.txt
```

* **MeloTTS(Optional)**

```shell
pip install git+https://github.com/myshell-ai/MeloTTS.git --no-deps
```

### Macでxformersをインストールする際の注意事項

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

## 実行方法

```zsh
python app.py
```

### Intel Macについて

ローカル環境向けAIコンパニオンはIntel搭載Macをサポートしなくなりました。  
Intel Macをお使いの場合は、Apple Silicon MacまたはNvidia GPU環境のWindows PC、Linuxマシンへの移行を検討することをお勧めします。 Intel Macでの移行が困難な場合は、Intel Macをサポートするコンパニオンアプリケーションを代わりに使用できます。  

### CUDA 12.4以降を使用できないGPUを搭載したWindowsのサポート案内

ローカル環境向けAIコンパニオンでは、PyTorchとxformersのバージョン互換性を確保するための依存関係のインストールをサポートしています。  
Windows環境では、CUDA 12.4未満では最新のxformersをインストールできません。そのため、CUDA 11.8および12.1環境では、**xformers 0.0.27.post2**との互換性のためにPyTorchのバージョンを**2.4.0**に固定しています。  
また、将来的にCUDA 12.4未満のバージョンがインストールされたWindowsのサポートは終了予定です。  
Windowsをご使用でCUDA 12.4未満をご利用の場合は、CUDAを12.4以上にアップグレードし、PyTorchおよびxformersをそれに合わせて再インストールすることを推奨します。CUDA 12.4以上が使用できないGPUをご使用の場合は、対応するGPUへのアップグレードまたは新しいPCへの移行を検討してください。  

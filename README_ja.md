# ローカルマシンのためのAIコンパニオン
ローカル環境で駆動する生成 AIモデル(LLM、Diffusionなど)を利用した人工知能コンパニオンGradioアプリ。

[English](README.md) [한국어](README_ko.md) [日本語](README_ja.md)

## 概要

チャットボットサービスとAIを活用した絵、ビデオ、オーディオ生成サービスを提供し、ユーザーのペルソナとキャラクターのペルソナシステムを活用したAIキャラクター機能を通じたペルソナチャットボット機能を導入し、AIを単純なツールを越えて友達やパートナーとして交流および協力し、会話を超えた様々な作業および遊びの提供するサービス

ユーザーペルソナは主に一般ユーザーを対象にしており、絵、ビデオ、音楽、オーディオ生成サービスはプロユーザーも対象に含まれる。

## 主要機能
### Chatbot
LLMを活用してAIと会話。

**対応モデル**
* **API**

|開発者|モデル名|
|-------|----|
|OpenAI|gpt-3.5-turbo <br> gpt-4 <br> gpt-4-turbo <br> gpt-4o-mini <br> gpt-4o <br> gpt-4.5-preview|
|Anthropic|claude-3-haiku-20240307 <br> claude-3-sonnet-20240229 <br> claude-3-opus-latest <br> claude-3-5-sonnet-latest <br> claude-3-5-haiku-latest <br> claude-3-7-sonnet-latest|
|Google GenAI|gemini-1.5-flash <br> gemini-1.5-flash-8b <br> gemini-1.5-pro <br> gemini-2.0-flash|

* **Local**: Transformers, GGUF, MLX(Apple Silicon Mac only)<br>Transformers Modelの場合、以下のモデルに対してDownload Centerでモデルダウンロードを事前提供。

|開発者|モデル名|
|--------|-----|
|meta-llama|meta-llama/Llama-3.1-8B <br> meta-llama/Llama-3.1-8B-Instruct <br> meta-llama/Llama-3.2-11B-Vision <br> meta-llama/Llama-3.2-11B-Vision-Instruct|
|google|google/gemma-2-9b <br> google/gemma-2-9b-it <br> google/gemma-3-12b-pt <br> google/gemma-3-12b-it|
|Qwen|Qwen/Qwen2.5-7B <br> Qwen/Qwen2.5-7B-Instruct <br> Qwen/Qwen2.5-14B <br> Qwen/Qwen2.5-14B-Instruct|
|mistralai|mistralai/Mistral-7B-v0.3 <br> mistralai/Mistral-7B-Instruct-v0.3 <br> mistralai/Mistral-Small-24B-Base-2501 <br> mistralai/Mistral-Small-24B-Instruct-2501|

**キャラクターペルソナ設定**
* **システムプロンプト(システムメッセージ)**: チャットボットに役割を付与したり、ユーザーの要求に合うようにシステムに指示するプロンプト。（デフォルト:あなたは有用なAI秘書です。）プリセットまたはキャラクター変更時、これに合わせて事前定義されたシステムプロンプトに自動設定される。
* **キャラクター**: 会話する相手を選択可能。
* **プリセット**: カスタマイズされたシステム プロンプトを適用。 キャラクター変更時、これに対応するプリセットに自動設定される。

**ハイパーパラメータ設定**<br>ハイパーパラメータの作動原理についてよくわからない場合は、デフォルトにしておくことをお勧めします。

* **Seed**: 生成過程で使用される乱数の初期値。(デフォルト: 42)
* **温度(Temperature)**: 回答の創造性とランダム性を調整するハイパーパラメータ。 高いほど予測が難しく、創意的な回答を生成。低いほど決定的で保守的な回答を生成。(デフォルト:0.6)
* **Top K**: 考慮するオプションの数を制限し、可能性が最も高い単語を選択して高品質の出力を保障するハイパーパラメータ。(デフォルト:20)
* **Top P**: 回答の創造性とランダム性を調整するハイパーパラメータ。 臨界確率を設定し、累積確率が臨界値を超える上位トークンを選択した後、モデルがトークンセットからランダムにサンプリングして出力を生成します。(デフォルト:0.9)
* **反復ペナルティ(repetition penalty)**: 重複する単語の数を調整するハイパーパラメータ。 高いほど重複する単語が少なくなる。(デフォルト:1.1)

### Image Generation
Stable Diffusion、Fluxなどの画像生成モデルを活用した画像生成。ComfyUIをバックエンドサーバとして活用し、画像を生成する。

**対応モデル**
* **API**
現在、画像生成APIモデルは制限的にサポート中。

|開発者|モデル名|
|-------|----|
|OpenAI|dall-e-3|
|Google GenAI|imagen-3.0-generate-002|

* **Local**: Diffusers, Checkpoints
 - **Diffusers**: 現在、Diffusersモデルの場合、スキャンも可能で選択も可能だが、Diffusersモデルを使用した実際のイメージ生成は使用できない。(今後機能実装予定。)
 - **Checkpoints**: ComfyUI経由で画像を生成する。 ComfyUIディレクトリのmodels/checkpointsにモデルファイルが入っている必要がある。

Local Modelの場合、次のようなBase Modelを対応。
- Stable Diffusion 1.5
- Stable Diffusion 2.x
- Stable Diffusion XL 1.0
- Stable Diffusion 3 Medium
- FLUX.1 Schnell
- FLUX.1 Dev
- Stable Diffusion 3.5 Large
- Stable Diffusion 3.5 Medium
- Illustrious XL 1.0

* **LoRA**: Local Modelの場合、LORAを最大10個まで選択可能。ただし、ベースモデルに合うLORAを適用しなければならない。
* **VAE**: VAEをカスタマイズ。 Defaultにする場合、Checkpointsに内蔵されたVAEを使用。
* **Embedding**: 適用時にembedding:nameと同じ方法で入力して適用。
* **ControlNet**: 現在、ai-companionではインターフェースが実装されておらず、今後実装予定。
* **Refiner**: Stable Diffusion XL 1.0 Refinerモデルで、Refiner Start Stepでリファイナサンプリングが始まる段階を指定できる。

**生成オプション**

* **Positive Prompt**: 入力した単語に対応する画像を生成。
* **Negative Prompt**: 結果画像で見たくないものを除く。
* **Width, Height**: 画像の幅、高さを調整。
* **Recommended Resolution**

|Base Model|Recommended Resolution|
|----------|----------|
|Stable Diffusion 1.5 <br> Stable Diffusion 2.x|512x512 <br> 512x768 <br> 768x512|
|Stable Diffusion XL 1.0 <br> Stable Diffusion 3 Medium <br> FLUX.1 Schnell <br> FLUX.1 Dev <br> Stable Diffusion 3.5 Large <br> Stable Diffusion 3.5 Medium|1024x1024 <br> 896x1152 <br> 1152x896 <br> 832x1216 <br> 1216x832 <br> 768x1344 <br> 1344x768 <br> 640x1536 <br> 1536x640|
|Illustrious XL 1.0|1536x1536 <br> 1248x1824 <br> 1824x1248|

* **generation Steps**: AIがノイジングされたイメージを復旧する際、何段階のステップをかけてイメージを復旧させるかを決定する値で、値が高いほど生成までかかる時間が長くなる。
* **Denoise Strength**: ノイズの強度を調整。

**Advanced Settings**

* **Sampler**: サンプリング方式が異なると、同じプロンプトでも異なる結果が得られる。
* **Scheduler**: スケジューラが異なる場合、同じサンプラーと同じプロンプトでも異なる結果が得られる。
* **CFG Scale**: CFG値が高いほどプロンプトの説明によく従い、低いほど創意的にイメージを生成する。
* **Seed**: 生成過程で使用される乱数の初期値。
* **Clip Skip**: 画像生成プロセスの一部をスキップする機能。
* **Batch Size**: 1回の実行で生成する画像の数。

**Image to Image**<br>イメージに変化を与えることができる。 Inpaintを使用してマスク区間に対してのみ変化を与えることができる。

### Storyteller
LLMを活用してテキストを生成。 Chatbotとは異なり、小説などの文章作成にUIが最適化される。（現在未完成。）

**対応モデル**<br>Chatbot段落内の対応モデルを参照。

**ハイパーパラメータ設定**<br>Chatbot段落内のハイパーパラメータ設定を参照。<br>ハイパーパラメータの作動原理についてよく分からない場合は、デフォルトにしておくことをお勧めします。

### Video Generation
Coming Soon

### Audio Generation
Coming Soon

### Translator
翻訳モデルを活用した多言語翻訳。 画像ファイルまたはpdfファイルをアップロードしてテキストを抽出した後、該当テキストを翻訳に活用することも可能。

## 準備

- **仮想環境設定**

```shell
# conda (おすすめ!)
# Python 3.10
conda create -n ai-companion python=3.10
# Python 3.11
conda create -n ai-companion python=3.11
# Python 3.12
conda create -n ai-companion python=3.12
conda activate ai-companion
# venv
python3 -m venv venv
# MacOS、Linux環境
source venv/bin/activate 
# Windows環境
source venv/Scripts/activate 
```

- **Windows環境**
```shell
pip install -r requirements_windows_amd64.txt
```

- **macOS環境(Apple Siliconを搭載したMac)**
```zsh
pip install -r requirements_macos_arm64.txt
```
(Intel Macはrequirements.txtでインストール！)

- **Linux環境**
```bash
pip install -r requirements_linux.txt
```

### Macでxformersをインストールする際の注意事項！

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

## 実行

```zsh
python app.py
```

### Intel Mac互換に関して

Intel Macでは正常な動作を保障せず、ほとんどの機能をサポートしません。
したがって、最終バージョンの配布時点およびそれ以前にIntel Macに対するサポートが削除される可能性があります。
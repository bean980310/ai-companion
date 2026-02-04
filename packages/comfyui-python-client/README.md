# comfyui-python-client

ComfyUI REST/WebSocket API를 Python에서 다루기 쉽게 감싼 경량 클라이언트입니다.

- 기본 서버: `127.0.0.1:8188`
- 지원 기능: 프롬프트 실행/대기, 히스토리, 큐 제어, 이미지 업/다운로드, 시스템/모델/템플릿/사용자 API

## 요구사항

- Python `>= 3.12`
- 실행 중인 ComfyUI 서버

## 설치

패키지 루트(`packages/comfyui-python-client`)에서:

```bash
pip install -e .
```

## 빠른 시작

```python
from comfy_sdk import ComfyUI
import json

client = ComfyUI(host="127.0.0.1", port=8188)

# 1) 연결 확인
stats = client.system.stats()
print(stats)

# 2) 워크플로우 전송
with open("workflow_api.json", "r", encoding="utf-8") as f:
    workflow = json.load(f)

res = client.prompt.send(workflow)
print("prompt_id:", res.prompt_id)

# 3) 완료 대기 후 히스토리 조회
history = client.prompt.wait(res.prompt_id)
print(history)
```

## API 개요

`ComfyUI` 인스턴스는 아래 리소스를 제공합니다.

- `client.prompt`
  - `send(workflow)`
  - `retrieve(prompt_id)`
  - `wait(prompt_id)`
  - `history()`, `delete(prompt_id)`, `clear()`
- `client.images`
  - `upload(data, name, overwrite=False)`
  - `download(filename, subfolder="", folder_type="output")`
  - `upload_mask(data, name, original_ref, overwrite=False, mask_type="mask")`
  - `metadata(filename, subfolder="", folder_type="output")`
- `client.system`
  - `stats()`, `extensions()`, `embeddings()`, `features()`
  - `free(unload_models=False, free_memory=False)`
- `client.queue`
  - `status()`, `interrupt()`, `clear()`
- `client.models`
  - `list(folder=None)` (`checkpoints`, `loras` 등)
- `client.templates`
  - `list()`
- `client.users`
  - `list()`, `create(username)`
- `client.userdata`
  - `get(file)`, `move(file, dest)`

## 이미지 업로드 예시

```python
from comfy_sdk import ComfyUI

client = ComfyUI()

with open("input.png", "rb") as f:
    image_bytes = f.read()

result = client.images.upload(image_bytes, "input.png", overwrite=True)
print(result)
```

## 참고

- `prompt.wait(prompt_id)`는 내부적으로 WebSocket(`ws://<host>:<port>/ws`)을 사용합니다.
- ComfyUI가 실행 중이 아니면 요청이 실패하므로 먼저 서버 상태를 확인하세요.

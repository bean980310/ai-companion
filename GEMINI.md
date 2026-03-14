# AI Companion for Local Machines - Gemini Context

## Project Overview

**AI Companion** is a local and API-driven generative AI platform featuring chatbot interactions (via LLMs) and multimedia generation (images, etc.). It emphasizes persona-based interactions, allowing users to talk with and collaborate with various AI characters.

### Key Technologies & Architecture
The project architecture is composed of a Python backend with Gradio, alongside a separate modern web frontend and integrated local inference engines.

*   **Backend / Main App (Python):** 
    *   **Framework:** Gradio (entry point is `app.py`).
    *   **Features:** Chat, Image Generation, Storyteller, Audio (TTS/STT), Translation, and MCP Client/Tools.
    *   **Model Support:** Connects to cloud APIs (OpenAI, Anthropic, Google Gemini) and supports local inference via Transformers, GGUF, MLX, and vLLM.
    *   **Image Generation:** Backed by an integrated **ComfyUI** instance (`ComfyUI/` directory) for checkpoints, LoRAs, and Diffusers.
*   **Web Frontend (`ai-companion-web/`):** 
    *   A modern Single Page Application (SPA) built with **React 19**, **TypeScript**, and **Vite**.
    *   **Styling:** Tailwind CSS v4.
    *   **State Management & Data:** Zustand, React Query (@tanstack/react-query).
    *   **Routing:** React Router DOM.
*   **Submodules & Integrations:** 
    *   Integrates with local packages (e.g., `packages/ai-companion-core`, `packages/comfyui-python-client`).

## Building and Running

### 1. Main Application (Python/Gradio)

**Environment Setup:**
Use `conda` (recommended), `uv`, or standard `venv` with Python 3.10, 3.11, or 3.12.

**Installation:**
Run the OS-specific installer script located in the root directory:
*   **Windows:** `.\installer_windows_amd64.bat` or `.\installer_windows_amd64.ps1`
*   **macOS (Apple Silicon):** `./installer_macos_arm64.sh`
*   **Linux:** `./installer_linux_amd64_cuda.sh`

**Running the App:**
```bash
# Ensure your virtual environment is activated
python app.py
```

### 2. Web Frontend (`ai-companion-web/`)

**Installation:**
```bash
cd ai-companion-web
npm install
```

**Development Server:**
```bash
npm run dev
```

**Build for Production:**
```bash
npm run build
```

## Development Conventions

*   **Python Code:**
    *   Modular structure: UI pages are separated in `src/pages/`, core functionality in `src/main/` and `src/common_blocks.py`.
    *   Gradio warnings and PyTorch warnings are selectively ignored in the entry point to keep logs clean.
*   **Web Frontend Code:**
    *   Strict TypeScript checking is enabled (`tsc -b`).
    *   Uses ESLint for linting (`npm run lint`).
    *   Internationalization supported via `i18next` and `react-i18next`.
*   **Cross-Platform Constraints:**
    *   Intel Macs are explicitly unsupported by the application.
    *   For Windows, WSL2 is highly recommended to leverage GPU acceleration natively (direct Windows execution disables GPU acceleration for some workloads).
    *   CUDA 12.4+ is generally required on Windows for the latest `xformers` support.

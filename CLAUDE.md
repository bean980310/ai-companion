# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Companion is a Gradio-based application that combines chatbot functionality with AI-powered content generation (images, videos, audio). The project integrates multiple AI frameworks and includes the ComfyUI submodule for image generation.

## Architecture

### Core Structure
- **Main Application**: Python Gradio app (`app.py`) with modular components in `src/`
- **ComfyUI Submodule**: Full ComfyUI installation for image generation (`ComfyUI/`)
- **Backend Components**: Specialized backends like Wan2.1 for video generation (`backend/`)
- **Model Storage**: Organized model directories (`models/`) with separate folders for different model types

### Key Components
- `src/main/` - Main UI components (chatbot, image generation, storyteller, TTS, etc.)
- `src/pipelines/` - Model pipeline handlers (LLM, diffusion, audio)
- `src/models/` - Model management and configuration
- `src/common/` - Shared utilities and database management
- `presets/` - Character and conversation presets
- `workflows/` - ComfyUI workflow definitions

## Development Commands

### Installation
Use platform-specific installers:
- macOS ARM64: `./installer_macos_arm64.sh`
- Linux CUDA: `./installer_linux_amd64_cuda.sh`
- Windows: `.\installer_windows_amd64.bat` or `.\installer_windows_amd64.ps1`

### Running the Application
```bash
python app.py
```

Optional arguments:
- `--debug` - Enable debug mode
- `--share` - Enable Gradio sharing
- `--port` - Specify port (default varies)
- `--mcp-server` - Enable MCP server functionality

### Testing

#### Main Application
No formal test suite - run application directly to test:
```bash
python app.py
```

#### ComfyUI Tests
```bash
cd ComfyUI
# Unit tests
python -m pytest tests-unit
# Inference tests
pytest tests/inference
```

#### Wan2.1 Backend Tests
```bash
cd backend/Wan2.1
./tests/test.sh /path/to/models <gpu_count>
```

### Code Quality

#### Linting (ComfyUI only)
```bash
cd ComfyUI
ruff check .
```

#### Code Formatting (Wan2.1)
```bash
cd backend/Wan2.1
make format
```

## Model Management

### Model Locations
- **LLM Models**: `models/llm/` (transformers, gguf, mlx subdirectories)
- **Diffusion Models**: `models/diffusion/` (checkpoints, loras, controlnet, etc.)
- **ComfyUI Models**: `ComfyUI/models/` (follows ComfyUI structure)

### Supported Model Types
- **LLM**: Transformers (HuggingFace), GGUF (llama.cpp), MLX (Apple Silicon)
- **Diffusion**: SD 1.5/2.x/XL/3.x, FLUX, Illustrious XL, with LoRA/ControlNet support
- **Audio**: TTS models in `models/tts/`

## Configuration

### Character Presets
- Located in `presets/` directory
- JSON format with character definitions and system prompts
- Character images stored in `assets/`

### ComfyUI Workflows
- Predefined workflows in `workflows/` directory
- JSON format compatible with ComfyUI API
- Covers txt2img, img2img, inpainting with various model configurations

### Database
- SQLite databases: `chat_history.db`, `persona_state.db`
- Managed through `src/common/database.py`

## Platform-Specific Notes

### macOS (Apple Silicon)
- MLX support enabled for local LLM inference
- xformers installation may require additional build tools
- No support for Intel Macs

### Windows
- CUDA 12.4+ recommended
- Legacy CUDA support (11.8, 12.1) with fixed PyTorch 2.4.0
- WSL2 installation option available

### Linux
- CUDA and ROCm support available
- Separate installation paths for CPU-only and GPU configurations

## Important Paths

### Configuration Files
- `src/common/args.py` - Command line argument handling
- `ComfyUI/pyproject.toml` - ComfyUI linting configuration
- `requirements/` - Platform-specific dependency lists

### Entry Points
- `app.py` - Main application entry point
- `src/gradio_ui.py` - Gradio interface creation
- `src/start_app/` - Application initialization

## Submodule Management

ComfyUI is included as a git submodule:
```bash
# Initialize/update submodules
git submodule init
git submodule update
# or clone with submodules
git clone --recursive https://github.com/bean980310/ai-companion.git
```
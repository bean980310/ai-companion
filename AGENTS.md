# Repository Guidelines

## Project Structure & Module Organization

The main application is a Gradio/FastAPI hybrid rooted at `app.py`, with Python feature code under `src/` (`src/pages/`, `src/main/`, `src/common/`, `src/mcp/`). UI assets for the Gradio app live in `html/` and `assets/`. The separate React frontend is in `ai-companion-web/`, with pages in `ai-companion-web/src/pages/`, API clients in `ai-companion-web/src/api/`, and Zustand stores in `ai-companion-web/src/stores/`. Model files, presets, and translations live in `models/`, `presets/`, and `translations/`. `ComfyUI/`, `backend/Wan2.1/`, and `vllm/` are upstream submodules; change them only intentionally.

## Build, Test, and Development Commands

Use Python 3.12 (`.python-version`) unless a platform installer says otherwise.

- `python app.py`: start the main local Gradio app on the configured port.
- `bash installer_macos_arm64.sh` or `bash installer_linux_amd64_cuda.sh`: install platform-specific Python dependencies.
- `cd ai-companion-web && npm install && npm run dev`: run the React frontend in Vite dev mode.
- `cd ai-companion-web && npm run build`: type-check and build the frontend bundle.
- `cd ai-companion-web && npm run lint`: run ESLint for TypeScript/React files.
- `cd ComfyUI && pytest`: run the bundled ComfyUI test suite when touching that submodule.

There is no dedicated root `pytest` target for `src/`; for core app changes, launch `python app.py` and exercise the affected tab or API flow.

## Coding Style & Naming Conventions

Python uses 4-space indentation, snake_case modules, and descriptive helper names. Keep Gradio page wiring close to the relevant feature module. In `ai-companion-web/`, use PascalCase for page/components (`ChatPage.tsx`), camelCase for hooks/stores (`chatStore.ts`), and follow the existing ESLint config in `ai-companion-web/eslint.config.js`. Avoid mixing generated assets, checkpoints, or logs into source directories.

## Testing Guidelines

Add or update tests alongside the code you touch. For submodules, follow their local conventions such as `test_*.py` under `ComfyUI/tests/` and `ComfyUI/tests-unit/`. For frontend changes, at minimum run `npm run lint` and `npm run build`. For root app changes, include manual verification notes in the PR.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects such as `Fix AttributeError...`, `Add oauth issuer`, and `Update README`. Keep the first line concise, capitalized, and focused on one change. PRs should describe impact, list verification steps, link related issues, and include screenshots for UI changes. Call out any submodule bump explicitly.

## Security & Configuration Tips

Do not commit secrets, model weights, local databases, or generated outputs. Runtime configuration is created under `~/.ai-companion/.env`; keep machine-specific values there, not in tracked files.

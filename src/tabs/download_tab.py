import os
import logging
import traceback

import gradio as gr
from huggingface_hub import HfApi

from src.tabs.main_tab import MainTab
from src.models.api_models import api_models
from src.models.known_hf_models import known_hf_models

from src.common.utils import download_model_from_hf, make_local_dir_name, get_all_local_models

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main_tab=MainTab()

def create_download_tab():
    with gr.Tab("Download"):
        with gr.Tabs():
            # Predefined 탭
            with gr.Tab("Predefined"):
                gr.Markdown("""### Predefined Models
                Select from a list of predefined models available for download.""")

                predefined_dropdown = gr.Dropdown(
                    label="Model Selection",
                    choices=sorted(known_hf_models),
                    value=known_hf_models[0] if known_hf_models else None,
                    info="Select a predefined model from the list."
                )

                # 다운로드 설정
                with gr.Row():
                    target_path = gr.Textbox(
                        label="Save Path",
                        placeholder="./models/my-model",
                        value="",
                        interactive=True,
                        info="Leave empty to use the default path."
                    )
                    use_auth = gr.Checkbox(
                        label="Authentication Required",
                        value=False,
                        info="Check if the model requires authentication."
                    )

                with gr.Column(visible=False) as auth_column_predefined:
                    hf_token = gr.Textbox(
                        label="HuggingFace Token",
                        placeholder="hf_...",
                        type="password",
                        info="Enter your HuggingFace token if authentication is required."
                    )

                # 다운로드 버튼과 진행 상태
                with gr.Row():
                    download_btn_predefined = gr.Button(
                        value="Start Download",
                        variant="primary",
                        scale=2
                    )
                    cancel_btn_predefined = gr.Button(
                        value="Cancel",
                        variant="stop",
                        scale=1,
                        interactive=False
                    )

                # 상태 표시
                download_status_predefined = gr.Markdown("")
                progress_bar_predefined = gr.Progress(track_tqdm=True)

                # 다운로드 결과와 로그
                with gr.Accordion("Download Details", open=False):
                    download_info_predefined = gr.TextArea(
                        label="Download Log",
                        interactive=False,
                        max_lines=10,
                        autoscroll=True
                    )

                # 이벤트 핸들러
                def toggle_auth_predefined(use_auth_val):
                    return gr.update(visible=use_auth_val)

                use_auth.change(
                    fn=toggle_auth_predefined,
                    inputs=[use_auth],
                    outputs=[auth_column_predefined]
                )

                def download_predefined_model(predefined_choice, target_dir, use_auth_val, token):
                    try:
                        repo_id = predefined_choice
                        if not repo_id:
                            download_status_predefined.update("❌ No model selected.")
                            return

                        model_type = main_tab.determine_model_type(repo_id)

                        download_status_predefined.update("🔄 Preparing to download...")
                        logger.info(f"Starting download for {repo_id}")

                        # 실제 다운로드 함수 호출 (비동기 처리를 원한다면 async 함수로 구현 필요)
                        result = download_model_from_hf(
                            repo_id,
                            target_dir or os.path.join("./models", model_type, make_local_dir_name(repo_id)),
                            model_type=model_type,
                            token=token if use_auth_val else None
                        )

                        download_status_predefined.update("✅ Download completed!" if "실패" not in result else "❌ Download failed.")
                        download_info_predefined.update(result)

                        # 다운로드 완료 후 모델 목록 업데이트
                        new_choices = sorted(api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"])
                        return gr.Dropdown.update(choices=new_choices)

                    except Exception as e:
                        logger.error(f"Error downloading model: {str(e)}")
                        download_status_predefined.update("❌ An error occurred during download.")
                        download_info_predefined.update(f"Error: {str(e)}\n{traceback.format_exc()}")

                download_btn_predefined.click(
                    fn=download_predefined_model,
                    inputs=[predefined_dropdown, target_path, use_auth, hf_token],
                    outputs=[download_status_predefined, download_info_predefined]
                )

            # Custom Repo ID 탭
            with gr.Tab("Custom Repo ID"):
                gr.Markdown("""### Custom Repository ID
                Enter a custom HuggingFace repository ID to download the model.""")

                custom_repo_id_box = gr.Textbox(
                    label="Custom Model ID",
                    placeholder="e.g., facebook/opt-350m",
                    info="Enter the HuggingFace model repository ID (e.g., organization/model-name)."
                )

                # 다운로드 설정
                with gr.Row():
                    target_path_custom = gr.Textbox(
                        label="Save Path",
                        placeholder="./models/custom-model",
                        value="",
                        interactive=True,
                        info="Leave empty to use the default path."
                    )
                    use_auth_custom = gr.Checkbox(
                        label="Authentication Required",
                        value=False,
                        info="Check if the model requires authentication."
                    )

                with gr.Column(visible=False) as auth_column_custom:
                    hf_token_custom = gr.Textbox(
                        label="HuggingFace Token",
                        placeholder="hf_...",
                        type="password",
                        info="Enter your HuggingFace token if authentication is required."
                    )

                # 다운로드 버튼과 진행 상태
                with gr.Row():
                    download_btn_custom = gr.Button(
                        value="Start Download",
                        variant="primary",
                        scale=2
                    )
                    cancel_btn_custom = gr.Button(
                        value="Cancel",
                        variant="stop",
                        scale=1,
                        interactive=False
                    )

                # 상태 표시
                download_status_custom = gr.Markdown("")
                progress_bar_custom = gr.Progress(track_tqdm=True)

                # 다운로드 결과와 로그
                with gr.Accordion("Download Details", open=False):
                    download_info_custom = gr.TextArea(
                        label="Download Log",
                        interactive=False,
                        max_lines=10,
                        autoscroll=True
                    )

                # 이벤트 핸들러
                def toggle_auth_custom(use_auth_val):
                    return gr.update(visible=use_auth_val)

                use_auth_custom.change(
                    fn=toggle_auth_custom,
                    inputs=[use_auth_custom],
                    outputs=[auth_column_custom]
                )

                def download_custom_model(custom_repo, target_dir, use_auth_val, token):
                    try:
                        repo_id = custom_repo.strip()
                        if not repo_id:
                            download_status_custom.update("❌ No repository ID entered.")
                            return

                        model_type = main_tab.determine_model_type(repo_id)

                        download_status_custom.update("🔄 Preparing to download...")
                        logger.info(f"Starting download for {repo_id}")

                        # 실제 다운로드 함수 호출 (비동기 처리를 원한다면 async 함수로 구현 필요)
                        result = download_model_from_hf(
                            repo_id,
                            target_dir or os.path.join("./models", model_type, make_local_dir_name(repo_id)),
                            model_type=model_type,
                            token=token if use_auth_val else None
                        )

                        download_status_custom.update("✅ Download completed!" if "실패" not in result else "❌ Download failed.")
                        download_info_custom.update(result)

                        # 다운로드 완료 후 모델 목록 업데이트
                        new_choices = sorted(api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"])
                        return gr.Dropdown.update(choices=new_choices)

                    except Exception as e:
                        logger.error(f"Error downloading model: {str(e)}")
                        download_status_custom.update("❌ An error occurred during download.")
                        download_info_custom.update(f"Error: {str(e)}\n{traceback.format_exc()}")

                download_btn_custom.click(
                    fn=download_custom_model,
                    inputs=[custom_repo_id_box, target_path_custom, use_auth_custom, hf_token_custom],
                    outputs=[download_status_custom, download_info_custom]
                )

            # Hub 탭
            with gr.Tab("Hub"):
                gr.Markdown("""### Hub Models
                Search and download models directly from HuggingFace Hub.""")

                with gr.Row():
                    search_box_hub = gr.Textbox(
                        label="Search",
                        placeholder="Enter model name, tag, or keyword...",
                        scale=4
                    )
                    search_btn_hub = gr.Button("Search", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        model_type_filter_hub = gr.Dropdown(
                            label="Model Type",
                            choices=["All", "Text Generation", "Vision", "Audio", "Other"],
                            value="All"
                        )
                        language_filter_hub = gr.Dropdown(
                            label="Language",
                            choices=["All", "Korean", "English", "Chinese", "Japanese", "Multilingual"],
                            value="All"
                        )
                        library_filter_hub = gr.Dropdown(
                            label="Library",
                            choices=["All", "Transformers", "GGUF", "MLX"],
                            value="All"
                        )
                    with gr.Column(scale=3):
                        model_list_hub = gr.Dataframe(
                            headers=["Model ID", "Description", "Downloads", "Likes"],
                            label="Search Results",
                            interactive=False
                        )

                with gr.Row():
                    selected_model_hub = gr.Textbox(
                        label="Selected Model",
                        interactive=False
                    )

                # 다운로드 설정
                with gr.Row():
                    target_path_hub = gr.Textbox(
                        label="Save Path",
                        placeholder="./models/hub-model",
                        value="",
                        interactive=True,
                        info="Leave empty to use the default path."
                    )
                    use_auth_hub = gr.Checkbox(
                        label="Authentication Required",
                        value=False,
                        info="Check if the model requires authentication."
                    )

                with gr.Column(visible=False) as auth_column_hub:
                    hf_token_hub = gr.Textbox(
                        label="HuggingFace Token",
                        placeholder="hf_...",
                        type="password",
                        info="Enter your HuggingFace token if authentication is required."
                    )

                # 다운로드 버튼과 진행 상태
                with gr.Row():
                    download_btn_hub = gr.Button(
                        value="Start Download",
                        variant="primary",
                        scale=2
                    )
                    cancel_btn_hub = gr.Button(
                        value="Cancel",
                        variant="stop",
                        scale=1,
                        interactive=False
                    )

                # 상태 표시
                download_status_hub = gr.Markdown("")
                progress_bar_hub = gr.Progress(track_tqdm=True)

                # 다운로드 결과와 로그
                with gr.Accordion("Download Details", open=False):
                    download_info_hub = gr.TextArea(
                        label="Download Log",
                        interactive=False,
                        max_lines=10,
                        autoscroll=True
                    )

                # 이벤트 핸들러
                def toggle_auth_hub(use_auth_val):
                    return gr.update(visible=use_auth_val)

                use_auth_hub.change(
                    fn=toggle_auth_hub,
                    inputs=[use_auth_hub],
                    outputs=[auth_column_hub]
                )

                def search_models_hub(query, model_type, language, library):
                    """Search models on HuggingFace Hub"""
                    try:
                        api = HfApi()
                        filter_str = ""
                        if model_type != "All":
                            filter_str += f"task_{model_type.lower().replace(' ', '_')}"
                        if language != "All":
                            if filter_str:
                                filter_str += " AND "
                            filter_str += f"language_{language.lower()}"
                        if library != "All":
                            filter_str += f"library_{library.lower()}"

                        models = api.list_models(
                            filter=filter_str if filter_str else None,
                            limit=100,
                            sort="lastModified",
                            direction=-1
                        )

                        filtered_models = [model for model in models if query.lower() in model.id.lower()]

                        model_list_data = []
                        for model in filtered_models:
                            description = model.cardData.get('description', '') if model.cardData else 'No description available.'
                            short_description = (description[:100] + "...") if len(description) > 100 else description
                            model_list_data.append([
                                model.id,
                                short_description,
                                model.downloads,
                                model.likes
                            ])
                        return model_list_data
                    except Exception as e:
                        logger.error(f"Error searching models: {str(e)}\n{traceback.format_exc()}")
                        return [["Error occurred", str(e), "", ""]]

                def select_model_hub(evt: gr.SelectData, data):
                    """Select model from dataframe"""
                    selected_model_id = data.at[evt.index[0], "Model ID"] if evt.index else ""
                    return selected_model_id

                def download_hub_model(model_id, target_dir, use_auth_val, token):
                    try:
                        if not model_id:
                            download_status_hub.update("❌ No model selected.")
                            return

                        model_type = main_tab.determine_model_type(model_id)

                        download_status_hub.update("🔄 Preparing to download...")
                        logger.info(f"Starting download for {model_id}")

                        # 실제 다운로드 함수 호출 (비동기 처리를 원한다면 async 함수로 구현 필요)
                        result = download_model_from_hf(
                            model_id,
                            target_dir or os.path.join("./models", model_type, make_local_dir_name(model_id)),
                            model_type=model_type,
                            token=token if use_auth_val else None
                        )

                        download_status_hub.update("✅ Download completed!" if "실패" not in result else "❌ Download failed.")
                        download_info_hub.update(result)

                        # 다운로드 완료 후 모델 목록 업데이트
                        new_choices = sorted(api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"])
                        return gr.Dropdown.update(choices=new_choices)

                    except Exception as e:
                        logger.error(f"Error downloading model: {str(e)}")
                        download_status_hub.update("❌ An error occurred during download.")
                        download_info_hub.update(f"Error: {str(e)}\n{traceback.format_exc()}")

                search_btn_hub.click(
                    fn=search_models_hub,
                    inputs=[search_box_hub, model_type_filter_hub, language_filter_hub, library_filter_hub],
                    outputs=model_list_hub
                )

                model_list_hub.select(
                    fn=select_model_hub,
                    inputs=[model_list_hub],
                    outputs=[selected_model_hub]
                )

                download_btn_hub.click(
                    fn=download_hub_model,
                    inputs=[selected_model_hub, target_path_hub, use_auth_hub, hf_token_hub],
                    outputs=[download_status_hub, download_info_hub]
                )
                
    return
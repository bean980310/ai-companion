import os
import traceback

import gradio as gr
from huggingface_hub import HfApi
from gradio_huggingfacehub_search import HuggingfaceHubSearch

from .. import logger

from ..hub import TASKS, LIBRARIES, LANGUAGES_HUB

from ..models.provider_llm_models import llm_api_models
from ..models.known_hf_models import known_hf_models

from ..common.utils import download_model_from_hf, make_local_dir_name, get_all_local_models
from src.main.chatbot.chatbot import Chatbot

# 로깅 설정

chat_bot=Chatbot()

def create_download_tab():
    with gr.Column(elem_classes="tab-container") as download_container:
        with gr.Row(elem_classes="model-container"):
            gr.Markdown("### Download Center")
        with gr.Tabs(elem_classes="chat-interface"):
            # Predefined 탭
            with gr.Tab("Predefined", elem_classes='tab-container'):
                gr.Markdown("""### Predefined Models
                Select from a list of predefined models available for download.""")

                predefined_dropdown = gr.Dropdown(
                    label="Model Selection",
                    choices=sorted(known_hf_models),
                    value=known_hf_models[0] if known_hf_models else None,
                    info="Select a predefined model from the list.",
                    elem_classes="model-dropdown"
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
                with gr.Accordion("Download Details", open=False, elem_classes="accordion-container"):
                    download_info_predefined = gr.TextArea(
                        label="Download Log",
                        interactive=False,
                        max_lines=10,
                        autoscroll=True
                    )

                # 이벤트 핸들러
                @use_auth.change(inputs=[use_auth], outputs=[auth_column_predefined])
                def toggle_auth_predefined(use_auth_val):
                    """
                    Toggle authentication visibility based on the checkbox value.
                    Args:
                        use_auth_val (bool): Value of the checkbox.
                    """
                    return gr.update(visible=use_auth_val)
                
                @download_btn_predefined.click(inputs=[predefined_dropdown, target_path, use_auth, hf_token], outputs=[download_status_predefined, download_info_predefined])
                def download_predefined_model(predefined_choice, target_dir, use_auth_val, token):
                    """
                    Download a predefined model.
                    Args:
                        predefined_choice (str): Predefined model choice.
                        target_dir (str): Target directory for saving the model.
                        use_auth_val (bool): Value of the authentication checkbox.
                        token (str): HuggingFace token for authentication.
                    """
                    try:
                        repo_id = predefined_choice
                        if not repo_id:
                            download_status_predefined.update("❌ No model selected.")
                            return

                        model_type = chat_bot.determine_model_type(repo_id)

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
                        new_choices = sorted(llm_api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"])
                        return gr.Dropdown.update(choices=new_choices)

                    except Exception as e:
                        logger.error(f"Error downloading model: {str(e)}")
                        download_status_predefined.update("❌ An error occurred during download.")
                        download_info_predefined.update(f"Error: {str(e)}\n{traceback.format_exc()}")

            # Custom Repo ID 탭
            with gr.Tab("Custom Repo ID"):
                gr.Markdown("""### Custom Repository ID
                Enter a custom HuggingFace repository ID to download the model.""")

                custom_repo_id_box = HuggingfaceHubSearch(
                    label="Custom Model ID",
                    placeholder="e.g., facebook/opt-350m",
                    # info="Enter the HuggingFace model repository ID (e.g., organization/model-name).",
                    search_type="model"
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
                with gr.Accordion("Download Details", open=False, elem_classes="accordion-container"):
                    download_info_custom = gr.TextArea(
                        label="Download Log",
                        interactive=False,
                        max_lines=10,
                        autoscroll=True
                    )

                # 이벤트 핸들러
                @use_auth_custom.change(inputs=[use_auth_custom], outputs=[auth_column_custom])
                def toggle_auth_custom(use_auth_val):
                    """
                    Toggle authentication visibility based on the checkbox value.
                    Args:
                        use_auth_val (bool): Value of the checkbox.
                    """
                    return gr.update(visible=use_auth_val)
                
                @download_btn_custom.click(inputs=[custom_repo_id_box, target_path_custom, use_auth_custom, hf_token_custom], outputs=[download_status_custom, download_info_custom])
                def download_custom_model(custom_repo, target_dir, use_auth_val, token):
                    """
                    Download a custom model.
                    Args:
                        custom_repo (str): Custom model repository ID.
                        target_dir (str): Target directory for saving the model.
                        use_auth_val (bool): Value of the authentication checkbox.
                        token (str): HuggingFace token for authentication.
                    """
                    try:
                        repo_id = custom_repo.strip()
                        if not repo_id:
                            download_status_custom.update("❌ No repository ID entered.")
                            return

                        model_type = chat_bot.determine_model_type(repo_id)

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
                        new_choices = sorted(llm_api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"])
                        return gr.Dropdown.update(choices=new_choices)

                    except Exception as e:
                        logger.error(f"Error downloading model: {str(e)}")
                        download_status_custom.update("❌ An error occurred during download.")
                        download_info_custom.update(f"Error: {str(e)}\n{traceback.format_exc()}")

            # Hub 탭
            with gr.Tab("Hub"):
                gr.Markdown("""### Hub Models
                Search and download models directly from HuggingFace Hub.""")

                with gr.Row():
                    search_box_hub = gr.Textbox(
                        label="Search",
                        placeholder="Enter model name, tag, or keyword...",
                        scale=4,
                    )
                    search_btn_hub = gr.Button("Search", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        model_type_filter_hub = gr.Dropdown(
                            label="Tasks",
                            choices=list(dict.fromkeys(TASKS)),
                            value="None"
                        )
                        language_filter_hub = gr.Dropdown(
                            label="Language",
                            choices=list(dict.fromkeys(LANGUAGES_HUB)),
                            multiselect=True,
                            value=[]
                        )
                        library_filter_hub = gr.Dropdown(
                            label="Library",
                            choices=list(dict.fromkeys(LIBRARIES)),
                            multiselect=True,
                            value=[]
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
                with gr.Accordion("Download Details", open=False, elem_classes="accordion-container"):
                    download_info_hub = gr.TextArea(
                        label="Download Log",
                        interactive=False,
                        max_lines=10,
                        autoscroll=True
                    )

                # 이벤트 핸들러
                @use_auth_hub.change(inputs=[use_auth_hub], outputs=[auth_column_hub])
                def toggle_auth_hub(use_auth_val):
                    """
                    Toggle authentication visibility based on the checkbox value.
                    Args:
                        use_auth_val (bool): Value of the checkbox.
                    """
                    return gr.update(visible=use_auth_val)
                
                @gr.on(triggers=[search_btn_hub.click, search_box_hub.submit], inputs=[search_box_hub, model_type_filter_hub, language_filter_hub, library_filter_hub], outputs=model_list_hub)
                def search_models_hub(query, model_type, language, library):
                    """
                    Search models on HuggingFace Hub
                    Args:
                        query (str): Search query.
                        model_type (str): Model type filter.
                        language (list): Language filter.
                        library (list): Library filter.
                    """
                    try:
                        api = HfApi()
                        filter_str = ""
                        task_filter = model_type
                        lib_filter = []
                        lang_filter = []
                        if model_type == "None":
                            return None
                        else:
                            filter_str += f"task_{model_type}"
                        for i in range(len(library)):
                            lib_filter.append(library[i])
                        for i in range(len(language)):
                            lang_filter.append(language[i])

                        models = api.list_models(
                            filter=[task_filter, lib_filter, lang_filter],
                            search=search_box_hub.value,
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
                    
                @model_list_hub.select(inputs=[model_list_hub], outputs=[selected_model_hub])
                def select_model_hub(evt: gr.SelectData, data):
                    """
                    Select model from dataframe
                    Args:
                        evt (gr.SelectData): Event data.
                        data (gr.Dataframe): Dataframe.
                    """
                    selected_model_id = data.at[evt.index[0], "Model ID"] if evt.index else ""
                    return selected_model_id
                    
                @download_btn_hub.click(inputs=[selected_model_hub, target_path_hub, use_auth_hub, hf_token_hub], outputs=[download_status_hub, download_info_hub])
                def download_hub_model(model_id, target_dir, use_auth_val, token):
                    """
                    Download a model from HuggingFace Hub.
                    Args:
                        model_id (str): Model ID.
                        target_dir (str): Target directory for saving the model.
                        use_auth_val (bool): Value of the authentication checkbox.
                        token (str): HuggingFace token for authentication.
                    """
                    try:
                        if not model_id:
                            download_status_hub.update("❌ No model selected.")
                            return

                        model_type = chat_bot.determine_model_type(model_id)

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
                        new_choices = sorted(llm_api_models + get_all_local_models()["transformers"] + get_all_local_models()["gguf"] + get_all_local_models()["mlx"])
                        return gr.Dropdown.update(choices=new_choices)

                    except Exception as e:
                        logger.error(f"Error downloading model: {str(e)}")
                        download_status_hub.update("❌ An error occurred during download.")
                        download_info_hub.update(f"Error: {str(e)}\n{traceback.format_exc()}")
                
    return download_container
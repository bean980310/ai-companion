import gradio as gr
import numpy as np
import random
from typing import Any, List, Sequence, Callable
from PIL import Image

from ..start_app import app_state, ui_component
from .chatbot import Chatbot, ChatbotComponent, ChatbotMain, chat_bot, chat_main
from .image_generation import diff_main, ImageGeneration, image_gen, DiffusionComponent, DiffusionMain
from .sidebar import create_sidebar
from .container import create_body_container
from ..common.database import get_existing_sessions
from ..common.character_info import characters
from ..common.html import show_confetti
from ..common.translations import translation_manager, _
from ..api.comfy_api import ComfyUIClient
from .header import HeaderUIComponent
from ..characters import PersonaSpeechManager

from .. import __version__

from presets import (
    AI_ASSISTANT_PRESET, 
    SD_IMAGE_GENERATOR_PRESET, 
    MINAMI_ASUKA_PRESET, 
    MAKOTONO_AOI_PRESET, 
    AINO_KOITO_PRESET, 
    ARIA_PRINCESS_FATE_PRESET, 
    ARIA_PRINCE_FATE_PRESET,
    WANG_MEI_LING_PRESET,
    MISTY_LANE_PRESET,
    LILY_EMPRESS_PRESET,
    CHOI_YUNA_PRESET,
    CHOI_YURI_PRESET
    )

def create_main_container(demo: gr.Blocks, client: ComfyUIClient = ComfyUIClient()):
    with gr.Column(elem_classes="main-container"):    
        header_ui_component = HeaderUIComponent.create_header_container()

        sidebar, tab_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_side, diff_side, storyteller_side, storytelling_model_type_dropdown, storytelling_model_dropdown, storytelling_api_key_text, storytelling_lora_dropdown, tts_side, tts_model_type_dropdown, tts_model_dropdown, translate_side = create_sidebar()

        chat_container, diff_container, story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider, tts_container, translate_container, download_container = create_body_container()

        with gr.Row():
            gr.Markdown(f"Version: {__version__}")
                            
        reset_modal, single_reset_content, all_reset_content, cancel_btn, confirm_btn = chat_bot.create_reset_confirm_modal()
        delete_modal, delete_message, delete_cancel_btn, delete_confirm_btn = chat_bot.create_delete_session_modal()

    title = header_ui_component.title
    settings_button = header_ui_component.settings_button
    language_dropdown = header_ui_component.language_dropdown

    session_select_dropdown = chat_side.session.session_select_dropdown
    chat_title_box = chat_side.session.chat_title_box
    add_session_icon_btn = chat_side.session.add_session_icon_btn
    delete_session_icon_btn = chat_side.session.delete_session_icon_btn

    text_model_type_dropdown = chat_side.model.model_type_dropdown
    text_model_dropdown = chat_side.model.model_dropdown
    text_api_key_text = chat_side.model.api_key_text
    text_lora_dropdown = chat_side.model.lora_dropdown

    system_message_accordion = chat_container.main_panel.system_message_accordion
    system_message_box = chat_container.main_panel.system_message_box
    chatbot = chat_container.main_panel.chatbot
    msg = chat_container.main_panel.msg
    multimodal_msg = chat_container.main_panel.multimodal_msg

    profile_image = chat_container.side_panel.profile_image
    character_dropdown = chat_container.side_panel.character_dropdown

    text_advanced_settings = chat_container.side_panel.advanced_setting

    text_seed_input = chat_container.side_panel.seed_input
    text_temperature_slider = chat_container.side_panel.temperature_slider
    text_top_k_slider = chat_container.side_panel.top_k_slider
    text_top_p_slider = chat_container.side_panel.top_p_slider
    text_repetition_penalty_slider = chat_container.side_panel.repetition_penalty_slider
    preset_dropdown = chat_container.side_panel.preset_dropdown
    change_preset_button = chat_container.side_panel.change_preset_button
    reset_btn = chat_container.side_panel.reset_btn
    reset_all_btn = chat_container.side_panel.reset_all_btn

    status_text = chat_container.status_bar.status_text
    image_info = chat_container.status_bar.image_info
    session_select_info = chat_container.status_bar.session_select_info

    diffusion_model_type_dropdown = diff_side.model.model_type_dropdown
    diffusion_model_dropdown = diff_side.model.model_dropdown
    diffusion_api_key_text = diff_side.model.api_key_text

    diffusion_refiner_model_dropdown = diff_side.refiner.refiner_model_dropdown
    diffusion_refiner_start = diff_side.refiner.refiner_start
    diffusion_with_refiner_image_to_image_start = diff_side.refiner.with_refiner_image_to_image_start

    diffusion_lora_multiselect = diff_side.lora.lora_multiselect
    diffusion_lora_text_encoder_sliders = diff_side.lora.lora_text_encoder_sliders
    diffusion_lora_unet_sliders = diff_side.lora.lora_unet_sliders

    image_to_image_mode = diff_container.image_panel.image_to_image_mode
    image_to_image_input = diff_container.image_panel.image_to_image_input
    image_inpaint_input = diff_container.image_panel.image_inpaint_input
    image_inpaint_masking = diff_container.image_panel.image_inpaint_masking

    blur_radius_slider = diff_container.image_panel.blur_radius_slider
    blur_expansion_radius_slider = diff_container.image_panel.blur_expansion_radius_slider
    denoise_strength_slider = diff_container.image_panel.denoise_strength_slider

    positive_prompt_input = diff_container.main_panel.positive_prompt_input
    negative_prompt_input = diff_container.main_panel.negative_prompt_input
    style_dropdown = diff_container.main_panel.style_dropdown

    width_slider = diff_container.main_panel.width_slider
    height_slider = diff_container.main_panel.height_slider
    generation_step_slider = diff_container.main_panel.generation_step_slider
    random_prompt_btn = diff_container.main_panel.random_prompt_btn
    generate_btn = diff_container.main_panel.generate_btn
    gallery = diff_container.main_panel.gallery

    diffusion_advanced_settings = diff_container.side_panel.advanced_setting

    sampler_dropdown = diff_container.side_panel.sampler_dropdown
    scheduler_dropdown = diff_container.side_panel.scheduler_dropdown
    cfg_scale_slider = diff_container.side_panel.cfg_scale_slider

    diffusion_seed_input = diff_container.side_panel.seed_input
    random_seed_checkbox = diff_container.side_panel.random_seed_checkbox
    vae_dropdown = diff_container.side_panel.vae_dropdown

    clip_skip_slider = diff_container.side_panel.clip_skip_slider
    enable_clip_skip_checkbox = diff_container.side_panel.enable_clip_skip_checkbox
    clip_g_checkbox = diff_container.side_panel.clip_g_checkbox

    batch_size_input = diff_container.side_panel.batch_size_input
    batch_count_input = diff_container.side_panel.batch_count_input

    image_history = diff_container.history_panel.image_history

    # 아래는 변경 이벤트 등록
    def apply_session_immediately(chosen_sid):
        """
        메인탭에서 세션이 선택되면 바로 main_tab.apply_session을 호출해 세션 적용.
        """
        return chat_bot.apply_session(chosen_sid)
    
    session_select_dropdown.change(
        fn=chat_main.apply_session_immediately,
        inputs=[session_select_dropdown],
        outputs=[app_state.history_state, app_state.session_id_state, session_select_info]
    ).then(
        fn=chat_bot.filter_messages_for_chatbot,
        inputs=[app_state.history_state],
        outputs=[chatbot]
    )

    def init_session_dropdown(sessions: list[str] | Sequence[str]) -> gr.update:
        if not sessions:
            return gr.update(choices=[], value=None)
        return gr.update(choices=sessions, value=sessions[0])

    @add_session_icon_btn.click(inputs=[character_dropdown, app_state.selected_language_state, app_state.speech_manager_state, app_state.history_state],outputs=[app_state.session_id_state, app_state.history_state, session_select_dropdown, session_select_info, chatbot])
    def create_and_apply_session(chosen_character: str, chosen_language: str, speech_manager_state: PersonaSpeechManager, history_state):
        """
        현재 캐릭터/언어에 맞춰 시스템 메시지를 가져온 뒤,
        새 세션을 생성합니다.
        """
        # 1) SpeechManager 인스턴스 획득
        speech_manager = speech_manager_state  # 전역 gr.State로 관리 중인 persona_speech_manager

        # 2) 캐릭터+언어를 설정하고 시스템 메시지 가져오기
        speech_manager.set_character_and_language(chosen_character, chosen_language)
        new_system_msg = speech_manager.get_system_message()

        # 3) DB에 기록할 새 세션 만들기
        new_sid, info, new_history = chat_bot.create_new_session(new_system_msg, chosen_character)

        sessions = get_existing_sessions()
        return [
            new_sid,
            new_history,
            gr.update(choices=sessions, value=new_sid),
            info,
            chat_bot.filter_messages_for_chatbot(new_history)
        ]
    
    # 이벤트 핸들러
    @delete_session_icon_btn.click(inputs=[session_select_dropdown, app_state.session_id_state], outputs=[delete_modal, delete_message])
    def show_delete_confirm(selected_sid: str, current_sid: str):
        """삭제 확인 모달 표시"""
        if not selected_sid:
            return gr.update(visible=True), "삭제할 세션을 선택하세요."
        if selected_sid == current_sid:
            return gr.update(visible=True), f"현재 활성 세션 '{selected_sid}'은(는) 삭제할 수 없습니다."
        return gr.update(visible=True), f"세션 '{selected_sid}'을(를) 삭제하시겠습니까?"
            
    def delete_selected_session(chosen_sid: str):
        # 선택된 세션을 삭제 (주의: None 또는 ""인 경우 처리)
        result_msg, _, updated_dropdown = chat_bot.delete_session(chosen_sid, "demo_session")
        return result_msg, updated_dropdown

    # 취소 버튼
    delete_cancel_btn.click(
        fn=lambda: (gr.update(visible=False), ""),
        outputs=[delete_modal, delete_message]
    )

    # 삭제 확인 버튼
    delete_confirm_btn.click(
        fn=chat_bot.delete_session,
        inputs=[session_select_dropdown, app_state.session_id_state],
        outputs=[delete_modal, delete_message, session_select_dropdown]
    ).then(
        fn=chat_bot.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown]
    )
    
    demo.load(None, None, None).then(
        fn=lambda evt: (gr.update(visible=False), "") if evt.key == "Escape" else (gr.update(), ""),
        inputs=[],
        outputs=[delete_modal, delete_message]
    )
    
    # 시드 입력과 상태 연결
    text_seed_input.change(
        fn=lambda seed: seed if seed is not None else 42,
        inputs=[text_seed_input],
        outputs=[app_state.seed_state]
    )
    text_temperature_slider.change(
        fn=lambda temp: temp if temp is not None else 0.6,
        inputs=[text_temperature_slider],
        outputs=[app_state.temperature_state]
    )
    text_top_k_slider.change(
        fn=lambda top_k: top_k if top_k is not None else 20,
        inputs=[text_top_k_slider],
        outputs=[app_state.top_k_state]
    )
    text_top_p_slider.change(
        fn=lambda top_p: top_p if top_p is not None else 0.9,
        inputs=[text_top_p_slider],
        outputs=[app_state.top_p_state]
    )
    text_repetition_penalty_slider.change(
        fn=lambda repetition_penalty: repetition_penalty if repetition_penalty is not None else 1.1,
        inputs=[text_repetition_penalty_slider],
        outputs=[app_state.repetition_penalty_state]
    )
            
    # 프리셋 변경 버튼 클릭 시 호출될 함수 연결
    gr.on(
        triggers=[character_dropdown.change, change_preset_button.click],
        fn=chat_bot.handle_change_preset,
        inputs=[preset_dropdown, app_state.history_state, app_state.selected_language_state],
        outputs=[app_state.history_state, system_message_box, profile_image]
    )

    character_dropdown.change(
        fn=chat_bot.update_system_message_and_profile,
        inputs=[character_dropdown, language_dropdown, app_state.session_id_state],
        outputs=[system_message_box, profile_image, preset_dropdown]
    )
    
    diffusion_model_dropdown.change(
        fn=lambda selected_model: (
            image_gen.toggle_diffusion_api_key_visibility(selected_model)
        ),
        inputs=[diffusion_model_dropdown],
        outputs=[diffusion_api_key_text]
    )
    
    demo.load(
        fn=lambda selected_model: (
            image_gen.toggle_diffusion_api_key_visibility(selected_model)
        ),
        inputs=[diffusion_model_dropdown],
        outputs=[diffusion_api_key_text]
    )
        
    # 모델 선택 변경 시 가시성 토글
    text_model_dropdown.change(
        fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
            chat_bot.toggle_multimodal_msg_input_visibility(selected_model),
            chat_bot.toggle_standard_msg_input_visibility(selected_model)
        ),
        inputs=[text_model_dropdown],
        outputs=[text_api_key_text, text_lora_dropdown, multimodal_msg, msg]
    )
    
    storytelling_model_dropdown.change(
        fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
        ),
        inputs=[storytelling_model_dropdown],
        outputs=[storytelling_api_key_text, storytelling_lora_dropdown]
    )
    
    demo.load(
         fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
        ),
        inputs=[storytelling_model_dropdown],
        outputs=[storytelling_api_key_text, storytelling_lora_dropdown]
    )
    storytelling_model_type_dropdown.change(
        fn=chat_bot.update_model_list,
        inputs=[storytelling_model_type_dropdown],
        outputs=[storytelling_model_dropdown]
    )

    text_model_type_dropdown.change(
        fn=chat_bot.update_model_list,
        inputs=[text_model_type_dropdown],
        outputs=[text_model_dropdown]
    )
    
    diffusion_model_type_dropdown.change(
        fn=image_gen.update_diffusion_model_list,
        inputs=[diffusion_model_type_dropdown],
        outputs=[diffusion_model_dropdown]
    )
    
    # def toggle_refiner_start_step(model):
    #     slider_visible = model != "None"
    #     return gr.update(visible=slider_visible)
    
    # def toggle_denoise_strength_dropdown(mode):
    #     slider_visible = mode != "None"
    #     return gr.update(visible=slider_visible)
    
    # def toggle_blur_radius_slider(mode):
    #     slider_visible = mode == "Inpaint" or mode == "Inpaint Upload"
    #     return gr.update(visible=slider_visible), gr.update(visible=slider_visible)
    
    # def toggle_diffusion_with_refiner_image_to_image_start(model, mode):
    #     slider_visible = model != "None" and mode != "None"
    #     return gr.update(visible=slider_visible)
    
    diffusion_refiner_model_dropdown.change(
        fn=lambda model: (
            image_gen.toggle_refiner_start_step(model)
            ),
        inputs=[diffusion_refiner_model_dropdown],
        outputs=[diffusion_refiner_start]
    ).then(
        fn=image_gen.toggle_diffusion_with_refiner_image_to_image_start,
        inputs=[diffusion_refiner_model_dropdown, image_to_image_mode],
        outputs=[diffusion_with_refiner_image_to_image_start]
    )

    # def process_uploaded_image(image: str | Image.Image | np.ndarray | Callable | Any):
    #     print(image)
    #     image = client.upload_image(image, overwrite=True)
    #     return image
    
    def process_uploaded_image_for_inpaint(image: str | Image.Image | Any):
        print(image)
        im = {
            "background": image,
            "layers": [],
            "composite": None
        }
        image = client.upload_image(image, overwrite=True)
        return image, gr.update(value=im)
    
    @image_inpaint_masking.apply(inputs=[image_inpaint_input, image_inpaint_masking], outputs=app_state.stored_image_inpaint)
    def process_uploaded_image_inpaint(original_image: str | Image.Image | Any, mask_image: list[str | Image.Image | Any]):
        print(original_image)
        print(mask_image)
        mask = client.upload_mask(original_image, mask_image)
        return mask
        
    # def toggle_image_to_image_input(mode):
    #     image_visible = mode == "Image to Image"
    #     return gr.update(visible=image_visible)
    
    # def toggle_image_inpaint_input(mode):
    #     image_visible = mode == "Inpaint"
    #     return gr.update(visible=image_visible)
    
    # def toggle_image_inpaint_mask(mode):
    #     image_visible = mode == "Inpaint"
    #     return gr.update(visible=image_visible)
        
    # def toggle_image_inpaint_mask_interactive(image: str | Image.Image | Any):
    #     image_interactive = image is not None
    #     return gr.update(interactive=image_interactive)

    def copy_image_for_inpaint(image_input, image) -> gr.update:
        import cv2
        print(type(image_input))
        im = cv2.imread(image_input)
        height, width, channels = im.shape[:3]
        image['background'] = image_input
        image['layers'][0] = np.zeros((height, width, 4), dtype=np.uint8)

        return gr.update(value=image)
        
    
    image_to_image_input.change(
        fn=image_gen.process_uploaded_image,
        inputs=image_to_image_input,
        outputs=app_state.stored_image
    )
    
    image_inpaint_input.upload(
        fn=image_gen.process_uploaded_image,
        inputs=[image_inpaint_input],
        outputs=app_state.stored_image
    ).then(
        fn=copy_image_for_inpaint,
        inputs=[image_inpaint_input, image_inpaint_masking],
        outputs=image_inpaint_masking
    ).then(
        fn=image_gen.toggle_image_inpaint_mask_interactive,
        inputs=image_inpaint_input,
        outputs=image_inpaint_masking
    )
    
    image_to_image_mode.change(
        fn=lambda mode: (
            image_gen.toggle_image_to_image_input(mode),
            image_gen.toggle_image_inpaint_input(mode),
            image_gen.toggle_image_inpaint_mask(mode),
            image_gen.toggle_denoise_strength_dropdown(mode)
            ),
        inputs=[image_to_image_mode],
        outputs=[image_to_image_input,
                 image_inpaint_input,
                 image_inpaint_masking, 
                 denoise_strength_slider]
    ).then(
        fn=image_gen.toggle_diffusion_with_refiner_image_to_image_start,
        inputs=[diffusion_refiner_model_dropdown, image_to_image_mode],
        outputs=[diffusion_with_refiner_image_to_image_start]
    ).then(
        fn=image_gen.toggle_blur_radius_slider,
        inputs=[image_to_image_mode],
        outputs=[blur_radius_slider, blur_expansion_radius_slider]
    )
        
    bot_message_inputs = [app_state.session_id_state, app_state.history_state, text_model_dropdown, app_state.custom_model_path_state, text_api_key_text, app_state.selected_device_state, app_state.seed_state]
    
    demo.load(
        fn=lambda selected_model: (
            chat_bot.toggle_api_key_visibility(selected_model),
            chat_bot.toggle_lora_visibility(selected_model),
            chat_bot.toggle_multimodal_msg_input_visibility(selected_model),
            chat_bot.toggle_standard_msg_input_visibility(selected_model)
        ),
        inputs=[text_model_dropdown],
        outputs=[text_api_key_text, text_lora_dropdown, multimodal_msg, msg]
    )

    def update_character_languages(selected_language: str, selected_character: str):
        """
        인터페이스 언어에 따라 선택된 캐릭터의 언어를 업데이트합니다.
        """
        speech_manager = chat_bot.get_speech_manager(app_state.session_id_state)
        if selected_language in characters[selected_character]["languages"]:
            # 인터페이스 언어가 캐릭터의 지원 언어에 포함되면 해당 언어로 설정
            speech_manager.current_language = selected_language
        else:
            # 지원하지 않는 언어일 경우 기본 언어로 설정
            speech_manager.current_language = characters[selected_character]["default_language"]
        return gr.update()
    
    def generate_diffusion_lora_weight_sliders(selected_loras: List[str]):
        updates=[]
        for i in range(app_state.max_diffusion_lora_rows):
            if i < len(selected_loras):
                # 선택된 LoRA가 있으면 해당 행을 보이게 하고 label 업데이트
                lora_name = selected_loras[i]
                text_update = gr.update(visible=True, label=f"{lora_name} - Text Encoder Weight")
                unet_update = gr.update(visible=True, label=f"{lora_name} - U-Net Weight")
            else:
                # 선택된 LoRA가 없는 행은 숨김 처리
                text_update = gr.update(visible=False)
                unet_update = gr.update(visible=False)
            updates.append(text_update)
            updates.append(unet_update)
        return updates

    @random_prompt_btn.click(outputs=[positive_prompt_input])
    def get_random_prompt():
        """랜덤 프롬프트 생성 함수"""
        prompts = [
            "A serene mountain landscape at sunset",
            "A futuristic cityscape with flying cars",
            "A mystical forest with glowing mushrooms"
        ]
        return random.choice(prompts)
    diffusion_lora_slider_outputs = []
    for te_slider, unet_slider in zip(diffusion_lora_text_encoder_sliders, diffusion_lora_unet_sliders):
        diffusion_lora_slider_outputs.extend([te_slider, unet_slider])
    diffusion_lora_multiselect.change(
        fn=generate_diffusion_lora_weight_sliders,
        inputs=[diffusion_lora_multiselect],
        outputs=diffusion_lora_slider_outputs
    )

    # 이벤트 핸들러 연결
    generate_btn.click(
        fn=image_gen.generate_images_wrapper,
        inputs=[
            positive_prompt_input,       # Positive Prompt
            negative_prompt_input,       # Negative Prompt
            style_dropdown,              # Style
            generation_step_slider,
            diffusion_with_refiner_image_to_image_start,
            diffusion_refiner_start,
            width_slider,                # Width
            height_slider,               # Height
            diffusion_model_dropdown,    # 선택한 이미지 생성 모델 (체크포인트 파일명 또는 diffusers model id)
            diffusion_refiner_model_dropdown, 
            diffusion_model_type_dropdown,  # "checkpoint" 또는 "diffusers" 선택 (라디오 버튼 등)
            diffusion_lora_multiselect,  # 선택한 LoRA 모델 리스트
            vae_dropdown,                # 선택한 VAE 모델
            clip_skip_slider,
            enable_clip_skip_checkbox,
            clip_g_checkbox,
            sampler_dropdown,
            scheduler_dropdown,
            batch_size_input,
            batch_count_input,
            cfg_scale_slider,
            diffusion_seed_input,
            random_seed_checkbox,
            image_to_image_mode, 
            app_state.stored_image,
            app_state.stored_image_inpaint,
            denoise_strength_slider,
            blur_radius_slider,
            blur_expansion_radius_slider,
            diffusion_api_key_text,
            *diffusion_lora_text_encoder_sliders,
            *diffusion_lora_unet_sliders
        ],
        outputs=[gallery, image_history]
    ).then(
        fn=None,
        js=show_confetti
    )

    @language_dropdown.change(
        inputs=[language_dropdown, character_dropdown],
        outputs=[title, session_select_info, language_dropdown, system_message_accordion, system_message_box, text_model_type_dropdown, text_model_dropdown, character_dropdown, text_api_key_text, msg, multimodal_msg, text_advanced_settings, text_seed_input, text_temperature_slider, text_top_k_slider, text_top_p_slider, text_repetition_penalty_slider, reset_btn, reset_all_btn, diffusion_model_type_dropdown, diffusion_model_dropdown, diffusion_api_key_text]
    )
    def change_language(selected_lang: str, selected_character: str):
        """언어 변경 처리 함수"""
        lang_map = {
            "한국어": "ko",
            "日本語": "ja",
            "中文(简体)": "zh_CN",
            "中文(繁體)": "zh_TW",
            "English": "en"
        }
        
        lang_code = lang_map.get(selected_lang, "ko")
        
        if translation_manager.set_language(lang_code):
            if selected_lang in characters[selected_character]["languages"]:
                app_state.speech_manager_state.current_language = selected_lang
            else:
                app_state.speech_manager_state.current_language = characters[selected_character]["languages"][0]
                
            
            system_presets = {
                "AI 비서 (AI Assistant)": AI_ASSISTANT_PRESET,
                "Image Generator": SD_IMAGE_GENERATOR_PRESET,
                "미나미 아스카 (南飛鳥, みなみあすか, Minami Asuka)": MINAMI_ASUKA_PRESET,
                "마코토노 아오이 (真琴乃葵, まことのあおい, Makotono Aoi)": MAKOTONO_AOI_PRESET,
                "아이노 코이토 (愛野小糸, あいのこいと, Aino Koito)": AINO_KOITO_PRESET,
                "아리아 프린세스 페이트 (アリア·プリンセス·フェイト, Aria Princess Fate)": ARIA_PRINCESS_FATE_PRESET,
                "아리아 프린스 페이트 (アリア·プリンス·フェイト, Aria Prince Fate)": ARIA_PRINCE_FATE_PRESET,
                "왕 메이린 (王美玲, ワン·メイリン, Wang Mei-Ling)": WANG_MEI_LING_PRESET,
                "미스티 레인 (ミスティ·レーン, Misty Lane)": MISTY_LANE_PRESET,
                '릴리 엠프레스 (リリー·エンプレス, Lily Empress)': LILY_EMPRESS_PRESET,
                "최유나 (崔有娜, チェ·ユナ, Choi Yuna)": CHOI_YUNA_PRESET,
                "최유리 (崔有莉, チェ·ユリ, Choi Yuri)": CHOI_YURI_PRESET
            }
                
            preset_name = system_presets.get(selected_character, AI_ASSISTANT_PRESET)
            system_content = preset_name.get(lang_code, "당신은 유용한 AI 비서입니다.")
            
            return [
                gr.update(value=f"## {_('main_title')}"),
                gr.update(value=_('select_session_info')),
                gr.update(label=_('language_select'),
                info=_('language_info')),
                gr.update(label=_("system_message")),
                gr.update(
                    label=_("system_message"),
                    value=system_content,
                    placeholder=_("system_message_placeholder")
                ),
                gr.update(label=_("model_type_label")),
                gr.update(label=_("model_select_label")),
                gr.update(label=_('character_select_label'), info=_('character_select_info')),
                gr.update(label=_("api_key_label")),
                gr.update(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder")
                ),
                gr.update(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder")
                ),
                gr.update(label=_("advanced_setting")),
                gr.update(label=_("seed_label"), info=_("seed_info")),
                gr.update(label=_("temperature_label")),
                gr.update(label=_("top_k_label")),
                gr.update(label=_("top_p_label")),
                gr.update(label=_("repetition_penalty_label")),
                gr.update(value=_("reset_session_button")),
                gr.update(value=_("reset_all_sessions_button")),
                gr.update(label=_("model_type_label")),
                gr.update(label=_("model_select_label")),
                gr.update(label=_("api_key_label"))
            ]
        else:
            # 언어 변경 실패 시 아무 것도 하지 않음
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        # 메시지 전송 시 함수 연결
    msg.submit(
        fn=chat_bot.process_message_user,
        inputs=[
            msg,  # 사용자 입력
            app_state.session_id_state,
            app_state.history_state,
            system_message_box,
            character_dropdown,
            app_state.selected_language_state
        ],
        outputs=[
            msg,            # 사용자 입력 필드 초기화
            app_state.history_state,  # 히스토리 업데이트
            chatbot,        # Chatbot UI 업데이트
        ],
        queue=False
    ).then(
        fn=chat_bot.process_message_bot,
        inputs=[
            app_state.session_id_state,
            app_state.history_state,
            text_model_dropdown,
            text_lora_dropdown,
            app_state.custom_model_path_state,
            msg,
            text_api_key_text,
            app_state.selected_device_state,
            app_state.seed_state,
            app_state.temperature_state,
            app_state.top_k_state,
            app_state.top_p_state,
            app_state.repetition_penalty_state,
            app_state.selected_language_state,
        ],
        outputs=[
            app_state.history_state,
            chatbot,
            status_text,  # 상태 메시지
            chat_title_box
        ],
        queue=True  # 모델 추론이 들어가므로 True
    )

    multimodal_msg.submit(
        fn=chat_bot.process_message_user,
        inputs=[
            multimodal_msg,  # 사용자 입력
            app_state.session_id_state,
            app_state.history_state,
            system_message_box,
            character_dropdown,
            app_state.selected_language_state
        ],
        outputs=[
            multimodal_msg, # 사용자 입력 필드 초기화
            app_state.history_state,  # 히스토리 업데이트
            chatbot,        # Chatbot UI 업데이트
        ],
        queue=False
    ).then(
        fn=chat_bot.process_message_bot,
        inputs=[
            app_state.session_id_state,
            app_state.history_state,
            text_model_dropdown,
            text_lora_dropdown,
            app_state.custom_model_path_state,
            multimodal_msg,
            text_api_key_text,
            app_state.selected_device_state,
            app_state.seed_state,
            app_state.temperature_state,
            app_state.top_k_state,
            app_state.top_p_state,
            app_state.repetition_penalty_state,
            app_state.selected_language_state,
        ],
        outputs=[
            app_state.history_state,
            chatbot,
            status_text,  # 상태 메시지
            chat_title_box
        ],
        queue=True  # 모델 추론이 들어가므로 True
    )
        
    demo.load(
        fn=chat_bot.refresh_sessions,
        inputs=[],
        outputs=[session_select_dropdown],
        queue=False
    )

    reset_btn.click(
        fn=lambda: chat_bot.show_reset_modal("single"),
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    reset_all_btn.click(
        fn=lambda: chat_bot.show_reset_modal("all"),
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    
    cancel_btn.click(
        fn=chat_bot.hide_reset_modal,
        outputs=[reset_modal, single_reset_content, all_reset_content]
    )
    
    confirm_btn.click(
        fn=chat_bot.handle_reset_confirm,
        inputs=[app_state.history_state, chatbot, system_message_box, app_state.selected_language_state, app_state.session_id_state],
        outputs=[reset_modal, single_reset_content, all_reset_content, 
                msg, app_state.history_state, chatbot, status_text]
    ).then(
        fn=chat_bot.refresh_sessions,  # 세션 목록 갱신 (전체 초기화의 경우)
        outputs=[session_select_dropdown]
    )
    
    @gr.on(triggers=[chatbot_sidetab.click, demo.load], inputs=[], outputs=[chat_side.sidebar, diff_side.sidebar, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container.container, diff_container.container, story_container, tts_container, translate_container, download_container])
    def select_chat_tab():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    @diffusion_sidetab.click(inputs=[], outputs=[chat_side.sidebar, diff_side.sidebar, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container.container, diff_container.container, story_container, tts_container, translate_container, download_container])
    def select_image_generation_tab():
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    @storyteller_sidetab.click(inputs=[], outputs=[chat_side.sidebar, diff_side.sidebar, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container.container, diff_container.container, story_container, tts_container, translate_container, download_container])
    def select_storyteller_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    @tts_sidetab.click(inputs=[], outputs=[chat_side.sidebar, diff_side.sidebar, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container.container, diff_container.container, story_container, tts_container, translate_container, download_container])
    def select_tts_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    @translate_sidetab.click(inputs=[], outputs=[chat_side.sidebar, diff_side.sidebar, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container.container, diff_container.container, story_container, tts_container, translate_container, download_container])
    def select_translate_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(elem_classes="tab"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

    @download_sidetab.click(inputs=[], outputs=[chat_side.sidebar, diff_side.sidebar, storyteller_side, tts_side, translate_side, chatbot_sidetab, diffusion_sidetab, storyteller_sidetab, tts_sidetab, translate_sidetab, download_sidetab, chat_container.container, diff_container.container, story_container, tts_container, translate_container, download_container])
    def select_download_tab():
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab"), gr.update(elem_classes="tab-active"), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
    demo.load(None, None, None).then(
        fn=lambda evt: (
            gr.update(visible=False),  # reset_modal
            gr.update(visible=False),  # single_content
            gr.update(visible=False),  # all_content
            None,  # msg (변경 없음)
            None,  # history (변경 없음)
            None,  # chatbot (변경 없음)
            None   # status (변경 없음)
        ) if evt.key == "Escape" else (
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            None,
            None,
            None
        ),
        inputs=[],
        outputs=[
            reset_modal,
            single_reset_content,
            all_reset_content,
            msg,
            app_state.history_state,
            chatbot,
            status_text
        ]
    )

    return settings_button, text_model_type_dropdown, text_model_dropdown, system_message_box, preset_dropdown
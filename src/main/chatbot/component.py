from dataclasses import dataclass

import gradio as gr
# from gradio_i18n import gettext as _, translate_blocks

# from translations import i18n as _

from ...start_app import app_state, ui_component
from ...models import PROVIDER_LIST
from ...common.translations import _
from ...common.utils import get_all_loras
from ...common.default_language import default_language
from ...common.character_info import characters
from ...common.database import get_preset_choices

@dataclass
class ChatbotComponent:
    session_select_dropdown: gr.Dropdown = None
    chat_title_box: gr.Textbox = None
    add_session_icon_btn: gr.Button = None
    delete_session_icon_btn: gr.Button = None

    model_provider_dropdown: gr.Dropdown = None
    model_type_dropdown: gr.Radio = None
    model_dropdown: gr.Dropdown = None
    api_key_text: gr.Textbox = None
    lora_dropdown: gr.Dropdown = None
    refresh_button: gr.Button = None
    clear_all_btn: gr.Button = None

    system_message_accordion: gr.Accordion = None
    system_message_box: gr.Textbox = None
    chatbot: gr.Chatbot = None
    msg: gr.Textbox = None
    multimodal_msg: gr.MultimodalTextbox = None

    profile_image: gr.Image = None
    character_dropdown: gr.Dropdown = None

    advanced_setting: gr.Accordion = None
    seed_input: gr.Number = None
    max_length_input: gr.Slider = None
    temperature_slider: gr.Slider = None
    top_k_slider: gr.Slider = None
    top_p_slider: gr.Slider = None
    repetition_penalty_slider: gr.Slider = None
    enable_thinking_checkbox: gr.Checkbox = None
    preset_dropdown: gr.Dropdown = None
    change_preset_button: gr.Button = None
    reset_btn: gr.Button = None
    reset_all_btn: gr.Button = None

    status_text: gr.Markdown = None
    image_info: gr.Markdown = None
    session_select_info: gr.Markdown = None

    @classmethod
    def create_chatbot_side_session_container(cls):
        with gr.Row(elem_classes="session-container"):
            with gr.Column():
                gr.Markdown("### Chat Session")
                session_select_dropdown = gr.Dropdown(
                    label="세션 선택",
                    choices=[],  # 앱 시작 시 혹은 별도의 로직으로 세션 목록을 채움
                    value=None,
                    interactive=True,
                    container=False,
                    scale=8,
                    elem_classes="session-dropdown"
                )
                chat_title_box=gr.Textbox(
                    value="",
                    interactive=False
                )
                add_session_icon_btn = gr.Button("📝", elem_classes="icon-button", scale=1, variant="secondary")
                delete_session_icon_btn = gr.Button("🗑️", elem_classes="icon-button-delete", scale=1, variant="stop")
                
        
        ui_component.session_select_dropdown = session_select_dropdown
        ui_component.chat_title_box = chat_title_box
        ui_component.add_session_icon_btn = add_session_icon_btn
        ui_component.delete_session_icon_btn = delete_session_icon_btn

        return cls(session_select_dropdown=session_select_dropdown, chat_title_box=chat_title_box, add_session_icon_btn=add_session_icon_btn, delete_session_icon_btn=delete_session_icon_btn)

    @classmethod
    def create_chatbot_side_model_container(cls):
        with gr.Row(elem_classes="model-container"):
            with gr.Column():
                gr.Markdown("### Model Selection")
                model_provider_dropdown = gr.Dropdown(
                    label=_("model_provider_label"),
                    choices=PROVIDER_LIST,
                    value=PROVIDER_LIST[0],
                    interactive=True,
                    elem_classes="model-dropdown"
                )
                model_type_dropdown = gr.Radio(
                    label=_("model_type_label"),
                    choices=app_state.llm_type_choices,
                    value=app_state.llm_type_choices[0],
                    elem_classes="model-dropdown"
                )
                model_dropdown = gr.Dropdown(
                    label=_("model_select_label"),
                    choices=app_state.initial_choices,
                    value=app_state.initial_choices[0] if len(app_state.initial_choices) > 0 else None,
                    elem_classes="model-dropdown"
                )
                api_key_text = gr.Textbox(
                    label=_("api_key_label"),
                    placeholder="sk-...",
                    visible=False,
                    elem_classes="api-key-input"
                )
                lora_dropdown = gr.Dropdown(
                    label=_("lora_select_label"),
                    choices=get_all_loras(),
                    value="None",
                    interactive=True,
                    visible=False,
                    elem_classes="model-dropdown"
                )
                refresh_button = gr.Button(_("refresh_model_list_button"))
                clear_all_btn = gr.Button(_("cache_clear_all_button"))
                
        ui_component.model_provider_dropdown = model_provider_dropdown
        ui_component.model_type_dropdown = model_type_dropdown
        ui_component.model_dropdown = model_dropdown
        ui_component.api_key_text = api_key_text
        ui_component.lora_dropdown = lora_dropdown
        ui_component.refresh_button = refresh_button
        ui_component.clear_all_btn = clear_all_btn

        return cls(model_provider_dropdown = model_provider_dropdown, model_type_dropdown=model_type_dropdown, model_dropdown=model_dropdown, api_key_text=api_key_text, lora_dropdown=lora_dropdown, refresh_button = refresh_button, clear_all_btn=clear_all_btn)

    @classmethod
    def create_chat_container_main_panel(cls, chat_wrapper_fn, additional_inputs):
        with gr.Column(scale=7):
            with gr.Accordion(_("system_message"), elem_classes="accordion-container") as system_message_accordion:
                system_message_box = gr.Textbox(
                    label=_("system_message"),
                    value=app_state.initial_system_message,
                    # placeholder=_("system_message_placeholder"),
                    elem_classes="system-message",
                    autofocus=True
                )
            
            # system_message_box is the 4th argument in chat_wrapper (after message, history, session_id)
            # We need to prepend session_id_state and append system_message_box to the inputs list?
            # chat_wrapper args: message, history, session_id, system_msg, ...
            # ChatInterface passes message, history automatically.
            # additional_inputs should start with session_id.
            
            # additional_inputs passed from __init__.py will contain:
            # [session_id_state, (system_msg will be inserted here), selected_character, ...]
            
            # Actually, let's construct the full list here.
            # The caller will pass [session_id_state, selected_character, ...]
            # We insert system_message_box at index 1.
            
            real_additional_inputs = list(additional_inputs)
            real_additional_inputs.insert(1, system_message_box)
            
            chatbot = gr.Chatbot(
                elem_classes=["chat-window"],
                avatar_images = [None, characters[app_state.initial_last_character]["profile_image"]],
                height = 600,
                label = "Chatbot",
                show_label = True,
                # type = "messages"
                reasoning_tags = [("<think>","</think>"), ("<thinking>","</thinking>")]
            )

            chat_interface = gr.ChatInterface(
                fn=chat_wrapper_fn,
                # type='messages',
                chatbot=chatbot,
                multimodal=True,
                additional_inputs=real_additional_inputs,
                additional_outputs=[app_state.history_state, ui_component.status_text, ui_component.chat_title_box],
                autofocus=True,
                fill_height=True,
                save_history=False # We handle history manually
            )


            # Access the textbox. In multimodal mode, it's a MultimodalTextbox.
            msg = chat_interface.textbox
            msg.elem_classes = ["message-input"]
            msg.show_label = False
            msg.container = False
            msg.scale = 9
            
            # For compatibility with existing code that expects multimodal_msg
            multimodal_msg = msg 
        
        ui_component.system_message_accordion = system_message_accordion
        ui_component.system_message_box = system_message_box
        ui_component.chatbot = chatbot
        ui_component.msg = msg
        ui_component.multimodal_msg = multimodal_msg

        return cls(system_message_accordion=system_message_accordion, system_message_box=system_message_box, chatbot=chatbot, msg=msg, multimodal_msg=multimodal_msg)

    @classmethod
    def create_chat_container_side_panel(cls):
        with gr.Column(scale=3, elem_classes="side-panel"):
            profile_image = gr.Image(
                label=_('profile_image_label'),
                visible=True,
                interactive=False,
                show_label=True,
                width="auto",
                height="auto",
                value=characters[app_state.initial_last_character]["profile_image"],
                elem_classes="profile-image"
            )
            character_dropdown = gr.Dropdown(
                label=_('character_select_label'),
                choices=list(characters.keys()),
                value=app_state.initial_last_character,
                interactive=True,
                info=_('character_select_info'),
                elem_classes='character-dropdown'
            )
            with gr.Accordion(_("advanced_setting"), open=False, elem_classes="accordion-container") as advanced_setting:                        
                seed_input = gr.Number(
                    label=_("seed_label"),
                    value=42,
                    precision=0,
                    step=1,
                    interactive=True,
                    info=_("seed_info"),
                    elem_classes="seed-input"
                )
                max_length_input = gr.Slider(
                    label="Max Length",
                    minimum=-1,
                    maximum=4096,
                    value=-1,
                    step=1,
                    interactive=True,
                    info="Set the maximum length of the generated response.",
                    elem_classes="max-length-input"
                )
                temperature_slider=gr.Slider(
                    label=_("temperature_label"),
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.1,
                    interactive=True
                )
                top_k_slider=gr.Slider(
                    label=_("top_k_label"),
                    minimum=0,
                    maximum=100,
                    value=20,
                    step=1,
                    interactive=True
                )
                top_p_slider=gr.Slider(
                    label=_("top_p_label"),
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    interactive=True
                )
                repetition_penalty_slider=gr.Slider(
                    label=_("repetition_penalty_label"),
                    minimum=0.0,
                    maximum=2.0,
                    value=1.1,
                    step=0.1,
                    interactive=True
                )
                enable_thinking_checkbox = gr.Checkbox(
                    label="Enable Thinking",
                    value=False,
                    info="Enable thinking for Qwen models.",
                    elem_classes="enable-thinking-checkbox"
                )
                preset_dropdown = gr.Dropdown(
                    label="프리셋 선택",
                    choices=get_preset_choices(default_language),
                    value=app_state.last_preset,
                    interactive=True,
                    elem_classes="preset-dropdown"
                )
                change_preset_button = gr.Button("프리셋 변경")
                reset_btn = gr.Button(
                    value=_("reset_session_button"),  # "세션 초기화"에 해당하는 번역 키
                    variant="secondary",
                    scale=1
                )
                reset_all_btn = gr.Button(
                    value=_("reset_all_sessions_button"),  # "모든 세션 초기화"에 해당하는 번역 키
                    variant="secondary",
                    scale=1
                )
        
        ui_component.profile_image = profile_image
        ui_component.character_dropdown = character_dropdown
        ui_component.text_advanced_settings = advanced_setting
        ui_component.text_seed_input = seed_input
        ui_component.text_max_length_input = max_length_input
        ui_component.text_temperature_slider = temperature_slider
        ui_component.text_top_k_slider = top_k_slider
        ui_component.text_top_p_slider = top_p_slider
        ui_component.text_repetition_penalty_slider = repetition_penalty_slider
        ui_component.text_enable_thinking_checkbox = enable_thinking_checkbox
        ui_component.text_preset_dropdown = preset_dropdown
        ui_component.text_change_preset_button = change_preset_button
        ui_component.text_reset_btn = reset_btn
        ui_component.text_reset_all_btn = reset_all_btn

        return cls(profile_image=profile_image, character_dropdown=character_dropdown, advanced_setting=advanced_setting, seed_input=seed_input, max_length_input=max_length_input, temperature_slider=temperature_slider, top_k_slider=top_k_slider, top_p_slider=top_p_slider, repetition_penalty_slider=repetition_penalty_slider, enable_thinking_checkbox=enable_thinking_checkbox, preset_dropdown=preset_dropdown, change_preset_button=change_preset_button, reset_btn=reset_btn, reset_all_btn=reset_all_btn)
    
    @classmethod
    def create_chat_container_status_bar(cls, render=True):
        status_text = gr.Markdown("Ready", elem_id="status_text", render=render)
        image_info = gr.Markdown("", visible=False, render=render)
        session_select_info = gr.Markdown(_('select_session_info'), render=render)

        ui_component.status_text = status_text
        ui_component.image_info = image_info
        ui_component.session_select_info = session_select_info

        return cls(status_text=status_text, image_info=image_info, session_select_info=session_select_info)
import gradio as gr
from dataclasses import dataclass

from ...start_app import app_state, ui_component
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

    model_type_dropdown: gr.Radio = None
    model_dropdown: gr.Dropdown = None
    api_key_text: gr.Textbox = None
    lora_dropdown: gr.Dropdown = None

    system_message_accordion: gr.Accordion = None
    system_message_box: gr.Textbox = None
    chatbot: gr.Chatbot = None
    msg: gr.Textbox = None
    multimodal_msg: gr.MultimodalTextbox = None

    profile_image: gr.Image = None
    character_dropdown: gr.Dropdown = None

    advanced_setting: gr.Accordion = None
    seed_input: gr.Number = None
    temperature_slider: gr.Slider = None
    top_k_slider: gr.Slider = None
    top_p_slider: gr.Slider = None
    repetition_penalty_slider: gr.Slider = None
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
                    label="LoRA 모델 선택",
                    choices=get_all_loras(),
                    value="None",
                    interactive=True,
                    visible=False,
                    elem_classes="model-dropdown"
                )
                
        ui_component.model_type_dropdown = model_type_dropdown
        ui_component.model_dropdown = model_dropdown
        ui_component.api_key_text = api_key_text
        ui_component.lora_dropdown = lora_dropdown

        return cls(model_type_dropdown=model_type_dropdown, model_dropdown=model_dropdown, api_key_text=api_key_text, lora_dropdown=lora_dropdown)

    @classmethod
    def create_chat_container_main_panel(cls):
        with gr.Column(scale=7):
            with gr.Accordion(_("system_message"), elem_classes="accordion-container") as system_message_accordion:
                system_message_box = gr.Textbox(
                    label=_("system_message"),
                    value=app_state.system_message,
                    placeholder=_("system_message_placeholder"),
                    elem_classes="system-message"
                )
                            
            chatbot = gr.Chatbot(
                height=400, 
                label="Chatbot", 
                type="messages", 
                elem_classes=["chat-messages"]
            )
                            
            with gr.Row(elem_classes="input-area"):
                msg = gr.Textbox(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder"),
                    scale=9,
                    show_label=False,
                    elem_classes="message-input",
                    submit_btn=True,
                    # stop_btn=True
                )
                multimodal_msg = gr.MultimodalTextbox(
                    label=_("message_input_label"),
                    placeholder=_("message_placeholder"),
                    file_types=["image"],
                    scale=9,
                    show_label=False,
                    elem_classes="message-input",
                    submit_btn=True,
                    # stop_btn=True
                )
        #         image_input = gr.Image(label=_("image_upload_label"), type="pil", visible=False)
        
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
                value=characters[app_state.last_character]["profile_image"],
                elem_classes="profile-image"
            )
            character_dropdown = gr.Dropdown(
                label=_('character_select_label'),
                choices=list(characters.keys()),
                value=app_state.last_character,
                interactive=True,
                info=_('character_select_info'),
                elem_classes='profile-image'
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
        ui_component.seed_input = seed_input
        ui_component.temperature_slider = temperature_slider
        ui_component.top_k_slider = top_k_slider
        ui_component.top_p_slider = top_p_slider
        ui_component.repetition_penalty_slider = repetition_penalty_slider
        ui_component.preset_dropdown = preset_dropdown
        ui_component.change_preset_button = change_preset_button
        ui_component.reset_btn = reset_btn
        ui_component.reset_all_btn = reset_all_btn

        return cls(profile_image=profile_image, character_dropdown=character_dropdown, advanced_setting=advanced_setting, seed_input=seed_input, temperature_slider=temperature_slider, top_k_slider=top_k_slider, top_p_slider=top_p_slider, repetition_penalty_slider=repetition_penalty_slider, preset_dropdown=preset_dropdown, change_preset_button=change_preset_button, reset_btn=reset_btn, reset_all_btn=reset_all_btn)
    
    @classmethod
    def create_chat_container_status_bar(cls):
        status_text = gr.Markdown("Ready", elem_id="status_text")
        image_info = gr.Markdown("", visible=False)
        session_select_info = gr.Markdown(_('select_session_info'))

        ui_component.status_text = status_text
        ui_component.image_info = image_info
        ui_component.session_select_info = session_select_info

        return cls(status_text=status_text, image_info=image_info, session_select_info=session_select_info)
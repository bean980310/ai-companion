import gradio as gr
from src.main.storyteller import create_story_side, create_story_container
from src.main.chatbot import chat_bot
from src.common.translations import translation_manager, _
from src.common_blocks import create_page_header, get_language_code

with gr.Blocks() as demo:
    # Page Header with Language Selector
    # page_header = create_page_header(page_title_key="storyteller_title")
    # language_dropdown = page_header.language_dropdown

    with gr.Sidebar():
        storyteller_side_panel, storytelling_model_type_dropdown, storytelling_model_dropdown, storytelling_api_key_text, storytelling_lora_dropdown = create_story_side()
        # Note: create_story_side returns the Column (storyteller_side_panel) as first arg, and components.
        # However, inside gr.Sidebar(), we might have nested Columns. That's fine.
        
    story_container, storytelling_input, storytelling_btn, storytelling_output, story_adv_setting, storyteller_seed_input, storyteller_temperature_slider, storyteller_top_k_slider, storyteller_top_p_slider, storyteller_repetition_penalty_slider = create_story_container()
    
    # Layout adjustment: create_story_side creates a Column that is initially hidden (visible=False) probably?
    # Actually checking src/main/sidebar.py:
    # storyteller_side, ... = create_story_side()
    # It seems to just create components.
    
    # Event wiring
    gr.on(
        triggers=[storytelling_model_dropdown.change, demo.load],
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

    # Language Change Event
    def on_storyteller_language_change(selected_lang: str):
        lang_code = get_language_code(selected_lang)
        translation_manager.set_language(lang_code)
        return [
            # gr.update(value=f"## {_('storyteller_title')}"),
            # gr.update(label=_('language_select'), info=_('language_info'))
        ]

    # language_dropdown.change(
    #     fn=on_storyteller_language_change,
    #     inputs=[language_dropdown],
    #     outputs=[page_header.title, language_dropdown]
    # )

if __name__ == "__main__":
    demo.launch()

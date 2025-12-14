import gradio as gr

from src.start_app import ui_component
from translations import i18n
from src.main.storyteller import create_story_side, create_story_container
from src.main.chatbot import chat_bot
from src.utils.translate import translate_page


with gr.Blocks() as demo:
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

    # translate_page(blocks=demo, translation=i18n, lang=ui_component.language_dropdown, persistant=True)

if __name__ == "__main__":
    demo.launch()

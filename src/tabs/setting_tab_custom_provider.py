import gradio as gr

from ..common.custom_providers import (
    add_custom_provider,
    delete_custom_provider,
    get_custom_provider_names,
    load_custom_providers,
    update_custom_provider,
)


def _refresh_choices():
    return gr.update(choices=get_custom_provider_names())


def create_custom_provider_tab():
    with gr.Tab("사용자 지정 Provider"):
        gr.Markdown("OpenAI 호환 API 엔드포인트 프로필을 추가하면 채팅의 Provider 드롭다운에서 `custom:<이름>`으로 선택할 수 있습니다.")

        with gr.Row():
            provider_name_input = gr.Textbox(label="프로필 이름", placeholder="예: my-llm")
            provider_base_url_input = gr.Textbox(label="Base URL", placeholder="예: https://api.example.com")
            provider_api_key_input = gr.Textbox(label="API Key", placeholder="sk-...", type="password")

        with gr.Row():
            add_provider_btn = gr.Button("추가", variant="primary")
            update_provider_btn = gr.Button("선택한 프로필 수정")
            delete_provider_btn = gr.Button("선택한 프로필 삭제", variant="stop")

        provider_list_dropdown = gr.Dropdown(label="등록된 프로필", choices=get_custom_provider_names(), interactive=True)
        provider_status = gr.Markdown("")

        # 등록된 프로필 선택 시 필드에 로드
        def load_selected(name):
            for provider in load_custom_providers():
                if provider["name"] == name:
                    return provider["name"], provider["base_url"], provider["api_key"]
            return "", "", ""

        provider_list_dropdown.change(
            fn=load_selected,
            inputs=[provider_list_dropdown],
            outputs=[provider_name_input, provider_base_url_input, provider_api_key_input],
        )

        def add_provider(name, base_url, api_key):
            ok, msg = add_custom_provider(name, base_url, api_key)
            if ok:
                return gr.update(), "", "", "", f"✅ {msg}", _refresh_choices()
            return gr.update(), name, base_url, api_key, f"❌ {msg}", gr.update()

        add_provider_btn.click(
            fn=add_provider,
            inputs=[provider_name_input, provider_base_url_input, provider_api_key_input],
            outputs=[provider_list_dropdown, provider_name_input, provider_base_url_input, provider_api_key_input, provider_status, provider_list_dropdown],
        )

        def update_provider(name, base_url, api_key):
            ok, msg = update_custom_provider(name, base_url, api_key)
            if ok:
                return f"✅ {msg}", _refresh_choices()
            return f"❌ {msg}", gr.update()

        update_provider_btn.click(
            fn=update_provider,
            inputs=[provider_name_input, provider_base_url_input, provider_api_key_input],
            outputs=[provider_status, provider_list_dropdown],
        )

        def delete_provider(name):
            ok, msg = delete_custom_provider(name)
            if ok:
                return f"✅ {msg}", gr.update(), "", "", "", _refresh_choices()
            return f"❌ {msg}", gr.update(), name, "", "", gr.update()

        delete_provider_btn.click(
            fn=delete_provider,
            inputs=[provider_name_input],
            outputs=[provider_status, provider_list_dropdown, provider_name_input, provider_base_url_input, provider_api_key_input, provider_list_dropdown],
        )
import gradio as gr

def create_text_to_image_layout(vae_choices):
    with gr.Row(elem_classes="chat-interface"):
        with gr.Column(scale=7):
            positive_prompt_input = gr.Textbox(
                label="Positive Prompt",
                placeholder="Enter positive prompt...",
                lines=3,
                elem_classes="message-input"
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                placeholder="Enter negative prompt...",
                lines=3,
                elem_classes="message-input"
            )
                        
            with gr.Row():
                style_dropdown = gr.Dropdown(
                    label="Style",
                    choices=["Photographic", "Digital Art", "Oil Painting", "Watercolor"],
                    value="Photographic"
                )
                            
            with gr.Row():
                width_slider = gr.Slider(
                    label="Width",
                    minimum=128,
                    maximum=2048,
                    step=64,
                    value=512
                )
                height_slider = gr.Slider(
                    label="Height",
                    minimum=128,
                    maximum=2048,
                    step=64,
                    value=512
                )
                        
            with gr.Row():
                generation_step_slider=gr.Slider(
                    label="Generation Steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=20
                )
                        
            with gr.Row():
                random_prompt_btn = gr.Button("🎲 Random Prompt", variant="secondary")
                generate_btn = gr.Button("🎨 Generate", variant="primary")
                        
            gallery = gr.Gallery(
                label="Generated Images",
                columns=2,
                rows=2
            )

        with gr.Column(scale=3, elem_classes="side-panel"):
            image_history = gr.Dataframe(
                headers=["Prompt", "Negative Prompt", "Steps", "Model", "Sampler", "Scheduler", "CFG Scale", "Seed", "Width", "Height"],
                label="Generation History"
            )
                        
            with gr.Accordion("Advanced Settings", open=False):
                sampler_dropdown = gr.Dropdown(
                    label="Sampler",
                    choices=["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim"],
                    value="euler"
                )
                scheduler_dropdown = gr.Dropdown(
                    label="Scheduler",
                    choices=["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic", "lm_optimal"],  # 실제 옵션에 맞게 변경
                    value="normal"
                )
                cfg_scale_slider = gr.Slider(
                    label="CFG Scale",
                    minimum=1,
                    maximum=20,
                    step=0.5,
                    value=7.5
                )
                with gr.Row():
                    diffusion_seed_input = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0
                    )
                    random_seed_checkbox = gr.Checkbox(
                        label="Random Seed",
                        value=True
                    )
                with gr.Row():
                    vae_dropdown=gr.Dropdown(
                        label="Select VAE Model",
                        choices=vae_choices,
                        value="Default",
                        interactive=True,
                        info="Select VAE model to apply to the diffusion model.",
                        elem_classes="model-dropdown"
                    )
                with gr.Row():
                    clip_skip_slider = gr.Slider(
                        label="Clip Skip",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1
                    )
                    clip_g_checkbox = gr.Checkbox(
                        label="Enable Clip G",
                        value=False
                    )
                with gr.Row():
                    batch_size_input = gr.Number(
                        label="Batch Size",
                        value=1,
                        precision=0
                    )
                    batch_count_input = gr.Number(
                        label="Batch Count",
                        value=1,
                        precision=0
                    )
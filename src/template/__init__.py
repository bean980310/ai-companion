
def llama3_template():
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "{system_input}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        "{user_input}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    
def qwen2_template():
    return (
        "<|im_start|>system\n"
        "{system_input}<|im_end|>\n"
        "<|im_start|>user\n"
        "{user_input}<|im_end|>\n"
        "<|im_start|>assistant"
    )
import os
import torch
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.common.utils import make_local_dir_name

from peft import PeftModel

logger = logging.getLogger(__name__)

class Aya23Handler:
    def __init__(self, model_id, lora_model_id=None, local_model_path=None, lora_path=None, model_type="transformers", device='cpu'):
        self.model_dir = local_model_path or os.path.join("./models/llm", model_id)
        self.lora_model_dir = lora_path or (os.path.join("./models/llm/loras", lora_model_id) if lora_model_id else None)
        self.tokenizer = None
        self.model = None
        self.device = device
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[*] Loading tokenizer from {self.model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, 
                trust_remote_code=True, 
                encode_special_tokens=True
            )
            
            logger.info(f"[*] Loading model from {self.model_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            if self.lora_model_dir and os.path.exists(self.lora_model_dir):
                logger.info(f"[*] Loading LoRA from {self.lora_model_dir}")
                self.model=PeftModel.from_pretrained(
                    self.model_dir,
                    self.lora_model_dir
                ).to(self.device)
            logger.info(f"[*] Model loaded successfully: {self.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load GLM4 Model: {str(e)}\n\n{traceback.format_exc()}")
            raise

    def _get_message_format(self, prompts):
        formatted_messages = []
        for p in prompts:
            if not isinstance(p, str):
                logger.warning(f"Unexpected prompt type: {type(p)}. Converting to string.")
                p = str(p)
            formatted_messages.append({"role": "user", "content": p})
        return formatted_messages

    def generate_answer(
            self,
            prompts,
            temperature=0.3,
            top_p=0.75,
            top_k=0,
            max_new_tokens=1024,
            verbose=False
        ):
        """
        Generate answers for the given prompts using the loaded model.
        
        Args:
            prompts (list): List of input prompts
            temperature (float): Sampling temperature (default: 0.3)
            top_p (float): Nucleus sampling parameter (default: 0.75)
            top_k (int): Top-k sampling parameter (default: 0)
            max_new_tokens (int): Maximum number of tokens to generate (default: 1024)
            verbose (bool): Whether to print prompt-response pairs (default: False)
            
        Returns:
            list: Generated responses for each prompt
        """
        try:
            messages = self._get_message_format(prompts)
            
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt"
            )
            input_ids = input_ids.to(self.model.device)
            prompt_padded_len = len(input_ids[0])

            gen_tokens = self.model.generate(
                input_ids,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                do_sample=True
            )

            # Get only generated tokens
            gen_tokens = [gt[prompt_padded_len:] for gt in gen_tokens]
            generations = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

            if verbose:
                for prompt, response in zip(prompts, generations):
                    print("PROMPT", prompt, "RESPONSE", response, "\n", sep="\n")

            return generations

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}\n\n{traceback.format_exc()}")
            raise
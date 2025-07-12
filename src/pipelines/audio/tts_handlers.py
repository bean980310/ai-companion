import outetts
import os
import torch

class OuteTTSTransformersHandler:
    def __init__(self, model_id, lora_model_id=None, model_type="transformers", device='cpu', speaker='speaker.json', **kwargs):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = device

        self.interface = None
        self.speaker = speaker

        self.temperature = kwargs.get("temperature", 1.0)
        self.repetition_penalty=kwargs.get("repetition_penalty", 1.0)
        self.repetition_range=kwargs.get("repetition_range", 64)
        self.top_k=kwargs.get("top_k", 50)
        self.top_p=kwargs.get("top_p", 1.0)
        self.min_p=kwargs.get("min_p", 0.0)
        self.mirostat=kwargs.get("mirostat", False)
        self.mirostat_tau=kwargs.get("mirostat_tau", 5)
        self.mirostat_eta=kwargs.get("mirostat_eta", 0.1)
        self.max_length=kwargs.get("max_length", 8192)

        self.model_id = model_id
        self.local_model_path = os.path.join("./models/tts/outetts", model_id)

    def load_model(self):
        self.config = outetts.ModelConfig(
            model_path=self.local_model_path,
            tokenizer_path=self.local_model_path,
            interface_version=outetts.InterfaceVersion.V3,
            backend=outetts.Backend.HF,
            device=self.device,
            dtype=torch.bfloat16
        )

        self.interface = outetts.Interface(config=self.config)
        self.speaker = self.interface.load_speaker(self.speaker)

    def generate_tts(self, prompt):
        output = self.interface.generate(
            config=outetts.GenerationConfig(
                text=prompt,
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=self.speaker,
                sampler_config=outetts.SamplerConfig(
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    repetition_range=self.repetition_range,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    min_p=self.min_p,
                    mirostat_tau=self.mirostat_tau,
                    mirostat_eta=self.mirostat_eta,
                    mirostat=self.mirostat,
                ),
                max_length=self.max_length
            )
        )

        return output
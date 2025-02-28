import torch
from transformers import VitsTokenizer, VitsModel, set_seed
import os
import scipy

def text_to_speech(model_id, inputs):
    model_path=os.path.join("./models/tts", model_id)
    tokenizer = VitsTokenizer.from_pretrained(model_path)
    model = VitsModel.from_pretrained(model_path)
    
    inputs=tokenizer(text=inputs, return_tensors="pt")
    
    set_seed(555)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    waveform = outputs.waveform[0]
    scipy.io.wavfile.write("./tts_outputs/output.wav", rate=model.config.sampling_rate, data=waveform)
    
    return waveform
import json

def load_txt2img_workflow():
    with open('workflows/txt2img.json') as f:
        data=json.load(f)
        
    return data

def load_txt2img_sdxl_workflow():
    with open('workflows/txt2img_sdxl.json') as f:
        data=json.load(f)
        
    return data
    
def load_img2img_workflow():
    with open('workflows/img2img.json') as f:
        data=json.load(f)
        
    return data
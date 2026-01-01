import json

def load_txt2img_workflow():
    with open('workflows/txt2img.json') as f:
        data=json.load(f)
        
    return data

def load_txt2img_sdxl_workflow():
    with open('workflows/txt2img_sdxl.json') as f:
        data=json.load(f)
        
    return data

def load_txt2img_sdxl_with_refiner_workflow():
    with open('workflows/txt2img_sdxl_with_refiner.json') as f:
        data=json.load(f)
        
    return data
    
def load_txt2img_workflow_clip_skip():
    with open('workflows/txt2img_clip_skip.json') as f:
        data=json.load(f)
        
    return data

def load_txt2img_sdxl_workflow_clip_skip():
    with open('workflows/txt2img_sdxl_clip_skip.json') as f:
        data=json.load(f)
        
    return data

def load_txt2img_sdxl_with_refiner_workflow_clip_skip():
    with open('workflows/txt2img_sdxl_with_refiner_clip_skip.json') as f:
        data=json.load(f)
        
    return data

def load_img2img_workflow():
    with open('workflows/img2img.json') as f:
        data=json.load(f)
        
    return data

def load_img2img_sdxl_workflow():
    with open('workflows/img2img_sdxl.json') as f:
        data=json.load(f)

    return data

def load_img2img_sdxl_with_refiner_workflow():
    with open('workflows/img2img_sdxl_with_refiner.json') as f:
        data=json.load(f)

    return data

def load_img2img_workflow_clip_skip():
    with open('workflows/img2img_clip_skip.json') as f:
        data=json.load(f)

    return data

def load_img2img_sdxl_workflow_clip_skip():
    with open('workflows/img2img_sdxl_clip_skip.json') as f:
        data=json.load(f)

    return data

def load_img2img_sdxl_with_refiner_workflow_clip_skip():
    with open('workflows/img2img_sdxl_with_refiner_clip_skip.json') as f:
        data=json.load(f)

    return data

def load_inpaint_workflow():
    with open('workflows/inpaint.json') as f:
        data=json.load(f)
        
    return data

def load_inpaint_sdxl_workflow():
    with open('workflows/inpaint_sdxl.json') as f:
        data=json.load(f)
        
    return data

def load_inpaint_sdxl_with_refiner_workflow():
    with open('workflows/inpaint_sdxl_with_refiner.json') as f:
        data=json.load(f)
        
    return data

def load_inpaint_workflow_clip_skip():
    with open('workflows/inpaint_clip_skip.json') as f:
        data=json.load(f)
        
    return data

def load_inpaint_sdxl_workflow_clip_skip():
    with open('workflows/inpaint_sdxl_clip_skip.json') as f:
        data=json.load(f)
        
    return data

def load_inpaint_sdxl_with_refiner_workflow_clip_skip():
    with open('workflows/inpaint_sdxl_with_refiner_clip_skip.json') as f:
        data=json.load(f)
        
    return data

def load_image2video_wan_workflow():
    with open('workflows/image2video_wan.json') as f:
        data=json.load(f)
        
    return data
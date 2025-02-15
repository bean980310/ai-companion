from PIL import Image
import numpy as np
from typing import TypedDict

class ImageData(TypedDict):
    background: np.ndarray
    layers: list[np.ndarray]
    
def combine_masks(im_1: np.ndarray, im_2: np.ndarray):
    """
    Layers images on top of each other, with the foreground image overwriting the background.
    params:
        im_1: background image
        im_2: foreground image
    return:
        combined image
    """
    if im_1.shape != im_2.shape:
        raise ValueError("Images must have the same dimensions")

    # replace rgb values of the background with the foreground where foreground is not fully transparent
    foreground_mask = im_2[:, :, 3] > 0
    im_1[foreground_mask] = im_2[foreground_mask]

    return im_1

def make_grey(image: Image.Image, grey_value: int = 128) -> Image.Image:
    """
    Converts opaque pixels within an image to a static grey value.
    params:
        image: RGBA image
    return:
        greyscale image
    """
    image = image.convert("RGBA")
    np_img = np.array(image, dtype=np.uint8)
    
    rgb = np_img[:, :, :3]
    alpha = np_img[:, :, 3]

    opaque_mask = alpha == 255

    # arbitrary grey value

    # apply grey values to the opaque pixels
    rgb[opaque_mask] = np.array([grey_value, grey_value, grey_value], dtype=np.uint8)
    # 다시 합치기
    combined = np.dstack((rgb, alpha))
    return Image.fromarray(combined)

def combine_image_and_mask(image: Image.Image, mask: Image.Image, background_color=(255, 255, 255)) -> Image.Image:
    """
    이미지와 마스크를 블렌딩하여 합성 이미지(composite)를 생성합니다.
    
    매개변수:
      image: 원본 이미지 (PIL.Image)
      mask: 마스크 이미지 (PIL.Image). 
            - 마스크의 픽셀값이 255(white)이면 inpaint 영역으로 간주되어 background_color로 대체됩니다.
            - 0(black)이면 원본 이미지 영역이 그대로 유지됩니다.
      background_color: inpaint 영역에 적용할 배경 색상 (기본은 흰색).
    
    동작:
      1. image는 RGB 모드로 변환하고, mask는 그레이스케일(L)로 변환합니다.
      2. mask를 [0, 1] 범위의 부동소수점 배열로 변환한 후, 이를 반전(invert)시켜
         원본 이미지 픽셀이 보존되어야 하는 영역(검정→1)과 배경이 적용되어야 하는 영역(흰색→0)을 구분합니다.
      3. 최종 composite = (alpha * image) + ((1 - alpha) * background)를 계산합니다.
    
    반환:
      합성된 이미지 (PIL.Image)
    """
    # 1. 모드 변환
    image = image.convert("RGB")
    mask = mask.convert("L")
    
    # 2. NumPy 배열로 변환: mask 값 [0, 255] → [0,1]
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    # 여기서, 우리가 가정하는 것은 마스크가 흰색(1)이면 inpaint할 영역이므로, 그 영역은 배경 색상이 적용되어야 함.
    # 따라서, 원본 이미지 영역을 유지하려면 alpha = 1 - mask_np가 되어야 함.
    alpha = 1.0 - mask_np  # 원본 이미지 영역: mask==0 → alpha=1, inpaint 영역: mask==1 → alpha=0
    
    # 3. 원본 이미지와 배경 이미지 배열 생성
    image_np = np.array(image, dtype=np.float32)
    background_np = np.full_like(image_np, background_color, dtype=np.float32)
    
    # 4. 합성: 각 픽셀마다 alpha * image + (1-alpha) * background
    composite_np = (alpha[..., None] * image_np + (1.0 - alpha[..., None]) * background_np)
    composite_np = np.clip(composite_np, 0, 255).astype(np.uint8)
    
    # 5. PIL.Image로 복원
    composite_image = Image.fromarray(composite_np)
    return composite_image
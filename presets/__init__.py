from .ai_assistant_preset import AI_ASSISTANT_PRESET
# from .minami_asuka_char_preset import MINAMI_ASUKA_PRESET
# from .makotono_aoi_char_preset import MAKOTONO_AOI_PRESET
# from .aino_koito_char_preset import AINO_KOITO_PRESET
# from .aria_princess_fate_char_preset import ARIA_PRINCESS_FATE_PRESET
# from .aria_prince_fate_char_preset import ARIA_PRINCE_FATE_PRESET
from .character_preset import MINAMI_ASUKA_PRESET, MAKOTONO_AOI_PRESET, AINO_KOITO_PRESET, ARIA_PRINCESS_FATE_PRESET, ARIA_PRINCE_FATE_PRESET, WANG_MEI_LING_PRESET, MISTY_LANE_PRESET, LILY_EMPRESS_PRESET, CHOI_YUNA_PRESET, CHOI_YURI_PRESET
from .stable_diffusion_image_generator_preset import SD_IMAGE_GENERATOR_PRESET

# system_name = {
#     "ai_assistant": "AI 비서 (AI Assistant)",
#     "sd_image_generator": "Image Generator",
#     "minami_asuka": "미나미 아스카 (南飛鳥, みなみあすか, Minami Asuka)",
#     "makotono_aoi": "마코토노 아오이 (真琴乃葵, まことのあおい, Makotono Aoi)",
#     "aino_koito": "아이노 코이토 (愛野小糸, あいのこいと, Aino Koito)",
#     "aria_princess_fate": "아리아 프린세스 페이트 (アリア·プリンセス·フェイト, Aria Princess Fate)",
#     "aria_prince_fate": "아리아 프린스 페이트 (アリア·プリンス·フェイト, Aria Prince Fate)",
#     "wang_mei_ling": "왕 메이린 (王美玲, ワン·メイリン, Wang Mei-Ling)",
#     "misty_lane": "미스티 레인 (ミスティ·レーン, Misty Lane)",
#     "lily_empress": '릴리 엠프레스 (リリー·エンプレス, Lily Empress)',
#     "choi_yuna": "최유나 (崔有娜, チェ·ユナ, Choi Yuna)",
#     "choi_yuri": "최유리 (崔有莉, チェ·ユリ, Choi Yuri)"
# }

# system_preset = {
#     "ai_assistant_system_message": AI_ASSISTANT_PRESET,
#     "sd_image_generator_system_message": SD_IMAGE_GENERATOR_PRESET,
#     "minami_asuka_system_message": MINAMI_ASUKA_PRESET,
#     "makotono_aoi_system_message": MAKOTONO_AOI_PRESET,
#     "aino_koito_system_message": AINO_KOITO_PRESET,
#     "aria_princess_fate_system_message": ARIA_PRINCESS_FATE_PRESET,
#     "aria_prince_fate_system_message": ARIA_PRINCE_FATE_PRESET,
#     "wang_mei_ling_system_message": WANG_MEI_LING_PRESET,
#     "misty_lane_system_message": MISTY_LANE_PRESET,
#     'lily_empress_system_message': LILY_EMPRESS_PRESET,
#     "choi_yuna_system_message": CHOI_YUNA_PRESET,
#     "choi_yuri_system_message": CHOI_YURI_PRESET
# }

character_key = ["ai_assistant_system_message", "sd_image_generator_system_message", "minami_asuka_system_message", "makotono_aoi_system_message", "aino_koito_system_message", "aria_princess_fate_system_message", "aria_prince_fate_system_message", "wang_mei_ling_system_message", "misty_lane_system_message", "lily_empress_system_message", "choi_yuna_system_message", "choi_yuri_system_message"]

CHARACTER_LIST: list[dict[str, str]] = [AI_ASSISTANT_PRESET, SD_IMAGE_GENERATOR_PRESET, MINAMI_ASUKA_PRESET, MAKOTONO_AOI_PRESET, AINO_KOITO_PRESET, ARIA_PRINCESS_FATE_PRESET, ARIA_PRINCE_FATE_PRESET, WANG_MEI_LING_PRESET, MISTY_LANE_PRESET, LILY_EMPRESS_PRESET, CHOI_YUNA_PRESET, CHOI_YURI_PRESET]

__all__=[
    'AI_ASSISTANT_PRESET',
    'MINAMI_ASUKA_PRESET',
    'MAKOTONO_AOI_PRESET',
    'AINO_KOITO_PRESET',
    "ARIA_PRINCESS_FATE_PRESET",
    "ARIA_PRINCE_FATE_PRESET",
    "WANG_MEI_LING_PRESET",
    "MISTY_LANE_PRESET",
    "LILY_EMPRESS_PRESET",
    "CHOI_YUNA_PRESET",
    "CHOI_YURI_PRESET",
    "SD_IMAGE_GENERATOR_PRESET"
]
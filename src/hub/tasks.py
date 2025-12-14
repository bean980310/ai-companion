MULTIMODAL=["audio-text-to-text", "image-text-to-text", "visual-question-answering", "document-question-answering", "video-text-to-text", "visual-document-retrieval", "any-to-any"]

COMPUTER_VISION=["depth-estimation", "image-classification", "object-detection", "image-segmentation", "text-to-image", "image-to-text", "image-to-image", "image-to-video", "unconditional-image-generation", "video-classification", "text-to-video", "zero-shot-image-classification", "mask-generation", "zero-shot-object-detection", "text-to-3d", "image-to-3d", "image-feature-extraction", "keypoint-detection", "video-to-video"]

NETURAL_LANGUAGE_PROCESSING=["text-classification", "token-classification", "table-question-answering", "zero-shot-classification", "translation", "summerization", "feature-extraction", "text-generation", "text2text-generation", "fill-mask", "sentence-similarity", "text-ranking"]

AUDIO=["text-to-speech", "text-to-audio", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "voice-activity-detection"]

TABULAR=["tabular-classification", "tabular-regression", "time-series-forecasting"]

REINFORCEMENT_LEARNING = ["reinforcement-learning", 'robotics']

OTHER = ['graph-machine-learning']

TASKS = []

TASKS.append("None")

for t in (MULTIMODAL):
    TASKS.append(t)
    
for t in COMPUTER_VISION:
    TASKS.append(t)
    
for t in NETURAL_LANGUAGE_PROCESSING:
    TASKS.append(t)
    
for t in AUDIO:
    TASKS.append(t)

for t in TABULAR:
    TASKS.append(t)

for t in REINFORCEMENT_LEARNING:
    TASKS.append(t)

for t in OTHER:
    TASKS.append(t)


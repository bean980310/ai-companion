MULTIMODAL=["audio-text-to-text", "image-text-to-text", "visual-question-answering", "document-question-answering", "video-text-to-text", "visual-document-retrieval", "any-to-any"]

COMPUTER_VISION=["depth-estimation", "image-classification", "object-detection", "image-segmentation", "text-to-image", "image-to-text", "image-to-image", "image-to-video", "unconditional-image-generation", "video-classification", "text-to-video", "zero-shot-image-classification", "mask-generation", "zero-shot-object-detection", "text-to-3d", "image-to-3d", "image-feature-extraction", "keypoint-detection"]

NETURAL_LANGUAGE_PROCESSING=["text-classification", "token-classification", "table-question-answering", "zero-shot-classification", "translation", "summerization", "feature-extraction", "text-generation", "text2text-generation", "fill-mask", "sentence-similarity"]

AUDIO=["text-to-speech", "text-to-audio", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "voice-activity-detection"]

TASKS = []

TASKS.append("None")

for i in range(len(MULTIMODAL)):
    TASKS.append(MULTIMODAL[i])
    
for i in range(len(COMPUTER_VISION)):
    TASKS.append(COMPUTER_VISION[i])
    
for i in range(len(NETURAL_LANGUAGE_PROCESSING)):
    TASKS.append(NETURAL_LANGUAGE_PROCESSING[i])
    
for i in range(len(AUDIO)):
    TASKS.append(AUDIO[i])
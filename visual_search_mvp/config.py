import torch

class Config:
    # Настройки модели
    MODEL_NAME = "ViT-B/32"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Настройки сегментации
    SEGMENTATION = {
        'num_segments': 50,
        'compactness': 10,
        'min_segment_size': 30
    }
    
    # Настройки поиска
    SEARCH = {
        'top_k': 5,
        'similarity_threshold': 0.6,
        'use_cache': True
    }
    
    # Пути
    PATHS = {
        'cache_dir': '.embedding_cache'
    }
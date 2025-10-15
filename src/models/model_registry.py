from typing import Dict, Type
from ..core.embedders import BaseEmbedder, CLIPEmbedder, DINOv2Embedder
from ..core.segmenters import BaseSegmenter, SLICSegmenter, ContourSegmenter

class ModelRegistry:
    EMBEDDERS: Dict[str, Type[BaseEmbedder]] = {
        "clip": CLIPEmbedder,
        "dinov2": DINOv2Embedder
    }
    
    SEGMENTERS: Dict[str, Type[BaseSegmenter]] = {
        "slic": SLICSegmenter,
        "contour": ContourSegmenter
    }
    
    @classmethod
    def get_embedder(cls, name: str, config: Dict) -> BaseEmbedder:
        if name not in cls.EMBEDDERS:
            raise ValueError(f"Embedder {name} not found. Available: {list(cls.EMBEDDERS.keys())}")
        return cls.EMBEDDERS[name](config)
    
    @classmethod
    def get_segmenter(cls, name: str, **kwargs) -> BaseSegmenter:
        if name not in cls.SEGMENTERS:
            raise ValueError(f"Segmenter {name} not found. Available: {list(cls.SEGMENTERS.keys())}")
        return cls.SEGMENTERS[name](**kwargs)
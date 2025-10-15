from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
from PIL import Image
import numpy as np

class BaseEmbedder(ABC):
    """Abstract base class for all embedders"""
    
    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        pass
    
    @abstractmethod
    def encode(self, image_tensor: torch.Tensor) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass

class CLIPEmbedder(BaseEmbedder):
    def __init__(self, model_config: Dict):
        import clip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess_fn = clip.load(
            model_config["name"], 
            device=self.device
        )
        self.embedding_dim = model_config["embedding_dim"]
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.preprocess_fn(image).unsqueeze(0).to(self.device)
    
    def encode(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
        return features.cpu().numpy().flatten()
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim

class DINOv2Embedder(BaseEmbedder):
    def __init__(self, model_config: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            'facebookresearch/dinov2', 
            model_config["name"]
        ).to(self.device)
        self.model.eval()
        self.embedding_dim = model_config["embedding_dim"]
        
        # Setup transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=model_config["preprocessing"]["mean"],
                std=model_config["preprocessing"]["std"]
            )
        ])
    
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def encode(self, image_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.cpu().numpy().flatten()
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
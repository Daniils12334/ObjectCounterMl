import clip
import torch
from PIL import Image
import numpy as np
import os

class CLIPModel:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"✅ CLIP model '{model_name}' loaded on {device}")
    
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        """Получаем эмбеддинг для изображения с кэшированием"""
        try:
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
            return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return None
    
    def batch_get_embeddings(self, images: list) -> list:
        """Обрабатываем несколько изображений за раз - БЫСТРЕЕ!"""
        if not images:
            return []
            
        try:
            # Подготавливаем батч
            image_tensors = torch.cat([self.preprocess(img).unsqueeze(0) for img in images]).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model.encode_image(image_tensors)
            
            return [emb.cpu().numpy().flatten() for emb in embeddings]
            
        except Exception as e:
            print(f"❌ Batch processing error: {e}")
            return [None] * len(images)
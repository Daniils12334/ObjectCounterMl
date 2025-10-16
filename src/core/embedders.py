import torch
import clip
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Загрузка CLIP модели {model_name} на {self.device}...")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            logger.info(f"CLIP модель {model_name} загружена успешно")
        except Exception as e:
            logger.error(f"Ошибка загрузки CLIP модели: {e}")
            raise
    
    def embed_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Получение эмбеддинга для изображения
        
        Args:
            image: входное изображение (BGR)
            
        Returns:
            Тензор с эмбеддингом
        """
        try:
            # Проверяем что изображение не пустое
            if image.size == 0:
                logger.warning("Пустое изображение для эмбеддинга")
                return torch.zeros(512).to(self.device)
                
            # Конвертация BGR to RGB and to PIL
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            pil_image = Image.fromarray(image_rgb)
            
            # Предобработка
            processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(processed_image)
                
            return image_features.squeeze()
            
        except Exception as e:
            logger.error(f"Ошибка при получении эмбеддинга: {e}")
            # Возвращаем нулевой вектор в случае ошибки
            return torch.zeros(512).to(self.device)
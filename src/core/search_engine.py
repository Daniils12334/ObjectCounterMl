import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple
import logging

from ..core.segmenters import Segmenter
from ..core.embedders import Embedder
from ..utils.visualization import draw_results

logger = logging.getLogger(__name__)

class VisualSearchEngine:
    def __init__(self, segmenter: Segmenter, embedder: Embedder, similarity_threshold: float = 0.75):
        self.segmenter = segmenter
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        
    def search_similar_objects(self, scene_image: np.ndarray, query_image: np.ndarray, 
                             min_area: int = 1000, max_area: int = 50000) -> Dict:
        """
        Поиск объектов на сцене, похожих на query_image
        
        Args:
            scene_image: изображение сцены (BGR)
            query_image: эталонное изображение объекта (BGR)
            min_area: минимальная площадь объекта
            max_area: максимальная площадь объекта
            
        Returns:
            Словарь с результатами поиска
        """
        try:
            # 1. Сегментация объектов на сцене
            logger.info("Сегментация объектов на сцене...")
            scene_objects = self.segmenter.segment(scene_image)
            
            # 2. Фильтрация объектов по размеру
            filtered_objects = []
            for obj in scene_objects:
                area = obj['bbox'][2] * obj['bbox'][3]  # width * height
                if min_area <= area <= max_area:
                    filtered_objects.append(obj)
            
            logger.info(f"Найдено {len(filtered_objects)} объектов после фильтрации")
            
            if not filtered_objects:
                return {"matches": [], "query_embedding": None, "scene_objects": []}
            
            # 3. Получение эмбеддинга для эталонного изображения
            logger.info("Получение эмбеддинга для эталонного изображения...")
            query_embedding = self.embedder.embed_image(query_image)
            
            # 4. Поиск похожих объектов
            logger.info("Поиск похожих объектов...")
            matches = self._find_similar_objects(scene_image, filtered_objects, query_embedding)
            
            return {
                "matches": matches,
                "query_embedding": query_embedding,
                "scene_objects": filtered_objects,
                "total_objects": len(filtered_objects)
            }
            
        except Exception as e:
            logger.error(f"Ошибка при поиске объектов: {e}")
            raise
    
    def _find_similar_objects(self, scene_image: np.ndarray, objects: List[Dict], 
                            query_embedding: torch.Tensor) -> List[Dict]:
        """Поиск объектов с похожими эмбеддингами"""
        matches = []
        
        for i, obj in enumerate(objects):
            # Извлечение региона объекта
            x, y, w, h = obj['bbox']
            crop = scene_image[y:y+h, x:x+w]
            
            if crop.size == 0:
                continue
                
            # Получение эмбеддинга для объекта
            obj_embedding = self.embedder.embed_image(crop)
            
            # Вычисление косинусного сходства
            similarity = self._cosine_similarity(query_embedding, obj_embedding)
            
            if similarity >= self.similarity_threshold:
                match_info = {
                    'object_id': i,
                    'bbox': obj['bbox'],
                    'similarity': similarity,
                    'mask': obj.get('mask', None),
                    'crop': crop
                }
                matches.append(match_info)
        
        # Сортировка по убыванию сходства
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
    
    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Вычисление косинусного сходства между векторами"""
        a_norm = a / a.norm(dim=-1, keepdim=True)
        b_norm = b / b.norm(dim=-1, keepdim=True)
        return (a_norm @ b_norm.T).item()
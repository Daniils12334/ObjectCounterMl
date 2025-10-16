import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from .color_analyzer import ColorAnalyzer
from .shape_analyzer import ShapeAnalyzer
from .multi_scale_searcher import MultiScaleSearcher

logger = logging.getLogger(__name__)

class ObjectCounter:
    def __init__(self, color_tolerance: int = 50, min_similarity: float = 0.6):
        self.color_analyzer = ColorAnalyzer(tolerance=color_tolerance)
        self.shape_analyzer = ShapeAnalyzer()
        self.searcher = MultiScaleSearcher()
        self.min_similarity = min_similarity
        
    def analyze_query(self, query_image: np.ndarray) -> Dict:
        """
        Анализ эталонного изображения
        """
        # Анализ цвета
        avg_color, color_range = self.color_analyzer.extract_dominant_color(query_image)
        color_distribution = self.color_analyzer.analyze_color_distribution(query_image)
        
        # Создаем маску по цвету (предполагаем, что объект занимает большую часть query)
        color_mask = self.color_analyzer.create_color_mask(query_image, color_range)
        
        # Анализ формы
        shape_features = self.shape_analyzer.extract_shape_features(color_mask)
        
        return {
            'color_range': color_range,
            'avg_color': avg_color,
            'color_distribution': color_distribution,
            'shape': shape_features,
            'query_mask': color_mask
        }
    
    def count_objects(self, scene_image: np.ndarray, query_image: np.ndarray) -> Dict:
        """
        Основной метод подсчета объектов
        """
        logger.info("Анализ эталонного изображения...")
        query_features = self.analyze_query(query_image)
        
        logger.info("Поиск объектов на сцене...")
        all_detections = []
        
        # Поиск на разных масштабах
        for scale in self.searcher.scales:
            logger.info(f"Поиск на масштабе {scale}x...")
            detections = self.searcher.search_at_scale(
                scene_image, query_features, self.color_analyzer, self.shape_analyzer, scale)
            all_detections.extend(detections)
        
        logger.info(f"Найдено кандидатов: {len(all_detections)}")
        
        # Объединение перекрывающихся обнаружений
        merged_objects = self.searcher.merge_overlapping_boxes(all_detections)
        
        # Фильтрация по минимальному сходству
        final_objects = [obj for obj in merged_objects if obj['similarity'] >= self.min_similarity]
        
        logger.info(f"После объединения и фильтрации: {len(final_objects)} объектов")
        
        return {
            'objects': final_objects,
            'query_features': query_features,
            'total_found': len(final_objects),
            'search_stats': {
                'total_detections': len(all_detections),
                'merged_detections': len(merged_objects),
                'scales_used': len(self.searcher.scales)
            }
        }
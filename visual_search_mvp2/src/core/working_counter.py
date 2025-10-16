import cv2
import numpy as np
from typing import List, Dict
import logging
from .color_analyzer_fixed import ColorAnalyzerFixed
from .simple_detector import SimpleDetector

logger = logging.getLogger(__name__)

class WorkingCounter:
    def __init__(self, color_tolerance: int = 80, min_similarity: float = 0.3):
        self.color_analyzer = ColorAnalyzerFixed(tolerance=color_tolerance)
        self.detector = SimpleDetector()
        self.min_similarity = min_similarity
    
    def analyze_query_simple(self, query_image: np.ndarray) -> Dict:
        """
        Простой анализ эталонного изображения
        """
        logger.info("Анализ эталонного изображения...")
        
        # Просто берем средний цвет всего изображения
        # Для лучших результатов query должен быть объектом на однородном фоне
        avg_color = np.mean(query_image, axis=(0, 1)).astype(np.uint8)
        
        logger.info(f"Определен цвет эталона: {avg_color}")
        
        return {
            'color': avg_color,
            'color_range': np.array([
                np.maximum(avg_color - 80, 0),
                np.minimum(avg_color + 80, 255)
            ])
        }
    
    def count_objects_working(self, scene_image: np.ndarray, query_image: np.ndarray) -> Dict:
        """
        Работающая версия подсчета объектов
        """
        # Анализ эталона
        query_features = self.analyze_query_simple(query_image)
        
        # Поиск объектов по цвету
        found_objects = self.detector.find_objects_by_color(
            scene_image, query_features['color'], color_tolerance=100)
        
        # Объединение перекрывающихся
        merged_objects = self.detector.merge_overlapping_objects(found_objects)
        
        # Фильтрация по минимальному сходству
        final_objects = [obj for obj in merged_objects if obj['similarity'] >= self.min_similarity]
        
        logger.info(f"Итоговое количество объектов: {len(final_objects)}")
        
        return {
            'objects': final_objects,
            'query_features': query_features,
            'total_found': len(final_objects),
            'search_stats': {
                'initial_detections': len(found_objects),
                'after_merging': len(merged_objects)
            }
        }
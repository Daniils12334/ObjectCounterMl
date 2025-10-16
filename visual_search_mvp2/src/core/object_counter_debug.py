import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from .color_analyzer_debug import ColorAnalyzerDebug
from .simple_searcher import SimpleSearcher

logger = logging.getLogger(__name__)

class ObjectCounterDebug:
    def __init__(self, color_tolerance: int = 80, min_similarity: float = 0.4):
        self.color_analyzer = ColorAnalyzerDebug(tolerance=color_tolerance)
        self.searcher = SimpleSearcher()
        self.min_similarity = min_similarity
        
    def analyze_query_debug(self, query_image: np.ndarray, debug_path: str = "query_analysis.jpg") -> Dict:
        """
        Детальный анализ эталонного изображения с визуализацией
        """
        logger.info("Детальный анализ эталонного изображения...")
        
        # Анализ цвета
        avg_color, color_range = self.color_analyzer.extract_dominant_color(query_image)
        
        # Анализ формы
        color_mask = self.color_analyzer.create_color_mask_hsv(query_image, color_range)
        shape_features = self.extract_simple_shape_features(color_mask)
        
        # Визуализация анализа
        debug_img = self.visualize_query_analysis(query_image, color_mask, avg_color, color_range)
        cv2.imwrite(debug_path, debug_img)
        logger.info(f"Анализ запроса сохранен в: {debug_path}")
        
        return {
            'color_range': color_range,
            'avg_color': avg_color,
            'shape': shape_features,
            'query_mask': color_mask
        }
    
    def extract_simple_shape_features(self, mask: np.ndarray) -> Dict:
        """Извлечение простых признаков формы"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'aspect_ratio': 1.0, 'extent': 1.0}
        
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        area = cv2.contourArea(main_contour)
        
        aspect_ratio = w / h if h > 0 else 1.0
        extent = area / (w * h) if w * h > 0 else 1.0
        
        return {
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'area': area,
            'contour': main_contour
        }
    
    def visualize_query_analysis(self, query: np.ndarray, mask: np.ndarray, 
                               avg_color: np.ndarray, color_range: np.ndarray) -> np.ndarray:
        """Визуализация анализа эталонного изображения"""
        # Создаем коллаж
        height = max(query.shape[0], 200)
        width = query.shape[1] * 2 + 20
        
        result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Исходное изображение
        result[0:query.shape[0], 0:query.shape[1]] = query
        
        # Маска
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result[0:query.shape[0], query.shape[1] + 10:query.shape[1] * 2 + 10] = mask_bgr
        
        # Информация о цвете
        color_block = np.full((50, 200, 3), avg_color, dtype=np.uint8)
        result[query.shape[0] + 10:query.shape[0] + 60, 10:210] = color_block
        
        # Текст
        color_text = f"Color: {avg_color}"
        range_text = f"Range: {color_range[0]} - {color_range[1]}"
        
        cv2.putText(result, color_text, (10, query.shape[0] + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, range_text, (10, query.shape[0] + 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def count_objects_debug(self, scene_image: np.ndarray, query_image: np.ndarray, 
                          debug_prefix: str = "debug") -> Dict:
        """
        Подсчет объектов с детальной отладкой
        """
        # Анализ эталона
        query_features = self.analyze_query_debug(query_image, f"{debug_prefix}_query.jpg")
        
        # Анализ цветового покрытия сцены
        color_analysis_img, full_mask, coverage = self.color_analyzer.analyze_color_compatibility(
            scene_image, query_features['color_range'])
        cv2.imwrite(f"{debug_prefix}_color_coverage.jpg", color_analysis_img)
        
        logger.info(f"Цветовое покрытие сцены: {coverage:.2f}%")
        
        if coverage < 0.1:
            logger.warning("Очень низкое цветовое покрытие! Возможно, объекты такого цвета отсутствуют.")
        
        # Поиск объектов
        found_objects = self.searcher.search_objects(scene_image, query_features, self.color_analyzer)
        
        # Фильтрация по минимальному сходству
        final_objects = [obj for obj in found_objects if obj['similarity'] >= self.min_similarity]
        
        logger.info(f"Итоговое количество объектов: {len(final_objects)}")
        
        return {
            'objects': final_objects,
            'query_features': query_features,
            'total_found': len(final_objects),
            'color_coverage': coverage,
            'debug_images': {
                'query_analysis': f"{debug_prefix}_query.jpg",
                'color_coverage': f"{debug_prefix}_color_coverage.jpg"
            }
        }
import cv2
import numpy as np
from typing import List, Dict, Tuple

class ShapeAnalyzer:
    def __init__(self, min_area: int = 100, max_area: int = 50000):
        self.min_area = min_area
        self.max_area = max_area
    
    def extract_shape_features(self, mask: np.ndarray) -> Dict:
        """
        Извлечение признаков формы из бинарной маски
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Берем самый большой контур
        main_contour = max(contours, key=cv2.contourArea)
        
        # Базовые параметры
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Моменты Хаара для формы
        moments = cv2.moments(main_contour)
        
        # Отношение сторон bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Коэффициент круглости
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Hu moments (инвариантные к масштабу и вращению)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Аппроксимация контура
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        vertices = len(approx)
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'hu_moments': hu_moments,
            'vertices': vertices,
            'bbox': (x, y, w, h),
            'contour': main_contour
        }
    
    def compare_shapes(self, shape1: Dict, shape2: Dict) -> float:
        """
        Сравнение двух форм по их признакам
        Возвращает оценку сходства (0-1)
        """
        # Сравнение Hu moments (логарифмическая разница)
        hu_diff = 0
        for i in range(7):
            hu1 = np.sign(shape1['hu_moments'][i]) * np.log10(np.abs(shape1['hu_moments'][i]) + 1e-10)
            hu2 = np.sign(shape2['hu_moments'][i]) * np.log10(np.abs(shape2['hu_moments'][i]) + 1e-10)
            hu_diff += abs(hu1 - hu2)
        hu_diff /= 7
        
        # Сравнение отношения сторон
        aspect_diff = abs(shape1['aspect_ratio'] - shape2['aspect_ratio'])
        
        # Сравнение круглости
        circularity_diff = abs(shape1['circularity'] - shape2['circularity'])
        
        # Общая оценка сходства (1 - разница)
        hu_similarity = max(0, 1 - hu_diff)
        aspect_similarity = max(0, 1 - aspect_diff)
        circularity_similarity = max(0, 1 - circularity_diff)
        
        # Взвешенное среднее
        total_similarity = 0.5 * hu_similarity + 0.3 * aspect_similarity + 0.2 * circularity_similarity
        
        return total_similarity
    
    def filter_by_shape(self, shape_features: Dict, reference_shape: Dict, 
                       min_similarity: float = 0.6) -> bool:
        """
        Фильтрация по сходству формы
        """
        similarity = self.compare_shapes(shape_features, reference_shape)
        return similarity >= min_similarity
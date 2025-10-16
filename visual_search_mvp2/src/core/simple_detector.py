import cv2
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SimpleDetector:
    def __init__(self, min_area: int = 100, max_area: int = 50000):
        self.min_area = min_area
        self.max_area = max_area
    
    def find_objects_by_color(self, scene: np.ndarray, query_color: np.ndarray, 
                            color_tolerance: int = 80) -> List[Dict]:
        """
        Простой поиск объектов по цвету
        """
        # Создаем цветовую маску
        lower_bound = np.maximum(query_color - color_tolerance, 0)
        upper_bound = np.minimum(query_color + color_tolerance, 255)
        
        mask = cv2.inRange(scene, lower_bound, upper_bound)
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Вычисляем сходство цвета
            roi = scene[y:y+h, x:x+w]
            color_similarity = self.calculate_color_similarity(roi, query_color)
            
            # Вычисляем сходство формы (простое)
            shape_similarity = self.calculate_shape_similarity(contour)
            
            # Общее сходство
            total_similarity = 0.7 * color_similarity + 0.3 * shape_similarity
            
            objects.append({
                'bbox': (x, y, w, h),
                'similarity': total_similarity,
                'color_similarity': color_similarity,
                'shape_similarity': shape_similarity,
                'area': area,
                'contour': contour
            })
        
        # Сортируем по сходству
        objects.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Найдено объектов по цвету: {len(objects)}")
        return objects
    
    def calculate_color_similarity(self, roi: np.ndarray, target_color: np.ndarray) -> float:
        """Вычисление сходства цвета"""
        if roi.size == 0:
            return 0.0
        
        # Средний цвет в ROI
        avg_color = np.mean(roi, axis=(0, 1))
        
        # Евклидово расстояние между цветами
        color_diff = np.linalg.norm(avg_color - target_color)
        max_diff = np.sqrt(3 * (255**2))  # максимальное возможное расстояние
        
        # Нормализуем в диапазон 0-1 (1 - идеальное совпадение)
        similarity = 1.0 - (color_diff / max_diff)
        
        return max(0.0, min(1.0, similarity))
    
    def calculate_shape_similarity(self, contour) -> float:
        """Простое вычисление сходства формы"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Коэффициент круглости
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # Нормализуем (идеальный круг = 1.0)
        return min(1.0, circularity)
    
    def merge_overlapping_objects(self, objects: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Объединение перекрывающихся объектов"""
        if not objects:
            return []
        
        merged = []
        used = set()
        
        for i, obj1 in enumerate(objects):
            if i in used:
                continue
            
            current_box = list(obj1['bbox'])
            current_similarity = obj1['similarity']
            count = 1
            
            for j, obj2 in enumerate(objects[i+1:], i+1):
                if j in used:
                    continue
                
                if self.calculate_iou(current_box, obj2['bbox']) > iou_threshold:
                    # Объединяем боксы
                    x1 = min(current_box[0], obj2['bbox'][0])
                    y1 = min(current_box[1], obj2['bbox'][1])
                    x2 = max(current_box[0] + current_box[2], obj2['bbox'][0] + obj2['bbox'][2])
                    y2 = max(current_box[1] + current_box[3], obj2['bbox'][1] + obj2['bbox'][3])
                    
                    current_box = [x1, y1, x2 - x1, y2 - y1]
                    current_similarity = max(current_similarity, obj2['similarity'])
                    count += 1
                    used.add(j)
            
            merged.append({
                'bbox': tuple(current_box),
                'similarity': current_similarity,
                'merged_count': count
            })
            used.add(i)
        
        return merged
    
    def calculate_iou(self, box1, box2):
        """Вычисление Intersection over Union"""
        x1_1, y1_1, w1, h1 = box1
        x2_1, y2_1, w2, h2 = box2
        
        x1_2, y1_2 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x2_1 + w2, y2_1 + h2
        
        # Вычисляем пересечение
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        
        iou = intersection_area / (area1 + area2 - intersection_area)
        return iou
import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleSearcher:
    def __init__(self):
        self.scales = [2.0, 1.5, 1.0, 0.7, 0.5, 0.3]
        self.window_sizes = [150, 100, 70, 50, 30]
    
    def search_objects(self, scene: np.ndarray, query_features: Dict, 
                      color_analyzer, min_color_coverage: float = 0.1) -> List[Dict]:
        """
        Упрощенный поиск объектов - сначала по цвету, потом по форме
        """
        all_detections = []
        
        # Сначала находим все регионы с подходящим цветом
        color_mask = color_analyzer.create_color_mask_hsv(scene, query_features['color_range'])
        
        # Находим контуры в маске
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.info(f"Найдено контуров по цвету: {len(contours)}")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Фильтруем по размеру
            if area < 500 or area > 50000:
                continue
            
            # Получаем bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Вычисляем простые признаки формы
            aspect_ratio = w / h
            extent = area / (w * h)
            
            # Сравниваем с эталоном
            similarity = self.calculate_similarity(query_features['shape'], 
                                                 {'aspect_ratio': aspect_ratio, 'extent': extent})
            
            if similarity > 0.3:  # Низкий порог для начала
                # Вырезаем регион для более детального анализа
                crop = scene[y:y+h, x:x+w]
                if crop.size > 0:
                    # Уточняем сходство по цвету
                    color_similarity = self.calculate_color_similarity(crop, query_features, color_analyzer)
                    
                    total_similarity = 0.6 * color_similarity + 0.4 * similarity
                    
                    if total_similarity > 0.4:
                        all_detections.append({
                            'bbox': (x, y, w, h),
                            'similarity': total_similarity,
                            'color_similarity': color_similarity,
                            'shape_similarity': similarity,
                            'contour': contour
                        })
        
        # Сортируем по убыванию сходства
        all_detections.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Объединяем перекрывающиеся обнаружения
        merged = self.merge_detections(all_detections)
        
        logger.info(f"После объединения: {len(merged)} объектов")
        return merged
    
    def calculate_similarity(self, shape1: Dict, shape2: Dict) -> float:
        """Упрощенное вычисление сходства форм"""
        aspect_diff = 1 - min(1.0, abs(shape1.get('aspect_ratio', 1) - shape2.get('aspect_ratio', 1)))
        extent_diff = 1 - min(1.0, abs(shape1.get('extent', 1) - shape2.get('extent', 1)))
        
        return (aspect_diff + extent_diff) / 2
    
    def calculate_color_similarity(self, crop: np.ndarray, query_features: Dict, color_analyzer) -> float:
        """Вычисление сходства по цвету"""
        crop_color_mask = color_analyzer.create_color_mask_hsv(crop, query_features['color_range'])
        color_coverage = np.sum(crop_color_mask > 0) / (crop.shape[0] * crop.shape[1])
        
        return min(1.0, color_coverage * 3)  # Нормализуем
    
    def merge_detections(self, detections: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
        """Объединение перекрывающихся обнаружений"""
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            current_box = list(det1['bbox'])
            current_similarity = det1['similarity']
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                box2 = det2['bbox']
                iou = self.calculate_iou(current_box, box2)
                
                if iou > overlap_threshold:
                    # Объединяем
                    x1 = min(current_box[0], box2[0])
                    y1 = min(current_box[1], box2[1])
                    x2 = max(current_box[0] + current_box[2], box2[0] + box2[2])
                    y2 = max(current_box[1] + current_box[3], box2[1] + box2[3])
                    
                    current_box = [x1, y1, x2 - x1, y2 - y1]
                    current_similarity = max(current_similarity, det2['similarity'])
                    used.add(j)
            
            merged.append({
                'bbox': tuple(current_box),
                'similarity': current_similarity
            })
            used.add(i)
        
        return merged
    
    def calculate_iou(self, box1, box2):
        """Вычисление Intersection over Union"""
        x11, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x1_1, y1_1 = x1, y1
        x1_2, y1_2 = x1 + w1, y1 + h1
        x2_1, y2_1 = x2, y2
        x2_2, y2_2 = x2 + w2, y2 + h2
        
        # Вычисляем координаты пересечения
        x_left = max(x1_1, x2_1)
        y_top = max(y1_1, y2_1)
        x_right = min(x1_2, x2_2)
        y_bottom = min(y1_2, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        iou = intersection_area / (box1_area + box2_area - intersection_area)
        return iou
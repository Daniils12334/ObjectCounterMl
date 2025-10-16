import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class MultiScaleSearcher:
    def __init__(self, scales: List[float] = None, window_sizes: List[int] = None):
        """
        Инициализация поисковика с разными масштабами и размерами окон
        
        Args:
            scales: список масштабов для поиска (например [2.0, 1.5, 1.0, 0.75, 0.5, 0.25])
            window_sizes: размеры скользящего окна в пикселях
        """
        self.scales = scales or [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.125]
        self.window_sizes = window_sizes or [200, 100, 50, 25, 12]
        
    def search_at_scale(self, scene: np.ndarray, query_features: Dict, 
                       color_analyzer, shape_analyzer, scale: float) -> List[Dict]:
        """
        Поиск объектов на одном масштабе
        """
        # Масштабируем сцену
        if scale != 1.0:
            new_width = int(scene.shape[1] * scale)
            new_height = int(scene.shape[0] * scale)
            scaled_scene = cv2.resize(scene, (new_width, new_height))
        else:
            scaled_scene = scene
        
        found_objects = []
        
        # Поиск с разными размерами окон
        for window_size in self.window_sizes:
            step = max(1, window_size // 4)  # шаг = 1/4 размера окна
            
            for y in range(0, scaled_scene.shape[0] - window_size + 1, step):
                for x in range(0, scaled_scene.shape[1] - window_size + 1, step):
                    # Вырезаем окно
                    window = scaled_scene[y:y+window_size, x:x+window_size]
                    
                    # Анализ цвета
                    color_mask = color_analyzer.create_color_mask(window, query_features['color_range'])
                    
                    # Если есть достаточно цветных пикселей
                    if np.sum(color_mask) > 100:  # минимальная площадь
                        # Анализ формы
                        shape_features = shape_analyzer.extract_shape_features(color_mask)
                        
                        if shape_features and shape_analyzer.filter_by_shape(
                            shape_features, query_features['shape'], min_similarity=0.5):
                            
                            # Масштабируем координаты обратно к оригинальному размеру
                            orig_x = int(x / scale)
                            orig_y = int(y / scale)
                            orig_w = int(window_size / scale)
                            orig_h = int(window_size / scale)
                            
                            found_objects.append({
                                'bbox': (orig_x, orig_y, orig_w, orig_h),
                                'scale': scale,
                                'window_size': window_size,
                                'similarity': shape_analyzer.compare_shapes(
                                    shape_features, query_features['shape'])
                            })
        
        return found_objects
    
    def merge_overlapping_boxes(self, objects: List[Dict], overlap_threshold: float = 0.5) -> List[Dict]:
        """
        Объединение перекрывающихся bounding box
        """
        if not objects:
            return []
        
        # Сортируем по убыванию similarity
        objects.sort(key=lambda x: x['similarity'], reverse=True)
        
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
                
                box2 = obj2['bbox']
                # Вычисление IoU (Intersection over Union)
                x1 = max(current_box[0], box2[0])
                y1 = max(current_box[1], box2[1])
                x2 = min(current_box[0] + current_box[2], box2[0] + box2[2])
                y2 = min(current_box[1] + current_box[3], box2[1] + box2[3])
                
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = current_box[2] * current_box[3]
                    area2 = box2[2] * box2[3]
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > overlap_threshold:
                        # Объединяем боксы
                        current_box[0] = min(current_box[0], box2[0])
                        current_box[1] = min(current_box[1], box2[1])
                        current_box[2] = max(current_box[0] + current_box[2], box2[0] + box2[2]) - current_box[0]
                        current_box[3] = max(current_box[1] + current_box[3], box2[1] + box2[3]) - current_box[1]
                        current_similarity = max(current_similarity, obj2['similarity'])
                        count += 1
                        used.add(j)
            
            merged.append({
                'bbox': tuple(current_box),
                'similarity': current_similarity,
                'detections_merged': count
            })
            used.add(i)
        
        return merged
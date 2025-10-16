import cv2
import numpy as np
from typing import List, Dict

def draw_results(scene_image: np.ndarray, matches: List[Dict], 
                query_image: np.ndarray = None, show_similarity: bool = True) -> np.ndarray:
    """
    Визуализация результатов поиска
    
    Args:
        scene_image: изображение сцены
        matches: список найденных совпадений
        query_image: эталонное изображение (опционально)
        show_similarity: показывать значение сходства
        
    Returns:
        Изображение с визуализацией
    """
    result_image = scene_image.copy()
    
    # Цвета для разных уровней сходства
    def get_color(similarity: float) -> tuple:
        if similarity > 0.9:
            return (0, 255, 0)  # зеленый
        elif similarity > 0.8:
            return (0, 255, 255)  # желтый
        else:
            return (0, 165, 255)  # оранжевый
    
    # Рисуем bounding boxes для найденных объектов
    for i, match in enumerate(matches):
        x, y, w, h = match['bbox']
        similarity = match['similarity']
        color = get_color(similarity)
        
        # Прямоугольник
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
        
        if show_similarity:
            # Текст с similarity
            text = f"{similarity:.3f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Фон для текста
            cv2.rectangle(result_image, 
                         (x, y - text_size[1] - 10),
                         (x + text_size[0] + 10, y),
                         color, -1)
            
            # Текст
            cv2.putText(result_image, text, 
                       (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Если есть query image, добавляем его в результат
    if query_image is not None:
        h_scene, w_scene = scene_image.shape[:2]
        h_query, w_query = query_image.shape[:2]
        
        # Масштабируем query image чтобы вписать в левый верхний угол
        scale = min(200 / w_query, 200 / h_query)
        new_w = int(w_query * scale)
        new_h = int(h_query * scale)
        query_resized = cv2.resize(query_image, (new_w, new_h))
        
        # Добавляем рамку
        cv2.rectangle(query_resized, (0, 0), (new_w-1, new_h-1), (0, 255, 0), 2)
        
        # Вставляем в основное изображение
        result_image[10:10+new_h, 10:10+new_w] = query_resized
        
        # Подпись для query image
        cv2.putText(result_image, "Query", (10, 10+new_h+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_image
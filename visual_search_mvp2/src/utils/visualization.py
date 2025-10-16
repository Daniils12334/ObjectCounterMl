import cv2
import numpy as np
from typing import List, Dict

def draw_detections(scene_image: np.ndarray, objects: List[Dict], query_image: np.ndarray = None) -> np.ndarray:
    """
    Визуализация найденных объектов на сцене
    
    Args:
        scene_image: изображение сцены
        objects: список объектов с информацией о bbox и similarity
        query_image: эталонное изображение (опционально)
        
    Returns:
        Изображение с нарисованными bounding boxes
    """
    result_image = scene_image.copy()
    
    # Цвета в зависимости от уровня сходства
    for obj in objects:
        similarity = obj['similarity']
        bbox = obj['bbox']
        x, y, w, h = bbox
        
        # Выбор цвета: от красного (низкое сходство) к зеленому (высокое)
        green = int(255 * similarity)
        red = int(255 * (1 - similarity))
        color = (0, green, red)
        
        # Рисуем прямоугольник
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
        
        # Подписываем сходство
        label = f"{similarity:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_image, (x, y - label_size[1] - 10), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(result_image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Если есть query_image, вставляем его в левый верхний угол
    if query_image is not None:
        h_query, w_query = query_image.shape[:2]
        # Масштабируем, чтобы не был слишком большим
        scale = min(200 / w_query, 200 / h_query)
        new_w = int(w_query * scale)
        new_h = int(h_query * scale)
        query_resized = cv2.resize(query_image, (new_w, new_h))
        
        # Вставляем в левый верхний угол
        result_image[10:10+new_h, 10:10+new_w] = query_resized
        
        # Рамка и подпись
        cv2.rectangle(result_image, (10, 10), (10+new_w, 10+new_h), (0, 255, 0), 2)
        cv2.putText(result_image, "Query", (10, 10+new_h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_image

def draw_color_analysis(image: np.ndarray, color_range: np.ndarray) -> np.ndarray:
    """
    Визуализация анализа цвета
    
    Args:
        image: исходное изображение
        color_range: диапазон цветов [[min], [max]]
        
    Returns:
        Изображение с визуализацией цвета
    """
    # Создаем маску по цвету
    mask = cv2.inRange(image, color_range[0], color_range[1])
    
    # Применяем маску к изображению
    result = image.copy()
    result[mask == 0] = result[mask == 0] // 2  # затемняем не совпадающие области
    
    # Добавляем информацию о цвете
    avg_color = np.mean([color_range[0], color_range[1]], axis=0).astype(int)
    color_block = np.full((50, 200, 3), avg_color, dtype=np.uint8)
    
    # Комбинируем изображение с блоком цвета
    h, w = result.shape[:2]
    color_block_resized = cv2.resize(color_block, (w, 50))
    result = np.vstack([result, color_block_resized])
    
    # Добавляем текст
    text = f"Color Range: {color_range[0]} - {color_range[1]}"
    cv2.putText(result, text, (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result
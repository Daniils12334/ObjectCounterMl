import cv2
import numpy as np

def resize_image(image: np.ndarray, max_dimension: int = 800) -> np.ndarray:
    """Изменение размера изображения с сохранением пропорций"""
    h, w = image.shape[:2]
    
    if max(h, w) <= max_dimension:
        return image
    
    if h > w:
        new_h = max_dimension
        new_w = int(w * (max_dimension / h))
    else:
        new_w = max_dimension
        new_h = int(h * (max_dimension / w))
    
    return cv2.resize(image, (new_w, new_h))

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Улучшение изображения для лучшего анализа"""
    # Увеличение контраста
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced
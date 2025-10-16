import cv2
import numpy as np
from typing import Tuple, List, Dict

class ColorAnalyzer:
    def __init__(self, tolerance: int = 50):
        """
        Инициализация анализатора цвета
        
        Args:
            tolerance: допуск по цвету (+- значение для каждого канала)
        """
        self.tolerance = tolerance
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение доминирующего цвета из изображения
        
        Returns:
            (средний цвет, диапазон цветов [min, max])
        """
        if mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            pixels = masked_image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            return np.array([0, 0, 0]), np.array([[0, 0, 0], [255, 255, 255]])
        
        # Вычисляем средний цвет
        avg_color = np.mean(pixels, axis=0).astype(np.uint8)
        
        # Вычисляем диапазон с допуском
        lower_bound = np.maximum(avg_color - self.tolerance, 0)
        upper_bound = np.minimum(avg_color + self.tolerance, 255)
        
        return avg_color, np.array([lower_bound, upper_bound])
    
    def create_color_mask(self, image: np.ndarray, color_range: np.ndarray) -> np.ndarray:
        """
        Создание маски по диапазону цветов
        
        Args:
            image: BGR изображение
            color_range: [[B_min, G_min, R_min], [B_max, G_max, R_max]]
            
        Returns:
            Бинарная маска
        """
        lower = color_range[0]
        upper = color_range[1]
        
        # Создаем маску
        mask = cv2.inRange(image, lower, upper)
        
        # Морфологические операции для очистки маски
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def analyze_color_distribution(self, image: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Анализ распределения цвета
        """
        if mask is not None:
            pixels = image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            return {}
        
        # Конвертируем в HSV для лучшего анализа цвета
        hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        
        return {
            'bgr_mean': np.mean(pixels, axis=0),
            'bgr_std': np.std(pixels, axis=0),
            'hsv_mean': np.mean(hsv_pixels, axis=0),
            'hsv_std': np.std(hsv_pixels, axis=0)
        }
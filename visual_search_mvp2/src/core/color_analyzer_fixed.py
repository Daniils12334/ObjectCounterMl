import cv2
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class ColorAnalyzerFixed:
    def __init__(self, tolerance: int = 80):
        self.tolerance = tolerance
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение доминирующего цвета - ИСПРАВЛЕННАЯ ВЕРСИЯ
        """
        if mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            pixels = masked_image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            logger.warning("Нет пикселей для анализа цвета")
            return np.array([0, 0, 0]), np.array([[0, 0, 0], [255, 255, 255]])
        
        # Используем модуль для устранения переполнения
        avg_color = np.mean(pixels, axis=0).astype(np.uint8)
        
        # Вычисляем диапазон с допуском
        lower_bound = np.maximum(avg_color.astype(int) - self.tolerance, 0).astype(np.uint8)
        upper_bound = np.minimum(avg_color.astype(int) + self.tolerance, 255).astype(np.uint8)
        
        logger.debug(f"Доминирующий цвет: {avg_color}, диапазон: {lower_bound} - {upper_bound}")
        
        return avg_color, np.array([lower_bound, upper_bound])
    
    def create_color_mask_simple(self, image: np.ndarray, color_range: np.ndarray) -> np.ndarray:
        """
        Простое создание маски по цвету в BGR пространстве
        """
        lower = color_range[0]
        upper = color_range[1]
        
        # Простая маска в BGR
        mask = cv2.inRange(image, lower, upper)
        
        # Морфологические операции для очистки
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def create_color_mask_adaptive(self, image: np.ndarray, target_color: np.ndarray) -> np.ndarray:
        """
        Адаптивное создание маски на основе расстояния до целевого цвета
        """
        # Вычисляем евклидово расстояние до целевого цвета
        diff = image.astype(np.float32) - target_color.astype(np.float32)
        distance = np.sqrt(np.sum(diff**2, axis=2))
        
        # Нормализуем и создаем маску
        max_distance = np.sqrt(3 * (255**2))  # максимальное расстояние в цветовом пространстве
        similarity = 1 - (distance / max_distance)
        
        # Пороговая обработка
        mask = (similarity > 0.3).astype(np.uint8) * 255
        
        return mask
    
    def debug_color_detection(self, image: np.ndarray, color_range: np.ndarray, save_path: str = "color_debug.jpg"):
        """
        Отладочная функция для визуализации обнаружения цвета
        """
        # Создаем маску
        mask = self.create_color_mask_simple(image, color_range)
        
        # Применяем маску к изображению
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Создаем визуализацию
        h, w = image.shape[:2]
        debug_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Исходное изображение
        debug_img[0:h, 0:w] = image
        
        # Маска и результат
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        debug_img[0:h, w:w*2] = mask_bgr
        
        # Добавляем информацию
        coverage = np.sum(mask > 0) / (h * w) * 100
        text1 = f"Color range: {color_range[0]} - {color_range[1]}"
        text2 = f"Coverage: {coverage:.2f}%"
        
        cv2.putText(debug_img, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(save_path, debug_img)
        logger.info(f"Отладочное изображение цвета сохранено: {save_path}, покрытие: {coverage:.2f}%")
        
        return mask, coverage
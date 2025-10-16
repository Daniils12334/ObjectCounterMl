import cv2
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class ColorAnalyzerDebug:
    def __init__(self, tolerance: int = 80, hue_tolerance: int = 20):
        self.tolerance = tolerance
        self.hue_tolerance = hue_tolerance
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение доминирующего цвета с улучшенной логикой
        """
        if mask is not None:
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            pixels = masked_image[mask > 0]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) == 0:
            logger.warning("Нет пикселей для анализа цвета")
            return np.array([0, 0, 0]), np.array([[0, 0, 0], [255, 255, 255]])
        
        # Используем медиану вместо среднего (менее чувствительна к выбросам)
        median_color = np.median(pixels, axis=0).astype(np.uint8)
        
        # Расширяем диапазон для лучшего покрытия
        lower_bound = np.maximum(median_color - self.tolerance, 0)
        upper_bound = np.minimum(median_color + self.tolerance, 255)
        
        logger.debug(f"Доминирующий цвет: {median_color}, диапазон: {lower_bound} - {upper_bound}")
        
        return median_color, np.array([lower_bound, upper_bound])
    
    def create_color_mask_hsv(self, image: np.ndarray, color_range: np.ndarray) -> np.ndarray:
        """
        Создание маски в HSV пространстве (более устойчиво к освещению)
        """
        # Конвертируем в HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Получаем HSV значение доминирующего цвета
        avg_color_bgr = np.mean([color_range[0], color_range[1]], axis=0).astype(np.uint8)
        avg_color_hsv = cv2.cvtColor(np.uint8([[avg_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        
        # Создаем диапазон в HSV
        h_low = max(0, avg_color_hsv[0] - self.hue_tolerance)
        h_high = min(179, avg_color_hsv[0] + self.hue_tolerance)
        s_low = max(0, avg_color_hsv[1] - 80)
        s_high = min(255, avg_color_hsv[1] + 80)
        v_low = max(0, avg_color_hsv[2] - 80)
        v_high = min(255, avg_color_hsv[2] + 80)
        
        lower_hsv = np.array([h_low, s_low, v_low])
        upper_hsv = np.array([h_high, s_high, v_high])
        
        mask_hsv = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        
        # Также создаем маску в BGR для комбинации
        mask_bgr = cv2.inRange(image, color_range[0], color_range[1])
        
        # Комбинируем маски
        combined_mask = cv2.bitwise_or(mask_hsv, mask_bgr)
        
        # Улучшаем маску морфологическими операциями
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        return combined_mask
    
    def analyze_color_compatibility(self, scene: np.ndarray, query_color_range: np.ndarray) -> np.ndarray:
        """
        Анализ совместимости цвета по всей сцене
        """
        # Создаем маску для всей сцены
        full_mask = self.create_color_mask_hsv(scene, query_color_range)
        
        # Визуализируем маску на сцене
        result = scene.copy()
        result[full_mask == 0] = result[full_mask == 0] // 3  # затемняем несовпадающие области
        
        # Показываем процент покрытия
        coverage = np.sum(full_mask > 0) / (scene.shape[0] * scene.shape[1]) * 100
        logger.info(f"Цветовое покрытие сцены: {coverage:.2f}%")
        
        # Добавляем информацию
        text = f"Color Coverage: {coverage:.1f}%"
        cv2.putText(result, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return result, full_mask, coverage
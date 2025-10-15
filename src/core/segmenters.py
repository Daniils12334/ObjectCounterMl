from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

class BaseSegmenter(ABC):
    """Abstract base class for image segmentation"""
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        pass

class SLICSegmenter(BaseSegmenter):
    def __init__(self, region_size: int = 100, ruler: float = 10.0):
        self.region_size = region_size
        self.ruler = ruler
    
    def segment(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        segments = cv2.ximgproc.createSuperpixelSLIC(
            image, algorithm=cv2.ximgproc.SLIC, region_size=self.region_size, ruler=self.ruler
        )
        segments.iterate(10)
        labels = segments.getLabels()
        
        bboxes = []
        for label in np.unique(labels):
            mask = (labels == label).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(mask)
            bboxes.append((x, y, w, h))
        
        return bboxes

class ContourSegmenter(BaseSegmenter):
    def __init__(self, min_area: int = 1000):
        self.min_area = min_area
    
    def segment(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
        
        return bboxes
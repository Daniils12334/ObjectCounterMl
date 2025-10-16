import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class Segmenter:
    def __init__(self, model_name: str = "mask_rcnn", confidence_threshold: float = 0.7, device: str = "cuda"):
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {self.device}")
        
        if model_name == "mask_rcnn":
            self.model = self._load_mask_rcnn()
        else:
            raise ValueError(f"Модель {model_name} не поддерживается")
    
    def _load_mask_rcnn(self):
        """Загрузка предобученной Mask R-CNN модели"""
        # Используем новый API torchvision
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        model.to(self.device)
        model.eval()
        self.transforms = weights.transforms()
        logger.info("Mask R-CNN модель загружена успешно")
        return model
    
    def segment(self, image: np.ndarray) -> List[Dict]:
        """
        Сегментация объектов на изображении
        
        Args:
            image: входное изображение (BGR)
            
        Returns:
            Список словарей с информацией об объектах
        """
        try:
            # Конвертируем BGR в RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Применяем трансформации
            image_tensor = self.transforms(torch.from_numpy(image_rgb).permute(2, 0, 1))
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            return self._process_predictions(predictions[0], image.shape)
            
        except Exception as e:
            logger.error(f"Ошибка при сегментации: {e}")
            return []
    
    def _process_predictions(self, prediction: Dict, image_shape: tuple) -> List[Dict]:
        """Обработка предсказаний модели"""
        objects = []
        
        scores = prediction['scores'].cpu().numpy()
        masks = prediction['masks'].cpu().numpy()
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        height, width = image_shape[:2]
        
        for i, score in enumerate(scores):
            if score < self.confidence_threshold:
                continue
                
            # Ббокс в формате [x1, y1, x2, y2]
            x1, y1, x2, y2 = boxes[i]
            # Конвертируем в [x, y, width, height]
            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Маска
            mask = (masks[i, 0] > 0.5).astype(np.uint8)
            
            objects.append({
                'bbox': bbox,
                'mask': mask,
                'score': score,
                'class_id': labels[i],
                'class_name': self._get_class_name(labels[i])
            })
        
        logger.info(f"Сегментировано {len(objects)} объектов с confidence > {self.confidence_threshold}")
        return objects
    
    def _get_class_name(self, class_id: int) -> str:
        """Получение имени класса по ID"""
        # Базовые классы COCO
        coco_classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
            'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
            'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        if class_id < len(coco_classes):
            return coco_classes[class_id]
        return f"class_{class_id}"
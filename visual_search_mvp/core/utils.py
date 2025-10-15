import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
import matplotlib.pyplot as plt

class ColorAnalyzer:
    @staticmethod
    def get_dominant_color(image_crop: Image.Image, k: int = 1) -> tuple:
        """Определяет доминирующий цвет в изображении"""
        # Конвертируем PIL в numpy для OpenCV
        img_np = np.array(image_crop)
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Преобразуем в HSV для лучшего анализа цвета
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        
        # Анализируем гистограмму для определения основного цвета
        pixels = img_hsv.reshape(-1, 3)
        
        # Используем k-means для нахождения доминирующего цвета
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)
        
        # Возвращаем основной цвет в HSV
        dominant_color = kmeans.cluster_centers_[0]
        return tuple(dominant_color)
    
    @staticmethod
    def is_red_color(hsv_color: tuple, tolerance: int = 20) -> bool:
        """Проверяет, является ли цвет красным (учитывая что красный в HSV может быть в двух диапазонах)"""
        h, s, v = hsv_color
        
        # Красный цвет в HSV: 0-10 и 170-180
        red_low1 = (0 - tolerance, 50, 50)
        red_high1 = (10 + tolerance, 255, 255)
        
        red_low2 = (170 - tolerance, 50, 50) 
        red_high2 = (180 + tolerance, 255, 255)
        
        is_red1 = (red_low1[0] <= h <= red_high1[0] and 
                  red_low1[1] <= s <= red_high1[1] and 
                  red_low1[2] <= v <= red_high1[2])
                  
        is_red2 = (red_low2[0] <= h <= red_high2[0] and 
                  red_low2[1] <= s <= red_high2[1] and 
                  red_low2[2] <= v <= red_high2[2])
        
        return is_red1 or is_red2
    
    @staticmethod
    def filter_by_color(image_crop: Image.Image, target_color: str = "red", tolerance: int = 20) -> bool:
        """Фильтрует изображение по цвету"""
        dominant_color = ColorAnalyzer.get_dominant_color(image_crop)
        
        if target_color == "red":
            return ColorAnalyzer.is_red_color(dominant_color, tolerance)
        elif target_color == "blue":
            # Синий в HSV: ~100-140
            h, s, v = dominant_color
            return (100 - tolerance <= h <= 140 + tolerance and s > 50 and v > 50)
        elif target_color == "green":
            # Зелёный в HSV: ~35-85  
            h, s, v = dominant_color
            return (35 - tolerance <= h <= 85 + tolerance and s > 50 and v > 50)
        
        return True  # Если цвет не указан, пропускаем все

class ImageUtils:
    @staticmethod
    def segment_image(image_path: str, num_segments: int = 50) -> list:
        """Умная сегментация с фильтрацией шума"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_float = img_as_float(image_rgb)
            
            # SLIC сегментация
            segments = slic(image_float, n_segments=num_segments, compactness=10, sigma=1)
            
            bboxes = []
            for segment_id in np.unique(segments):
                mask = (segments == segment_id).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Фильтруем по размеру и форме
                    if (w > 30 and h > 30 and 
                        w < image.shape[1] * 0.8 and 
                        h < image.shape[0] * 0.8):
                        bboxes.append((x, y, w, h))
            
            print(f"✅ Found {len(bboxes)} valid segments")
            return bboxes
            
        except Exception as e:
            print(f"❌ Segmentation error: {e}")
            return []
    
    @staticmethod
    def visualize_search_results(query_path: str, results: list, max_display: int = 5):
        """Красивая визуализация результатов"""
        if not results:
            print("❌ No results to visualize")
            return
            
        fig, axes = plt.subplots(2, max_display, figsize=(15, 8))
        
        # Query image
        try:
            query_img = Image.open(query_path)
            axes[0, 0].imshow(query_img)
            axes[0, 0].set_title("QUERY", fontweight='bold', color='blue')
            axes[0, 0].axis('off')
        except:
            axes[0, 0].text(0.5, 0.5, "Query\nNot Found", ha='center', va='center', fontweight='bold')
            axes[0, 0].axis('off')
        
        # Clear other query cells
        for i in range(1, max_display):
            axes[0, i].axis('off')
        
        # Results
        for i, (metadata, similarity) in enumerate(results[:max_display]):
            try:
                img = cv2.imread(metadata['image_path'])
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x, y, w, h = metadata['bbox']
                
                # Draw bounding box
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_rgb, f"{similarity:.2f}", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                axes[1, i].imshow(img_rgb)
                axes[1, i].set_title(f"Match #{i+1}", fontweight='bold')
                axes[1, i].axis('off')
                
            except Exception as e:
                print(f"❌ Visualization error: {e}")
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
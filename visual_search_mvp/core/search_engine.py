import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from typing import List, Tuple, Dict, Any

from config import Config
from models.clip_model import CLIPModel
from core.utils import ImageUtils, ColorAnalyzer

class VisualSearchEngine:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Инициализируем модель
        self.model = CLIPModel(
            model_name=self.config.MODEL_NAME,
            device=self.config.DEVICE
        )
        
        # Хранилище данных
        self.embeddings = []
        self.metadata = []
        
        print("🚀 Visual Search Engine initialized!")
    
    def index_image(self, image_path: str, image_id: str = None) -> List[tuple]:
        """Индексируем изображение - ОСНОВНАЯ ФУНКЦИЯ"""
        print(f"📸 Indexing: {image_path}")
        
        # Сегментируем
        bboxes = ImageUtils.segment_image(
            image_path, 
            self.config.SEGMENTATION['num_segments']
        )
        
        if not bboxes:
            print("❌ No segments found")
            return []
        
        # Подготавливаем crops
        pil_image = Image.open(image_path).convert('RGB')
        crops = []
        valid_bboxes = []
        
        for i, (x, y, w, h) in enumerate(bboxes):
            try:
                # Вырезаем с отступом
                padding = 5
                x1, y1 = max(0, x-padding), max(0, y-padding)
                x2, y2 = min(pil_image.width, x+w+padding), min(pil_image.height, y+h+padding)
                
                crop = pil_image.crop((x1, y1, x2, y2))
                crops.append(crop)
                valid_bboxes.append((x, y, w, h))
                
            except Exception as e:
                print(f"⚠️  Skipping segment {i}: {e}")
                continue
        
        # БАТЧНОЕ извлечение эмбеддингов - БЫСТРЕЕ!
        embeddings_batch = self.model.batch_get_embeddings(crops)
        
        # Сохраняем результаты
        added_count = 0
        for i, (embedding, bbox) in enumerate(zip(embeddings_batch, valid_bboxes)):
            if embedding is not None:
                self.embeddings.append(embedding)
                self.metadata.append({
                    'image_id': image_id or image_path,
                    'segment_id': len(self.metadata),
                    'bbox': bbox,
                    'image_path': image_path
                })
                added_count += 1
        
        print(f"✅ Successfully indexed {added_count}/{len(bboxes)} segments")
        return valid_bboxes[:added_count]
    
    def search(self, query_image_path: str, top_k: int = None, threshold: float = None) -> List[tuple]:
        """Поиск похожих объектов"""
        if not self.embeddings:
            print("❌ No images indexed yet!")
            return []
        
        top_k = top_k or self.config.SEARCH['top_k']
        threshold = threshold or self.config.SEARCH['similarity_threshold']
        
        print(f"🔍 Searching: {query_image_path}")
        
        try:
            # Эмбеддинг для query
            query_image = Image.open(query_image_path).convert('RGB')
            query_embedding = self.model.get_embedding(query_image)
            
            if query_embedding is None:
                return []
            
            # Вычисляем схожести
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # Сортируем и фильтруем
            results = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    results.append((self.metadata[i], similarity))
            
            # Сортируем по убыванию схожести
            results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🎯 Found {len(results)} matches (showing top {top_k})")
            return results[:top_k]
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика системы"""
        return {
            'total_segments': len(self.embeddings),
            'total_images': len(set(m['image_id'] for m in self.metadata)),
            'embedding_dim': self.embeddings[0].shape[0] if self.embeddings else 0
        }
    
class ImprovedVisualSearchEngine(VisualSearchEngine):
    def __init__(self, config=None):
        super().__init__(config)
        self.color_analyzer = ColorAnalyzer()
    
    def search_with_color_filter(self, query_image_path: str, target_color: str = None, 
                               color_tolerance: int = 20, top_k: int = 5, threshold: float = 0.6) -> List[tuple]:
        """Поиск с фильтрацией по цвету"""
        
        # Сначала делаем обычный поиск
        all_results = self.search(query_image_path, top_k=top_k*3, threshold=threshold*0.7)
        
        if not all_results or not target_color:
            return all_results[:top_k]
        
        # Фильтруем по цвету
        filtered_results = []
        
        for metadata, similarity in all_results:
            try:
                # Загружаем оригинальное изображение и вырезаем сегмент
                img = Image.open(metadata['image_path']).convert('RGB')
                x, y, w, h = metadata['bbox']
                
                # Добавляем отступы
                padding = 5
                x1, y1 = max(0, x-padding), max(0, y-padding)
                x2, y2 = min(img.width, x+w+padding), min(img.height, y+h+padding)
                
                segment_crop = img.crop((x1, y1, x2, y2))
                
                # Проверяем цвет
                if self.color_analyzer.filter_by_color(segment_crop, target_color, color_tolerance):
                    filtered_results.append((metadata, similarity))
                    
                    if len(filtered_results) >= top_k:
                        break
                        
            except Exception as e:
                print(f"⚠️ Color filtering error: {e}")
                continue
        
        print(f"🎨 After color filtering: {len(filtered_results)}/{len(all_results)} matches")
        return filtered_results
    
    def search_combined(self, query_image_path: str, target_color: str = None,
                    shape_weight: float = 0.7, color_weight: float = 0.3,
                    top_k: int = 5) -> List[tuple]:
        """Комбинированный поиск по форме и цвету"""
        
        # 1. Получаем эмбеддинг query изображения (форма)
        query_image = Image.open(query_image_path).convert('RGB')
        query_embedding = self.model.get_embedding(query_image)
        
        if query_embedding is None:
            return []
        
        # 2. Вычисляем схожести по форме
        shape_similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # 3. Вычисляем схожести по цвету (если указан целевой цвет)
        color_similarities = np.ones(len(self.embeddings))
        
        if target_color:
            for i, metadata in enumerate(self.metadata):
                try:
                    img = Image.open(metadata['image_path']).convert('RGB')
                    x, y, w, h = metadata['bbox']
                    
                    padding = 5
                    x1, y1 = max(0, x-padding), max(0, y-padding)
                    x2, y2 = min(img.width, x+w+padding), min(img.height, y+h+padding)
                    
                    segment_crop = img.crop((x1, y1, x2, y2))
                    
                    # Чем больше совпадение по цвету, тем выше score
                    color_match = self.color_analyzer.filter_by_color(segment_crop, target_color, 30)
                    color_similarities[i] = 1.0 if color_match else 0.3
                    
                except:
                    color_similarities[i] = 0.5
        
        # 4. Комбинируем scores
        combined_scores = (shape_similarities * shape_weight + 
                        color_similarities * color_weight)
        
        # 5. Сортируем и возвращаем результаты
        results = []
        for i in np.argsort(combined_scores)[::-1]:
            if combined_scores[i] > 0.5:  # Общий порог
                results.append((self.metadata[i], combined_scores[i]))
            
            if len(results) >= top_k:
                break
        
        return results
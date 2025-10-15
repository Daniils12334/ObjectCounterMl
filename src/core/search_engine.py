import numpy as np
from typing import List, Tuple, Optional
import faiss
from sklearn.metrics.pairwise import cosine_similarity

from .embedders import BaseEmbedder
from .segmenters import BaseSegmenter

class VisualSearchEngine:
    def __init__(
        self,
        embedder: BaseEmbedder,
        segmenter: BaseSegmenter,
        use_faiss: bool = True
    ):
        self.embedder = embedder
        self.segmenter = segmenter
        self.use_faiss = use_faiss
        self.index = None
        self.segment_metadata = []
        
        if use_faiss:
            dim = self.embedder.get_embedding_dim()
            self.index = faiss.IndexFlatIP(dim)
    
    def index_image(
        self, 
        image_path: str, 
        image_id: Optional[str] = None
    ) -> List[Tuple[int, int, int, int]]:
        """Segment image and index all segments"""
        import cv2
        from PIL import Image
        
        # Load and segment image
        image_cv = cv2.imread(image_path)
        image_pil = Image.open(image_path)
        segments = self.segmenter.segment(image_cv)
        
        embeddings = []
        for i, (x, y, w, h) in enumerate(segments):
            # Extract segment and get embedding
            segment_img = image_pil.crop((x, y, x + w, y + h))
            processed = self.embedder.preprocess(segment_img)
            embedding = self.embedder.encode(processed)
            
            embeddings.append(embedding)
            
            # Store metadata
            self.segment_metadata.append({
                'image_id': image_id or image_path,
                'segment_id': i,
                'bbox': (x, y, w, h),
                'image_path': image_path
            })
        
        # Add to index
        if self.use_faiss and self.index is not None:
            embeddings_np = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_np)
            self.index.add(embeddings_np)
        
        return segments
    
    def search(
        self,
        query_image_path: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[dict, float]]:
        """Search for similar segments"""
        from PIL import Image
        
        # Get query embedding
        query_image = Image.open(query_image_path)
        processed = self.embedder.preprocess(query_image)
        query_embedding = self.embedder.encode(processed)
        
        if self.use_faiss and self.index is not None:
            # FAISS search
            query_embedding_np = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding_np)
            
            similarities, indices = self.index.search(query_embedding_np, top_k)
            
            results = []
            for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
                if sim >= similarity_threshold:
                    results.append((self.segment_metadata[idx], float(sim)))
            
            return results
        else:
            # Fallback to sklearn (for small datasets)
            all_embeddings = np.array([self.get_embedding_by_idx(i) 
                                     for i in range(len(self.segment_metadata))])
            
            similarities = cosine_similarity([query_embedding], all_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            return [
                (self.segment_metadata[idx], float(similarities[idx]))
                for idx in top_indices if similarities[idx] >= similarity_threshold
            ]
    
    def get_embedding_by_idx(self, idx: int) -> np.ndarray:
        """Get embedding by index (for non-FAISS mode)"""
        # This would need to cache embeddings in non-FAISS mode
        # Implementation depends on storage strategy
        pass
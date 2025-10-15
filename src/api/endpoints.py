from fastapi import FastAPI, UploadFile, File, HTTPException
from .schemas import SearchResponse, IndexResponse
from ..core.search_engine import VisualSearchEngine
from ..models.model_registry import ModelRegistry
import yaml

app = FastAPI(title="Visual Search API")

# Load configuration
with open("config/model_configs.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize search engine
embedder = ModelRegistry.get_embedder(
    config["search"]["default_model"],
    config["models"][config["search"]["default_model"]]
)
segmenter = ModelRegistry.get_segmenter("slic")
search_engine = VisualSearchEngine(embedder, segmenter)

@app.post("/index", response_model=IndexResponse)
async def index_image(file: UploadFile = File(...), image_id: str = None):
    """Index an image for search"""
    try:
        # Save uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        segments = search_engine.index_image(file_path, image_id)
        return IndexResponse(
            image_id=image_id or file.filename,
            segments_found=len(segments),
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_similar(
    file: UploadFile = File(...),
    top_k: int = 10,
    threshold: float = 0.7
):
    """Search for similar segments"""
    try:
        # Save query image temporarily
        file_path = f"/tmp/query_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        results = search_engine.search(file_path, top_k, threshold)
        return SearchResponse(
            query_image=file.filename,
            results_found=len(results),
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
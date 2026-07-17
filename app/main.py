from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from app.models import RecommendResponse
from app.recommender import PinNotFoundError, PinRecommender
from app.vectorstore import load_vectorstore

load_dotenv()

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"

recommender: PinRecommender | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    store = load_vectorstore()
    recommender = PinRecommender(store)
    yield


app = FastAPI(title="Pinterest-style Content Recommender", lifespan=lifespan)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/recommend/pin/{pin_id}", response_model=RecommendResponse)
def recommend_by_pin(pin_id: str, k: int = Query(default=5, ge=1, le=50)):
    try:
        results = recommender.similarity_search_by_pin_id(pin_id, k=k)
    except PinNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return RecommendResponse(results=results)


@app.get("/recommend/query", response_model=RecommendResponse)
def recommend_by_query(q: str, k: int = Query(default=5, ge=1, le=50)):
    results = recommender.similarity_search_by_query(q, k=k)
    return RecommendResponse(results=results)


@app.get("/health")
def health():
    return {"status": "ok"}

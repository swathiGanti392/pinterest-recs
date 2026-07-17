from typing import List

from langchain_chroma import Chroma

from app.models import RecommendedPin


class PinNotFoundError(Exception):
    pass


class PinRecommender:
    def __init__(self, store: Chroma):
        self.store = store

    def _doc_to_recommendation(self, doc, score: float) -> RecommendedPin:
        return RecommendedPin(
            pin_id=doc.metadata["id"],
            title=doc.metadata["title"],
            board=doc.metadata["board"],
            image_url=doc.metadata["image_url"],
            score=score,
        )

    def similarity_search_by_query(self, query: str, k: int = 5) -> List[RecommendedPin]:
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        return [self._doc_to_recommendation(doc, score) for doc, score in results]

    def similarity_search_by_pin_id(self, pin_id: str, k: int = 5) -> List[RecommendedPin]:
        stored = self.store.get(ids=[pin_id], include=["documents", "metadatas"])
        if not stored["ids"]:
            raise PinNotFoundError(f"Pin '{pin_id}' not found in vector store.")

        source_text = stored["documents"][0]
        # Fetch extra results since the source pin itself will match with score 1.0.
        results = self.store.similarity_search_with_relevance_scores(source_text, k=k + 1)

        recommendations = [
            self._doc_to_recommendation(doc, score)
            for doc, score in results
            if doc.metadata["id"] != pin_id
        ]
        return recommendations[:k]

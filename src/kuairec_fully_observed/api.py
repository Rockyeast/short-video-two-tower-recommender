"""FastAPI application for the frozen recommendation pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .pipeline import RecommendationEngine, RecommendationResult
from .serving import load_recommendation_engine


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int
    history: list[int] = Field(default_factory=list, max_length=50)
    history_weights: list[float] | None = Field(
        default=None, max_length=50
    )
    top_k: int | None = Field(default=None, ge=1)
    use_reranker: bool | None = None

    @model_validator(mode="after")
    def validate_history_weights(self) -> "RecommendationRequest":
        if (
            self.history_weights is not None
            and len(self.history_weights) != len(self.history)
        ):
            raise ValueError("history_weights must align with history")
        if self.history_weights is not None and any(
            value < 0 for value in self.history_weights
        ):
            raise ValueError("history_weights must be non-negative")
        return self


class BatchRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests: list[RecommendationRequest] = Field(
        min_length=1, max_length=256
    )


class RecommendationResponse(BaseModel):
    user_id: int
    item_ids: list[int]
    strategy: str
    candidate_count: int
    seen_count: int


class BatchRecommendationResponse(BaseModel):
    count: int
    results: list[RecommendationResponse]


class HealthResponse(BaseModel):
    status: str
    schema_version: int
    fit_context: str
    bundle_sha256: str
    catalog_count: int
    reranker_loaded: bool
    reranker_enabled_by_default: bool


@dataclass(frozen=True)
class ServiceState:
    engine: RecommendationEngine
    metadata: dict[str, Any]


def _response(result: RecommendationResult) -> RecommendationResponse:
    return RecommendationResponse(
        user_id=result.user_id,
        item_ids=list(result.item_ids),
        strategy=result.strategy,
        candidate_count=result.candidate_count,
        seen_count=result.seen_count,
    )


def _recommend(
    engine: RecommendationEngine,
    payload: RecommendationRequest,
) -> RecommendationResponse:
    try:
        result = engine.recommend(
            payload.user_id,
            payload.history,
            top_k=payload.top_k,
            history_weights=payload.history_weights,
            use_reranker=payload.use_reranker,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _response(result)


def _state(request: Request) -> ServiceState:
    state = getattr(request.app.state, "recommendation_service", None)
    if not isinstance(state, ServiceState):
        raise HTTPException(status_code=503, detail="service is not ready")
    return state


def create_app(
    *,
    config_path: Path,
    bundle_path: Path,
    metadata_path: Path,
    reranker_model_path: Path | None = None,
    reranker_feature_path: Path | None = None,
    reranker_metadata_path: Path | None = None,
) -> FastAPI:
    """Create an app whose engine is loaded and verified exactly once."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine, metadata = load_recommendation_engine(
            config_path=config_path,
            bundle_path=bundle_path,
            metadata_path=metadata_path,
            reranker_model_path=reranker_model_path,
            reranker_feature_path=reranker_feature_path,
            reranker_metadata_path=reranker_metadata_path,
        )
        app.state.recommendation_service = ServiceState(engine, metadata)
        yield

    app = FastAPI(
        title="Short Video Recommendation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        service = _state(request)
        metadata = service.metadata
        return HealthResponse(
            status="ok",
            schema_version=int(metadata["schema_version"]),
            fit_context=str(metadata["fit_context"]),
            bundle_sha256=str(metadata["bundle_sha256"]),
            catalog_count=int(metadata["catalog_count"]),
            reranker_loaded=bool(metadata["reranker_loaded"]),
            reranker_enabled_by_default=bool(
                metadata["reranker_enabled_by_default"]
            ),
        )

    @app.post("/v1/recommend", response_model=RecommendationResponse)
    async def recommend(
        payload: RecommendationRequest,
        request: Request,
    ) -> RecommendationResponse:
        return _recommend(_state(request).engine, payload)

    @app.post(
        "/v1/recommend/batch",
        response_model=BatchRecommendationResponse,
    )
    async def recommend_batch(
        payload: BatchRecommendationRequest,
        request: Request,
    ) -> BatchRecommendationResponse:
        engine = _state(request).engine
        results = [_recommend(engine, item) for item in payload.requests]
        return BatchRecommendationResponse(
            count=len(results),
            results=results,
        )

    return app

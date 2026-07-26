from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import numpy as np

from kuairec_fully_observed import api
from kuairec_fully_observed.api import create_app
from kuairec_fully_observed.pipeline import (
    BPRRetriever,
    DynamicTwoTowerRetriever,
    PipelineConfig,
    PopularityRetriever,
    RecommendationEngine,
)


def _engine() -> RecommendationEngine:
    items = np.arange(1, 7, dtype=np.int64)
    item_vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    return RecommendationEngine(
        catalog=items,
        two_tower=DynamicTwoTowerRetriever(
            item_ids=items,
            item_vectors=item_vectors,
            user_ids=np.asarray([10], dtype=np.int64),
            user_id_embeddings=np.zeros((1, 2), dtype=np.float32),
            mlp_input_weight=np.asarray(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            mlp_input_bias=np.zeros(2, dtype=np.float32),
            mlp_output_weight=np.eye(2, dtype=np.float32),
            mlp_output_bias=np.zeros(2, dtype=np.float32),
        ),
        bpr=BPRRetriever(
            user_ids=np.asarray([10], dtype=np.int64),
            item_ids=items,
            user_factors=np.asarray([[0.0, 1.0]], dtype=np.float32),
            item_factors=item_vectors,
        ),
        popularity=PopularityRetriever(
            {1: 1.0, 2: 2.0, 3: 6.0, 4: 5.0, 5: 4.0, 6: 3.0}
        ),
        config=PipelineConfig(route_top_k=6, output_k=3),
    )


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 2,
        "fit_context": "canonical_big_train_plus_validation",
        "bundle_sha256": "a" * 64,
        "catalog_count": 6,
        "reranker_loaded": False,
        "reranker_enabled_by_default": False,
    }


def _app(monkeypatch):
    load_calls: list[int] = []

    def load(**kwargs):
        load_calls.append(1)
        return _engine(), _metadata()

    monkeypatch.setattr(api, "load_recommendation_engine", load)
    application = create_app(
        config_path=Path("unused-config"),
        bundle_path=Path("unused-bundle"),
        metadata_path=Path("unused-metadata"),
    )
    return application, load_calls


async def _with_client(
    application,
    callback: Callable[[httpx.AsyncClient], Awaitable[None]],
) -> None:
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            await callback(client)


def test_api_loads_engine_once_and_reports_bundle_health(monkeypatch):
    application, load_calls = _app(monkeypatch)

    async def exercise(client):
        first = await client.get("/healthz")
        second = await client.get("/healthz")

        assert first.status_code == second.status_code == 200
        assert first.json() == {
            "status": "ok",
            "schema_version": 2,
            "fit_context": "canonical_big_train_plus_validation",
            "bundle_sha256": "a" * 64,
            "catalog_count": 6,
            "reranker_loaded": False,
            "reranker_enabled_by_default": False,
        }

    asyncio.run(_with_client(application, exercise))
    assert load_calls == [1]


def test_single_endpoint_uses_history_and_filters_seen_items(monkeypatch):
    application, _ = _app(monkeypatch)

    async def exercise(client):
        first = await client.post(
            "/v1/recommend",
            json={"user_id": 10, "history": [1], "top_k": 3},
        )
        second = await client.post(
            "/v1/recommend",
            json={"user_id": 10, "history": [2], "top_k": 3},
        )

        assert first.status_code == second.status_code == 200
        assert first.json()["strategy"] == "two_tower_bpr_rrf"
        assert first.json()["item_ids"] != second.json()["item_ids"]
        assert 1 not in first.json()["item_ids"]
        assert 2 not in second.json()["item_ids"]

    asyncio.run(_with_client(application, exercise))


def test_batch_endpoint_reuses_engine_and_preserves_request_order(monkeypatch):
    application, load_calls = _app(monkeypatch)
    payload = {
        "requests": [
            {"user_id": 10, "history": [1], "top_k": 2},
            {"user_id": 999, "history": [3], "top_k": 2},
        ]
    }

    async def exercise(client):
        response = await client.post("/v1/recommend/batch", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 2
        assert [result["user_id"] for result in body["results"]] == [10, 999]
        assert body["results"][0]["strategy"] == "two_tower_bpr_rrf"
        assert body["results"][1]["strategy"] == "cold_user_popularity"

    asyncio.run(_with_client(application, exercise))
    assert load_calls == [1]


def test_api_rejects_invalid_history_weights_and_oversized_top_k(monkeypatch):
    application, _ = _app(monkeypatch)

    async def exercise(client):
        mismatch = await client.post(
            "/v1/recommend",
            json={
                "user_id": 10,
                "history": [1, 2],
                "history_weights": [1.0],
            },
        )
        negative = await client.post(
            "/v1/recommend",
            json={
                "user_id": 10,
                "history": [1],
                "history_weights": [-1.0],
            },
        )
        oversized = await client.post(
            "/v1/recommend",
            json={"user_id": 10, "history": [], "top_k": 4},
        )
        unknown = await client.post(
            "/v1/recommend",
            json={"user_id": 10, "history": [], "typo_top_k": 3},
        )

        assert mismatch.status_code == 422
        assert negative.status_code == 422
        assert oversized.status_code == 422
        assert unknown.status_code == 422
        assert "configured output limit" in oversized.json()["detail"]

    asyncio.run(_with_client(application, exercise))

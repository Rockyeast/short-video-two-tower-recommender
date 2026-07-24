#!/usr/bin/env python3
"""Start the local FastAPI recommendation service."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from kuairec_fully_observed.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recommendation_pipeline_v1.yaml"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reranker-model", type=Path)
    parser.add_argument("--reranker-features", type=Path)
    parser.add_argument("--reranker-metadata", type=Path)
    arguments = parser.parse_args()
    app = create_app(
        config_path=arguments.config,
        bundle_path=arguments.bundle,
        metadata_path=arguments.metadata,
        reranker_model_path=arguments.reranker_model,
        reranker_feature_path=arguments.reranker_features,
        reranker_metadata_path=arguments.reranker_metadata,
    )
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        workers=1,
    )


if __name__ == "__main__":
    main()

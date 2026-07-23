"""Simple fixed-catalog KuaiRec retrieval protocol for the primary model path."""

from .data import (
    RetrievalQueries,
    build_big_validation_queries,
    build_fixed_validation_catalog,
    build_small_observed_queries,
    data_cold_items,
    is_quick_skip,
    is_strong_positive,
    resolve_kuairec_data_dir,
)
from .evaluation import evaluate_retrieval
from .features import (
    MODEL_ITEM_FEATURE_COLUMNS,
    StaticItemFeatures,
    load_static_item_features,
    validate_model_item_feature_columns,
)
from .models import (
    BPRModel,
    BPRTrainingResult,
    NumpyTwoTowerReference,
    PopularityBaseline,
    in_batch_softmax_loss,
    stable_random_rank,
    train_bpr_sgd,
)
from .retrieval import ExactDotProductRetriever
from .sealed_small import (
    FROZEN_OUTPUT_K,
    FROZEN_ROUTE_TOP_K,
    FROZEN_SMALL_ALPHA,
    FROZEN_SMALL_METHODS,
    evaluate_frozen_small_routes,
    require_sealed_execution,
)
from .training import (
    BPRTrainingDataset,
    TwoTowerTrainingDataset,
    TwoTowerTrainingExamples,
    build_bpr_training_dataset,
    build_in_batch_logit_mask,
    build_two_tower_training_dataset,
    build_two_tower_training_examples,
)

__all__ = [
    "BPRModel",
    "BPRTrainingDataset",
    "BPRTrainingResult",
    "ExactDotProductRetriever",
    "FROZEN_OUTPUT_K",
    "FROZEN_ROUTE_TOP_K",
    "FROZEN_SMALL_ALPHA",
    "FROZEN_SMALL_METHODS",
    "NumpyTwoTowerReference",
    "MODEL_ITEM_FEATURE_COLUMNS",
    "PopularityBaseline",
    "RetrievalQueries",
    "StaticItemFeatures",
    "TwoTowerTrainingDataset",
    "TwoTowerTrainingExamples",
    "build_big_validation_queries",
    "build_bpr_training_dataset",
    "build_fixed_validation_catalog",
    "build_in_batch_logit_mask",
    "build_small_observed_queries",
    "build_two_tower_training_dataset",
    "build_two_tower_training_examples",
    "data_cold_items",
    "evaluate_retrieval",
    "evaluate_frozen_small_routes",
    "require_sealed_execution",
    "in_batch_softmax_loss",
    "is_quick_skip",
    "is_strong_positive",
    "load_static_item_features",
    "resolve_kuairec_data_dir",
    "stable_random_rank",
    "train_bpr_sgd",
    "validate_model_item_feature_columns",
]

"""The single trainable PyTorch Two-Tower architecture for Phase B2A."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TorchItemFeatures:
    """Dense, position-aligned item features used by all three item paths."""

    item_indices: torch.Tensor
    category_indices: torch.Tensor
    caption_embeddings: torch.Tensor
    caption_present: torch.Tensor
    numeric_features: torch.Tensor
    upload_type_indices: torch.Tensor

    def select(self, positions: torch.Tensor) -> "TorchItemFeatures":
        return TorchItemFeatures(
            item_indices=self.item_indices[positions],
            category_indices=self.category_indices[positions],
            caption_embeddings=self.caption_embeddings[positions],
            caption_present=self.caption_present[positions],
            numeric_features=self.numeric_features[positions],
            upload_type_indices=self.upload_type_indices[positions],
        )


class TwoTowerV1(nn.Module):
    """Fixed V1 item/content tower plus weighted-history user tower."""

    def __init__(
        self,
        *,
        num_items: int,
        num_users: int,
        num_category_tokens: int,
        num_upload_types: int,
    ) -> None:
        super().__init__()
        self.item_id_embedding = nn.Embedding(num_items + 1, 64, padding_idx=0)
        self.category_embedding = nn.Embedding(
            num_category_tokens + 1, 32, padding_idx=0
        )
        self.caption_projection = nn.Sequential(nn.Linear(384, 64), nn.GELU())
        self.static_projection = nn.Sequential(nn.Linear(4, 16), nn.GELU())
        self.upload_type_embedding = nn.Embedding(
            num_upload_types + 1, 8, padding_idx=0
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(64 + 32 + 64 + 16 + 8, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        self.user_id_embedding = nn.Embedding(num_users + 1, 64, padding_idx=0)
        self.user_mlp = nn.Sequential(
            nn.Linear(64 + 128, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )
        self._zero_padding_rows()

    def _zero_padding_rows(self) -> None:
        with torch.no_grad():
            for embedding in (
                self.item_id_embedding,
                self.category_embedding,
                self.upload_type_embedding,
                self.user_id_embedding,
            ):
                embedding.weight[0].zero_()

    def encode_items(
        self,
        features: TorchItemFeatures,
        *,
        use_id_embedding: torch.Tensor,
    ) -> torch.Tensor:
        use_id = use_id_embedding.to(dtype=torch.bool)
        if use_id.shape != features.item_indices.shape:
            raise ValueError("use_id_embedding must be one boolean per item")
        id_vector = self.item_id_embedding(features.item_indices)
        id_vector = id_vector * use_id.unsqueeze(1).to(id_vector.dtype)

        category_mask = features.category_indices.ne(0)
        categories = self.category_embedding(features.category_indices)
        category_sum = (categories * category_mask.unsqueeze(-1)).sum(dim=1)
        category_count = category_mask.sum(dim=1, keepdim=True).clamp_min(1)
        category_vector = category_sum / category_count

        caption_vector = self.caption_projection(features.caption_embeddings)
        caption_vector = caption_vector * features.caption_present.unsqueeze(1).to(
            caption_vector.dtype
        )
        numeric_vector = self.static_projection(features.numeric_features)
        upload_vector = self.upload_type_embedding(features.upload_type_indices)
        output = self.item_mlp(
            torch.cat(
                (
                    id_vector,
                    category_vector,
                    caption_vector,
                    numeric_vector,
                    upload_vector,
                ),
                dim=1,
            )
        )
        return F.normalize(output, p=2, dim=1, eps=1e-12)

    @staticmethod
    def weighted_history_mean(
        history_vectors: torch.Tensor,
        history_weights: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if history_vectors.ndim != 3:
            raise ValueError("history_vectors must have shape [batch, history, dim]")
        if history_weights.shape != history_vectors.shape[:2]:
            raise ValueError("history_weights shape does not match histories")
        if padding_mask.shape != history_vectors.shape[:2]:
            raise ValueError("padding_mask shape does not match histories")
        effective = history_weights.to(history_vectors.dtype) * padding_mask.to(
            history_vectors.dtype
        )
        numerator = (history_vectors * effective.unsqueeze(-1)).sum(dim=1)
        denominator = effective.sum(dim=1, keepdim=True)
        return numerator / denominator.clamp_min(1e-12)

    def encode_users(
        self,
        *,
        user_indices: torch.Tensor,
        history_vectors: torch.Tensor,
        history_weights: torch.Tensor,
        padding_mask: torch.Tensor,
        use_id_embedding: torch.Tensor,
    ) -> torch.Tensor:
        use_id = use_id_embedding.to(dtype=torch.bool)
        if use_id.shape != user_indices.shape:
            raise ValueError("use_id_embedding must be one boolean per user")
        history_mean = self.weighted_history_mean(
            history_vectors, history_weights, padding_mask
        )
        user_vector = self.user_id_embedding(user_indices)
        user_vector = user_vector * use_id.unsqueeze(1).to(user_vector.dtype)
        output = self.user_mlp(torch.cat((user_vector, history_mean), dim=1))
        return F.normalize(output, p=2, dim=1, eps=1e-12)


def masked_in_batch_cross_entropy(
    user_vectors: torch.Tensor,
    target_vectors: torch.Tensor,
    allowed_logits: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Temperature-scaled diagonal CE with explicit false-negative masking."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if user_vectors.shape != target_vectors.shape or user_vectors.ndim != 2:
        raise ValueError("User and target vectors need matching rank-2 shapes")
    count = len(user_vectors)
    if allowed_logits.shape != (count, count) or allowed_logits.dtype != torch.bool:
        raise ValueError("allowed_logits must be one boolean square batch matrix")
    diagonal = torch.arange(count, device=user_vectors.device)
    if not torch.all(allowed_logits[diagonal, diagonal]):
        raise ValueError("Every diagonal positive must remain unmasked")
    valid_negatives = allowed_logits.sum(dim=1) - 1
    if torch.any(valid_negatives < 1):
        raise ValueError("Every in-batch row needs at least one valid negative")
    raw_logits = user_vectors @ target_vectors.T / temperature
    logits = raw_logits.masked_fill(~allowed_logits, -torch.inf)
    loss = F.cross_entropy(logits, diagonal)
    valid_negative_mask = allowed_logits.clone()
    valid_negative_mask[diagonal, diagonal] = False
    return loss, {
        "diagonal_top1_rate": float(
            logits.argmax(dim=1).eq(diagonal).float().mean().detach().cpu()
        ),
        "mean_positive_logit": float(
            raw_logits[diagonal, diagonal].mean().detach().cpu()
        ),
        "mean_valid_negative_logit": float(
            raw_logits[valid_negative_mask].mean().detach().cpu()
        ),
        "off_diagonal_masked_fraction": float(
            (~allowed_logits & ~torch.eye(count, dtype=torch.bool, device=logits.device))
            .sum()
            .detach()
            .cpu()
            / max(1, count * (count - 1))
        ),
        "valid_negative_min": float(valid_negatives.min().detach().cpu()),
        "valid_negative_median": float(valid_negatives.float().median().detach().cpu()),
    }

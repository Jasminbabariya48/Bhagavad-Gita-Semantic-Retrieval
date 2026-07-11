"""
Fine-tune a SentenceTransformer on the triplets/pairs dataset.

Refactor of notebooks/04_train_embedding.ipynb. Same training logic
(MultipleNegativesRankingLoss over triplets, since it only needs
anchor/positive and generalizes better than TripletLoss for retrieval —
TripletLoss remains available via config for parity with the notebook) but
config-driven, loggable, and MLflow-instrumented.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from src.config import get_config
from src.logging_utils import get_logger, log_pipeline_version

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model_name: str
    output_path: str
    num_examples: int
    epochs: int
    batch_size: int
    loss_name: str
    final_train_loss: float | None = None
    extra: dict = field(default_factory=dict)


def _build_input_examples(triplets_df: pd.DataFrame):
    from sentence_transformers import InputExample

    examples = []
    for _, row in triplets_df.iterrows():
        examples.append(
            InputExample(texts=[row["anchor"], row["positive"], row["negative"]])
        )
    return examples


def train_embedding_model(
    model_name: str | None = None,
    triplets_path: str | None = None,
    output_path: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    loss_name: str | None = None,
) -> TrainingResult:
    """Fine-tune `model_name` on the triplets CSV and save to `output_path`.

    Mirrors the notebook's training loop 1:1 in substance; every knob that
    was hardcoded there is now a config value / function argument.
    """
    log_pipeline_version(logger)
    cfg = get_config()

    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model_name = model_name or cfg.embedding.active_model
    triplets_path = triplets_path or str(cfg.path("triplets"))
    epochs = epochs or cfg.training.epochs
    batch_size = batch_size or cfg.training.batch_size
    loss_name = loss_name or cfg.training.loss

    if output_path is None:
        safe_name = model_name.replace("/", "__")
        output_path = str(
            cfg.path("models_dir") / f"{safe_name}_{cfg.training.output_subdir}"
        )
    os.makedirs(output_path, exist_ok=True)

    logger.info("Loading triplets from %s", triplets_path)
    triplets_df = pd.read_csv(triplets_path)
    logger.info("Training examples: %d", len(triplets_df))

    logger.info("Loading base model '%s'", model_name)
    model = SentenceTransformer(model_name)

    train_examples = _build_input_examples(triplets_df)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    if loss_name == "TripletLoss":
        train_loss = losses.TripletLoss(model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * cfg.training.warmup_ratio)

    logger.info(
        "Training: epochs=%d batch_size=%d loss=%s warmup_steps=%d",
        epochs,
        batch_size,
        loss_name,
        warmup_steps,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
    )
    model.save(output_path)
    logger.info("Model saved to %s", output_path)

    return TrainingResult(
        model_name=model_name,
        output_path=output_path,
        num_examples=len(triplets_df),
        epochs=epochs,
        batch_size=batch_size,
        loss_name=loss_name,
    )

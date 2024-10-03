"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data.tfds import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, ImageTransform, PaddedCollatorForActionPrediction


def get_dataset_and_collator_tfds(
    data_root_dir: Path,
    data_mix: str,
    tokenizer: PreTrainedTokenizerBase,
    padding_side: str = "right",
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    window_size: int = 1,
    action_model_max_length: int = 1024,
) -> Tuple[Dataset, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS) and initialize transform/collation functions."""
    batch_transform = RLDSBatchTransform(tokenizer)
    collator = PaddedCollatorForActionPrediction(tokenizer.model_max_length, 
                                                 tokenizer.pad_token_id, 
                                                 padding_side=padding_side,
                                                 action_model_max_length=action_model_max_length
                                                )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=(None, None),
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        window_size=window_size,
    )

    return dataset, collator

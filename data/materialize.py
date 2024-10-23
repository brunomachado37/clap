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
from data.libero.dataset import LiberoBatchTransform
from data.libero.libero_utils import build_dataset


def get_dataset_and_collator(dataset, **kwargs):
    if dataset == "rlds":
        return get_dataset_and_collator_tfds(**kwargs)
    elif dataset == "libero":
        return get_dataset_and_collator_libero(**kwargs)
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")
    

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


def get_dataset_and_collator_libero(data_root_dir,
                                    tokenizer,
                                    benchmark_name="LIBERO_90",
                                    action_dim=7,
                                    lang_dim=512,
                                    n_demos=50,
                                    padding_side="right",
                                    action_model_max_length=1024):

    batch_transform = LiberoBatchTransform(tokenizer)

    shape_meta = {"action_dim": action_dim, "task": {"type": "vector", "dim": lang_dim}}                            
    dataset = build_dataset(data_prefix=data_root_dir,
                            suite_name="libero",
                            benchmark_name=benchmark_name,
                            mode="all",
                            seq_len=1,
                            frame_stack=1,
                            shape_meta=shape_meta,
                            n_demos=n_demos,
                            extra_obs_modality=None,
                            obs_seq_len=1,
                            load_obs=False,
                            task_embedding_format="clip",
                            episodic=True,
                            pre_compute_task_embeddings=False,
                            batch_transform=batch_transform)
    
    collator = PaddedCollatorForActionPrediction(tokenizer.model_max_length, 
                                                 tokenizer.pad_token_id, 
                                                 padding_side=padding_side,
                                                 action_model_max_length=action_model_max_length)
    
    return dataset, collator
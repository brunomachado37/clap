import hydra
import torch
import pathlib
import numpy as np

from tqdm import tqdm
from omegaconf import open_dict
from transformers import AutoTokenizer

from clap import Clap, ContrastiveTraining
from data.materialize import get_dataset_and_collator_tfds


@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def eval(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.language_encoder.model_name)
    dataset, collator = get_dataset_and_collator_tfds(**config.data, tokenizer=tokenizer)

    with open_dict(config):
        config.model.language_encoder.eos_token_id = tokenizer.eos_token_id

    backbone = Clap(**config.model)
    model = ContrastiveTraining.load_from_checkpoint(config.checkpoint_path, model=backbone)
    model.eval()

    trajectory_embeddings, language_embeddings, language_descriptions = [], [], []

    for item in tqdm(dataset):
        language_descriptions.append(tokenizer.decode(item["input_ids"], skip_special_tokens=True))
        batched_input = {k: v.to(model.device) for k,v in collator([item]).items()}
        with torch.no_grad():
            trajectory_embedding, language_embedding = model.model(**batched_input)

        trajectory_embeddings.append(trajectory_embedding.detach().cpu())
        language_embeddings.append(language_embedding.detach().cpu())

    trajectory_embeddings = torch.cat(trajectory_embeddings, dim=0).numpy()
    language_embeddings = torch.cat(language_embeddings, dim=0).numpy()

    set = "train" if config.data.train else "test"
    pathlib.Path(f"{config.save_path}").mkdir(parents=True, exist_ok=True)

    np.save(f"{config.save_path}/{set}_language_descriptions.npy", np.array(language_descriptions))
    np.save(f"{config.save_path}/{set}_trajectory_embeddings.npy", trajectory_embeddings)
    np.save(f"{config.save_path}/{set}_language_embeddings.npy", language_embeddings)


if __name__ == "__main__":
    eval()
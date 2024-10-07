import hydra
import torch

from tqdm import tqdm
from omegaconf import open_dict
from transformers import AutoTokenizer
from sklearn.neighbors import KNeighborsClassifier

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

    trajectory_embeddings, language_embeddings = [], []

    for item in tqdm(dataset):
        batched_input = {k: v.to(model.device) for k,v in collator([item]).items()}
        trajectory_embedding, language_embedding = model.model(**batched_input)

        trajectory_embeddings.append(trajectory_embedding)
        language_embeddings.append(language_embedding)

    trajectory_embeddings = torch.cat(trajectory_embeddings, dim=0).detach().cpu().numpy()
    language_embeddings = torch.cat(language_embeddings, dim=0).detach().cpu().numpy()

    y = torch.arange(trajectory_embeddings.shape[0]).numpy()

    classifier = KNeighborsClassifier(n_neighbors=1).fit(trajectory_embeddings, y)
    distances, kneighbours = classifier.kneighbors(language_embeddings, n_neighbors=1, return_distance=True)

    acc = 0.0
    for i in range(kneighbours.shape[0]):
        if y[i] in kneighbours[i]:
            acc += 1.0

    print(f"Accuracy: {acc / kneighbours.shape[0]*100:.2f} %")


if __name__ == "__main__":
    eval()
import torch
import hydra

from torch.utils.data import DataLoader, random_split
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer
from omegaconf import open_dict

from clap import Clap, ContrastiveTraining
from data.materialize import get_dataset_and_collator_tfds


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(config):
    seed_everything(42, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.language_encoder.model_name)
    training_set, collator = get_dataset_and_collator_tfds(**config.data, tokenizer=tokenizer)

    with open_dict(config):
        config.model.language_encoder.eos_token_id = tokenizer.eos_token_id

    model = Clap(**config.model)
    lightning_model = ContrastiveTraining(model, epochs=config.trainer.max_epochs, **config.training)

    if config.dataloader.validate:
        validation_set_size = int(len(training_set) * config.dataloader.validation_percentage)
        training_set_size = len(training_set) - validation_set_size

        training_set, validation_set = random_split(training_set, [training_set_size, validation_set_size])

        val_dataloader = DataLoader(validation_set, 
                   batch_size=config.dataloader.batch_size, 
                   num_workers=0,
                   collate_fn=collator,  
                   drop_last=False)

    train_dataloader = DataLoader(training_set, 
                   batch_size=config.dataloader.batch_size, 
                   num_workers=0,
                   collate_fn=collator, 
                   drop_last=False)
    
    logger = WandbLogger(**config.logger)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(logger=logger, callbacks=[lr_monitor], **config.trainer)
    trainer.fit(lightning_model, train_dataloader, val_dataloader) if config.dataloader.validate else trainer.fit(lightning_model, train_dataloader)

if __name__ == "__main__":
    train()
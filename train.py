import torch
import hydra

from torch.utils.data import DataLoader, random_split
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import seed_everything
from transformers import AutoTokenizer
from omegaconf import open_dict

from clap import Clap, ContrastiveTraining
from data.materialize import get_dataset_and_collator


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(config):
    seed_everything(42, workers=True)

    if any(gpu in torch.torch.cuda.get_device_name() for gpu in ['A100', 'H100']):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    tokenizer = AutoTokenizer.from_pretrained(config.model.language_encoder.model_name)
    training_set, collator = get_dataset_and_collator(config.data.name, **config.data.dataset, tokenizer=tokenizer)

    with open_dict(config):
        config.model.language_encoder.eos_token_id = tokenizer.eos_token_id

    model = Clap(**config.model)

    max_num_steps = (len(training_set) // (config.dataloader.batch_size * torch.cuda.device_count())) * config.trainer.max_epochs
    lightning_model = ContrastiveTraining(model, scheduler_max_steps=max_num_steps, **config.training)

    if config.dataloader.validate:
        validation_set_size = int(len(training_set) * config.dataloader.validation_percentage)
        training_set_size = len(training_set) - validation_set_size

        training_set, validation_set = random_split(training_set, [training_set_size, validation_set_size])

        val_dataloader = DataLoader(validation_set, 
                   batch_size=config.dataloader.batch_size, 
                   num_workers=0,
                   collate_fn=collator)

    train_dataloader = DataLoader(training_set, 
                   batch_size=config.dataloader.batch_size, 
                   num_workers=config.data.workers,
                   collate_fn=collator)
    
    logger = WandbLogger(**config.logger)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(every_n_epochs=1)

    trainer = Trainer(logger=logger, callbacks=[lr_monitor, checkpoint_callback], **config.trainer)

    fit_args = {'model': lightning_model, 'train_dataloaders': train_dataloader}

    if config.dataloader.validate:
        fit_args['val_dataloaders'] = val_dataloader
    if config.resume_training:
        fit_args['ckpt_path'] = config.checkpoint_path

    trainer.fit(**fit_args)

if __name__ == "__main__":
    train()
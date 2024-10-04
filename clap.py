import torch
from torch import nn
import numpy as np
import lightning as L
from transformers import CLIPTextModelWithProjection

from action_encoder import ActionEncoder


class Clap(nn.Module):
    def __init__(self, action_encoder, language_encoder):
        super().__init__()

        self.action_encoder = ActionEncoder(**action_encoder)      
        self.language_encoder = CLIPTextModelWithProjection.from_pretrained(language_encoder.model_name)

        print(f"\nAction encoder has {sum([np.prod(p.size()) for p in self.action_encoder.parameters()])/1_000_000:.1f} M parameters")
        print(f"Language encoder has {sum([np.prod(p.size()) for p in self.language_encoder.parameters()])/1_000_000:.1f} M parameters\n")

        self.language_encoder.text_model.eos_token_id = language_encoder.eos_token_id
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, 
                action_tokens, 
                language_tokens,
                action_pad_attention_mask,
                language_pad_attention_mask
               ):
        trajectory_embedding = self.action_encoder(action_tokens, action_pad_attention_mask)
        language_embedding = self.language_encoder(language_tokens, attention_mask=language_pad_attention_mask).text_embeds

        # normalized features
        trajectory_features = trajectory_embedding / trajectory_embedding.norm(dim=1, keepdim=True)
        language_features = language_embedding / language_embedding.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_trajectory = logit_scale * trajectory_features @ language_features.t()
        logits_per_description = logits_per_trajectory.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_trajectory, logits_per_description


class ContrastiveTraining(L.LightningModule):
    def __init__(self,  
                 model: nn.Module, 
                 learning_rate: float = 1e-3, 
                 weight_decay: float = 0.01,
                 steps: int = 200_000,
                ):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.steps = steps

    def training_step(self, batch, batch_idx):
        """
        batch is composed of:
        language_tokens: torch.Tensor - A batch of language tokens of shape (B, N, T_D).
        actions: torch.Tensor - A batch of actions of shape (B, T, A_D). 
        """   
        trajectory_similarity, description_similarity = self.model(**batch)

        batch_size = trajectory_similarity.size(0)
        labels = torch.arange(batch_size, device=trajectory_similarity.device)

        loss = (
                torch.nn.functional.cross_entropy(trajectory_similarity, labels) +
                torch.nn.functional.cross_entropy(description_similarity, labels)
               ) / 2

        self.log("train_loss", loss)

        return loss

    
    def validation_step(self, batch, batch_idx): 
        trajectory_similarity, description_similarity = self.model(**batch)

        batch_size = trajectory_similarity.size(0)
        labels = torch.arange(batch_size, device=trajectory_similarity.device)

        loss = (
                torch.nn.functional.cross_entropy(trajectory_similarity, labels) +
                torch.nn.functional.cross_entropy(description_similarity, labels)
               ) / 2

        self.log("val_loss", loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]

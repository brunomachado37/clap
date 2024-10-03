import torch
from torch import nn


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim, num_heads, num_layers, dropout_rate, projection_dim, max_sequence_length):
        super().__init__()

        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)

        self.sequence_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=4*hidden_dim,
                    dropout=dropout_rate,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=num_layers,
                norm=nn.LayerNorm(hidden_dim)
        )

        self.action_projection = nn.Linear(hidden_dim, projection_dim)

    def forward(self, action_tokens, action_pad_attention_mask=None):
        if action_pad_attention_mask is None:
            action_pad_attention_mask = action_tokens.ne(-100).sum(-1).bool()

        # Get the readout token indexes
        readout_indexes = (~action_pad_attention_mask).double().argmax(dim=-1)              # If there is no padding, this function will return 0
        readout_indexes[readout_indexes==0] = action_pad_attention_mask.size(-1) - 1        # Correct the readout token index for sequences without padding

        # Zero out the input for padding tokens
        action_tokens = action_tokens.masked_fill(~action_pad_attention_mask.unsqueeze(-1), 0)

        action_embedding = self.action_embedding(action_tokens)

        sequence_length = action_tokens.size(1)
        position_ids = torch.arange(sequence_length, device=action_tokens.device).unsqueeze(0)
        position_embedding = self.position_embedding(position_ids)

        action_embedding += position_embedding

        causal_mask = torch.triu(torch.full((sequence_length, sequence_length), float("-inf"), dtype=action_embedding.dtype, device=action_embedding.device), diagonal=1)
        padding_mask = torch.zeros_like(action_pad_attention_mask, dtype=causal_mask.dtype, device=causal_mask.device).masked_fill(~action_pad_attention_mask, float("-inf"))
        sequence_embedding = self.sequence_encoder(action_embedding, mask=causal_mask, is_causal=True, src_key_padding_mask=padding_mask)
        
        trajectory_embedding = self.action_projection(sequence_embedding[torch.arange(readout_indexes.size(0)), readout_indexes])

        return trajectory_embedding
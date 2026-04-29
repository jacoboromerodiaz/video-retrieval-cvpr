"""Flan-T5 encoder-only text encoder for query encoding."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5EncoderModel


class FlanT5Encoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/flan-t5-xl",
        max_length: int = 512,
    ):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        last_hidden = self.model(**inputs).last_hidden_state  # (B, T, D)

        # mean-pool over non-padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        embeddings = last_hidden * mask  # (B, T, D)

        return embeddings

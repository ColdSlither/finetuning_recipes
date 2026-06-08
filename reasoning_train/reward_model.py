import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_reward_model(path):
    """
    Load a trained MiniLM reward model from a directory.

    Usage:
        model, tokenizer = load_reward_model("paperbd/minilm-reward")
        score = model.score("reference text", "candidate text")
    """
    path = Path(path).resolve()

    encoder = AutoModel.from_pretrained(str(path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)

    head = nn.Sequential(nn.Dropout(0.1), nn.Linear(384, 1), nn.Sigmoid())

    state = torch.load(str(path / "pytorch_model.bin"), weights_only=True, map_location="cpu")
    own = head.state_dict()
    for k in own:
        if f"head.{k}" in state:
            own[k].copy_(state[f"head.{k}"])
    head.load_state_dict(own)
    head.eval()

    class RewardScorer:
        def __init__(self):
            self.encoder = encoder
            self.head = head

        def score(self, reference, response):
            text = f"{reference} [SEP] {response}"
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                return self.head(pooled).item()

        def score_batch(self, references, responses):
            texts = [f"{r} [SEP] {c}" for r, c in zip(references, responses)]
            enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                pooled = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
                return self.head(pooled).squeeze(-1).tolist()

    return RewardScorer(), tokenizer

"""Load the released PLM-Interact checkpoint on the Hugging Face ESM backbone."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM

from computing_scores.plm_interact.config import BASE_MODEL, EMBEDDING_SIZE, model_directory


class PPIModel(nn.Module):
    def __init__(self, base_directory: Path):
        super().__init__()
        self.esm_mask = AutoModelForMaskedLM.from_pretrained(str(base_directory))
        self.classifier = nn.Linear(EMBEDDING_SIZE, 1)

    def forward_logits(self, features):
        output = self.esm_mask.base_model(**features, return_dict=True)
        embedding = F.relu(output.last_hidden_state[:, 0, :])
        return self.classifier(embedding).view(-1)

    def forward_test(self, features):
        return torch.sigmoid(self.forward_logits(features))


def load_ppi_model(*, cache_root: Path, checkpoint_path: Path, device):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and isinstance(raw.get("model"), dict) else raw
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = PPIModel(model_directory(cache_root, BASE_MODEL)).to(device)
    incompat = model.load_state_dict(state, strict=False)
    allowed_unexpected = {
        "esm_mask.esm.embeddings.position_embeddings.weight",
        "esm_mask.esm.embeddings.position_ids",
    }
    bad_unexpected = set(incompat.unexpected_keys).difference(allowed_unexpected)
    if incompat.missing_keys or bad_unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    return model.eval()

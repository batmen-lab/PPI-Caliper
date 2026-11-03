"""Load official MINT components with the settings used for our human PPI heads."""
from argparse import Namespace

import torch
from mint.helpers.extract import CollateFn, MINTWrapper
from mint.helpers.predict import SimpleMLP
from tqdm import tqdm

ENCODER_CONFIG = Namespace(
    encoder_layers=33,
    encoder_embed_dim=1280,
    encoder_attention_heads=20,
    token_dropout=True,
)


def load_encoder(checkpoint, device):
    with torch.serialization.safe_globals([Namespace]):
        return MINTWrapper(
            ENCODER_CONFIG, checkpoint, freeze_percent=1.0,
            use_multimer=True, sep_chains=True, device=device,
        ).eval()


def embedding_batches(encoder, dataset, device, batch_size, description):
    """Yield embeddings on the encoder's device in input order."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=CollateFn())
    for chains, chain_ids in tqdm(loader, desc=description):
        yield encoder(chains.to(device), chain_ids.to(device))


def load_head(checkpoint, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    state = saved.get("model_state_dict", saved)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    head = SimpleMLP(state["project.0.weight"].shape[1], 1).to(device)
    head.load_state_dict(state)
    return head.eval()

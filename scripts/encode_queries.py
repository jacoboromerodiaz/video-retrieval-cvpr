import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from covr.data.dataset import RetrievalDataset
from covr.models.flan_t5_encoder import FlanT5Encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_encoder(encoder: str, model_name: str = "google/flan-t5-large"):
    if encoder == "FlanT5":
        model = FlanT5Encoder(model_name)
        return model
    else:
        raise ValueError(f"Unknown encoder: '{encoder}'")


@torch.no_grad()
def encode_queries(
    json_path: str | Path,
    video_root: str | Path,
    splits: list[str],
    output_dir: str | Path,
    batch_key: str = "modification_text",
    encoder: str = "FlanT5",
    pretrained_model: str = "google/flan-t5-xl",
    batch_size: int = 64,
    num_workers: int = 4,
    mode: str = "prod",
) -> None:
    device = "mps" if mode == "dev" else "cuda"

    encoder_model = load_encoder(encoder, pretrained_model).to(device).eval()

    for split in splits:
        split_output_dir = Path(output_dir) / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = RetrievalDataset(
            json_path, video_root, split=split, load_frames=False
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        log.info(
            "Encoding queries split=%s (%d queries) → %s",
            split,
            len(dataset),
            split_output_dir,
        )

        for batch in tqdm(loader, desc=f"Encoding {split}"):
            embeddings = encoder_model(batch[batch_key])
            for query_id, emb in zip(batch["source_video_id"], embeddings):
                torch.save(emb.cpu(), split_output_dir / f"{query_id}.pt")

    log.info("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode queries with text encoder")
    p.add_argument("--config", default="configs/encoder/flan_prod.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    encode_queries(**cfg)

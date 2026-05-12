import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from covr.data.dataset import (
    RichTextRetrievalDataset,
    RetrievalDatasetTest,
)
from covr.models.flan_t5_encoder import FlanT5Encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_encoder(encoder: str, model_name: str = "google/flan-t5-large"):
    if encoder == "FlanT5":
        return FlanT5Encoder(model_name)
    raise ValueError(f"Unknown encoder: '{encoder}'")


@torch.no_grad()
def encode_queries(
    data_file_path: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    batch_key: str = "modification_text",
    encoder: str = "FlanT5",
    pretrained_model: str = "google/flan-t5-xl",
    batch_size: int = 64,
    num_workers: int = 4,
    mode: str = "prod",
    test: bool = False,
    splits: list[str] | None = None,
) -> None:
    device = "mps" if mode == "dev" else "cuda"

    encoder_model = load_encoder(encoder, pretrained_model).to(device).eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iterations: list = splits if test else [None]

    for split in iterations:
        if test:
            dataset = RetrievalDatasetTest(
                video_root, data_file_path, split=split, load_frames=False
            )
        else:
            dataset = RichTextRetrievalDataset(
                video_root, csv_path=data_file_path, load_frames=False
            )

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        label = f"split={split}" if split else "rich-text"
        log.info(
            "Encoding queries %s (%d queries) → %s",
            label,
            len(dataset),
            output_dir,
        )

        for batch in tqdm(loader, desc=f"Encoding {label}"):
            embeddings = encoder_model(batch[batch_key])
            for query_id, emb in zip(batch["source_video_id"], embeddings):
                if test:
                    out_path = output_dir / split / f"{query_id}.pt"
                else:
                    out_path = output_dir / f"{query_id}.pt"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(emb.cpu(), out_path)

    log.info("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode queries with text encoder")
    p.add_argument("--config", default="configs/encoder/flan/flan_train_prod.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    encode_queries(**cfg)

import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from covr.data.dataset import RetrievalDataset
from covr.models.vjepa import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


@torch.no_grad()
def encode_gallery(
    json_path: str | Path,
    video_root: str | Path,
    splits: list[str],
    output_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 4,
    mode: str = "prod",
) -> None:
    model, device = load_model(
        mode
    )  # fix E0606 + W0613: usaba cfg en vez del argumento

    for split in splits:
        split_output_dir = Path(output_dir) / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = RetrievalDataset(json_path, video_root, split=split)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        log.info(
            "Encoding split=%s (%d videos) → %s", split, len(dataset), split_output_dir
        )

        for batch in tqdm(loader, desc=f"Encoding {split}"):
            frames = batch["source_frames"].to(device)
            embeddings = model(frames)
            for video_id, emb in zip(batch["source_video_id"], embeddings):
                torch.save(emb.cpu(), split_output_dir / f"{video_id}.pt")

    log.info("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode gallery videos with V-JEPA2")
    p.add_argument("--config", default="configs/encoder/vjepa.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:  # fix W1514
        cfg = yaml.safe_load(f)

    encode_gallery(**cfg)

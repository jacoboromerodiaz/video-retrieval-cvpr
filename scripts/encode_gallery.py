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


def encode_video(model, frames: torch.Tensor, chunk_size: int = 16) -> torch.Tensor:
    """
    frames: [C, T, H, W]
    returns: [N, vid_embd]
    """
    T = frames.shape[1]
    chunks = []
    for start in range(0, T, chunk_size):
        chunk = frames[:, start : start + chunk_size].unsqueeze(
            0
        )  # [1, C, chunk_size, H, W]
        emb = model(chunk).squeeze(0)  # [N, vid_embd]
        chunks.append(emb)
    return torch.cat(chunks, dim=0)  # [N, vid_embd]


@torch.no_grad()
def encode_gallery(
    json_path: str | Path,
    video_root: str | Path,
    splits: list[str],
    output_dir: str | Path,
    chunk_size: int = 16,
    num_workers: int = 4,
    target_fps: int = 4,
    mode: str = "prod",
) -> None:
    model, device = load_model(mode)

    for split in splits:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # batch_size!= 1 for future implementation
        dataset = RetrievalDataset(
            json_path, video_root, split=split, target_fps=target_fps
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=num_workers)

        log.info("Encoding split=%s (%d videos) → %s", split, len(dataset), output_dir)

        for batch in tqdm(loader, desc=f"Encoding {split}"):
            frames = batch["source_frames"].squeeze(0).to(device)  # [C, T, H, W]
            video_id = batch["source_video_id"][0]

            emb = encode_video(model, frames, chunk_size)  # [N, vid_embd]

            out_path = output_dir / f"{video_id}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb.cpu(), out_path)

    log.info("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode gallery videos with V-JEPA2")
    p.add_argument("--config", default="configs/encoder/vjepa_prod.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    encode_gallery(**cfg)

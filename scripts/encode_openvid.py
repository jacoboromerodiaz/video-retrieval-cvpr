import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from covr.data.dataset import SingleVideoDataset, find_video, load_frames
from covr.models.vjepa import load_model, load_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


@torch.inference_mode()
def encode_video(
    model,
    processor,
    frames: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    frames: [C, T, H, W]
    returns: [vid_embd]  (mean-pooled over patches)
    """
    device = next(model.parameters()).device
    T = frames.shape[1]
    chunks = []
    use_cuda = device.type == "cuda"

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = frames[:, start:end].permute(1, 0, 2, 3)
        processed = processor(chunk)
        processed_chunk = processed[0] if isinstance(processed, list) else processed
        processed_chunk = processed_chunk.unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            emb = model(processed_chunk)

        chunks.append(emb.squeeze(0).float())

    patches = torch.cat(chunks, dim=0)  # [N, D]
    return patches.mean(dim=0)  # [D]


@torch.inference_mode()
def encode_openvid(
    data_file_path: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    chunk_size: int = 64,
    num_workers: int = 4,
    target_fps: int = 4,
    mode: str = "prod",
    id_col: str = "videoid",
    caption_col: str = "caption",
) -> None:
    model, device = load_model(mode)
    processor = load_processor()
    output_dir = Path(output_dir)
    video_root_path = Path(video_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SingleVideoDataset(
        video_root=video_root,
        csv_path=data_file_path,
        id_col=id_col,
        caption_col=caption_col,
        load_frames=False,
        target_fps=target_fps,
    )

    done_ids = {p.stem for p in output_dir.rglob("*.pt")}
    log.info("Found %d already-encoded videos, skipping.", len(done_ids))

    def _video_exists(stem: str) -> bool:
        return any(
            (video_root_path / f"{stem}{ext}").exists()
            for ext in (".mp4", ".webm", ".avi", ".mkv")
        )

    pending, skipped = [], 0
    for i, s in enumerate(dataset.samples):
        vid_id = s["video_source"]
        if not _video_exists(vid_id):
            skipped += 1
            continue
        if vid_id not in done_ids:
            pending.append(i)

    if skipped:
        log.warning("Skipped %d samples with missing video files.", skipped)
    if not pending:
        log.info("All videos already encoded, skipping.")
        return

    loader = DataLoader(Subset(dataset, pending), batch_size=1, num_workers=num_workers)
    log.info("Encoding %d/%d videos → %s", len(pending), len(dataset), output_dir)

    for batch in tqdm(loader, desc="Encoding"):
        video_id = batch["source_video_id"][0]
        out_path = output_dir / f"{video_id}.pt"
        if out_path.exists():
            continue
        try:
            frames = load_frames(find_video(video_root_path, video_id), target_fps)
        except FileNotFoundError:
            log.warning("Video not found, skipping: %s", video_id)
            continue

        emb = encode_video(model, processor, frames.to(device), chunk_size)
        if torch.isnan(emb).any():
            log.warning("NaN from %s on %s, retrying on CPU", device, video_id)
            model.to("cpu")
            emb = encode_video(model, processor, frames.to("cpu"), chunk_size)
            model.to(device)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb.cpu().to(torch.bfloat16), out_path)

    log.info("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode OpenVid-1M videos with V-JEPA2.1")
    p.add_argument("--config", default="configs/encode/openvid/openvid_dev.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    encode_openvid(**cfg)

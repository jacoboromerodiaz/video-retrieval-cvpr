import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from covr.data.dataset import (
    RichTextRetrievalDataset,
    RetrievalDatasetTest,
    find_video,
    load_frames,
)
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
    returns: [N, vid_embd]
    """
    device = next(model.parameters()).device
    T = frames.shape[1]
    chunks = []

    print(f"[input] frames.shape = {frames.shape}")
    print(f"[input] frames.dtype = {frames.dtype}")
    print(f"[input] frames.device = {frames.device}")
    print(f"[model] device = {device}")
    print(f"[config] chunk_size = {chunk_size}, T = {T}")

    use_cuda = device.type == "cuda"

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        print(f"\n--- chunk {start}:{end} ---")

        chunk = frames[:, start:end]
        print(f"[1] chunk              {chunk.shape}  # [C, t, H, W]")

        chunk = chunk.permute(1, 0, 2, 3)
        print(f"[2] after permute      {chunk.shape}  # [t, C, H, W]")

        processed = processor(chunk)
        print(f"[3] processor output type = {type(processed)}")

        if isinstance(processed, list):
            print(f"[3] processor list len = {len(processed)}")
            processed_chunk = processed[0]
        else:
            processed_chunk = processed

        print(f"[4] processed_chunk    {processed_chunk.shape}  # [C, t, H, W]")
        print(f"[4] processed dtype    {processed_chunk.dtype}")
        print(f"[4] processed device   {processed_chunk.device}")

        processed_chunk = processed_chunk.unsqueeze(0)
        print(f"[5] after unsqueeze    {processed_chunk.shape}  # [B, C, t, H, W]")

        processed_chunk = processed_chunk.to(device)
        print(f"[6] after to(device)   {processed_chunk.shape}")
        print(f"[6] device             {processed_chunk.device}")

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_cuda,
        ):
            emb = model(processed_chunk)

        print(f"[7] model output       {emb.shape}")

        emb = emb.squeeze(0)
        print(f"[8] after squeeze      {emb.shape}")

        emb = emb.float()
        print(f"[9] after float        {emb.shape}, dtype={emb.dtype}")

        chunks.append(emb)

    out = torch.cat(chunks, dim=0)
    print(f"\n[output] cat output    {out.shape}")

    return out


@torch.inference_mode()
def encode_gallery(
    data_file_path: str | Path,
    video_root: str | Path,
    output_dir: str | Path,
    chunk_size: int = 64,
    num_workers: int = 4,
    target_fps: int = 4,
    mode: str = "prod",
    test: bool = False,
    splits: list[str] | None = None,
) -> None:
    model, device = load_model(mode)
    processor = load_processor()
    output_dir = Path(output_dir)
    video_root_path = Path(video_root)
    iterations: list = splits if test else [None]

    for split in iterations:
        output_dir.mkdir(parents=True, exist_ok=True)

        # batch_size!= 1 for future implementation
        if test:
            dataset = RetrievalDatasetTest(
                video_root, data_file_path, split=split, target_fps=target_fps
            )
        else:
            dataset = RichTextRetrievalDataset(
                video_root, csv_path=data_file_path, target_fps=target_fps
            )

            done_ids = {
                str(p.relative_to(output_dir).with_suffix(""))
                for p in output_dir.rglob("*.pt")
            }
            log.info("Found %d already-encoded videos, skipping.", len(done_ids))

        def _video_exists(stem: str) -> bool:
            return any(
                (video_root_path / f"{stem}{ext}").exists()
                for ext in (".mp4", ".webm", ".avi", ".mkv")
            )

        pending = []
        skipped = 0
        for i, s in enumerate(dataset.samples):
            src_id = s["video_source"]
            tgt_id = s["video_target"]
            if not _video_exists(src_id) or (not test and not _video_exists(tgt_id)):
                skipped += 1
                continue
            if src_id not in done_ids or (not test and tgt_id not in done_ids):
                pending.append(i)
        if skipped:
            log.warning("Skipped %d samples with missing video files.", skipped)

        label = f"split={split}" if split else "rich-text"
        if not pending:
            if skipped and skipped == len(dataset.samples):
                log.warning("No samples for %s, missing video files.", label)
            else:
                log.info("All videos already encoded for %s, skipping.", label)
            continue

        loader = DataLoader(
            Subset(dataset, pending), batch_size=1, num_workers=num_workers
        )

        log.info(
            "Encoding %s (%d/%d pending) → %s",
            label,
            len(pending),
            len(dataset),
            output_dir,
        )

        for batch in tqdm(loader, desc=f"Encoding {label}"):
            video_items = [
                (batch["source_video_id"][0], batch["source_frames"].squeeze(0))
            ]
            if not test:
                video_items.append((batch["target_video_id"][0], None))
            for video_id, frames in video_items:
                out_path = (
                    (output_dir / split / f"{video_id}.pt")
                    if test
                    else (output_dir / f"{video_id}.pt")
                )
                if out_path.exists():
                    continue
                if frames is None:
                    try:
                        frames = load_frames(
                            find_video(video_root_path, video_id), target_fps
                        )
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
    p = argparse.ArgumentParser(description="Encode gallery videos with V-JEPA2.1")
    p.add_argument("--config", default="configs/encoder/vjepa_dev.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    encode_gallery(**cfg)

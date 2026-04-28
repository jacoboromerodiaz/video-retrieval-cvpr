"""Datasets used for VJEPA-2.1 embeddings"""

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

_SPLIT_INDEX = {"webvid": 0, "ss2": 1}


def load_frames(
    path: str | Path,
    num_frames: int = 8,
) -> torch.Tensor:
    decoder = VideoDecoder(str(path))

    indices = torch.linspace(0, len(decoder) - 1, num_frames).long().tolist()
    frames = decoder.get_frames_at(indices=indices).data  # [T, C, H, W]

    tensor = frames.float() / 255.0
    tensor = tensor.permute(1, 0, 2, 3)  # [C, T, H, W]

    return tensor


def find_video(video_root: Path, stem: str) -> Path:
    for ext in (".mp4", ".webm", ".avi", ".mkv"):
        p = video_root / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No video found for id '{stem}' in {video_root}")


class RetrievalDataset(Dataset):
    """Source video + modification text → target video id triples."""

    def __init__(
        self,
        json_path: str | Path,
        video_root: str | Path,
        split: Literal["ss2", "webvid"],
        load_frames: bool = True,
    ):
        self.video_root = Path(video_root)
        self.load_frames = load_frames

        with open(json_path, encoding="utf-8") as f:
            self.samples = json.load(f)[_SPLIT_INDEX[split]][split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "source_video_id": str(s["id"]),
            "description_source": s["description_source"],
            "description_target": s["description_target"],
            "modification_text": s["modification_text"],
            "reasoned_description": s["reasoned_target_video_description__main"],
            "target_video_id": str(s["video_target"]),
        }
        if self.load_frames:
            video_filename = Path(str(s["video_source"])).name
            item["source_frames"] = load_frames(
                find_video(self.video_root, video_filename)
            )
        return item


class VideoEmbeddingDataset(Dataset):
    """.pt embeddings from a directory."""

    def __init__(self, embeddings_dir: str | Path):
        self.paths = sorted(Path(embeddings_dir).glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files in {embeddings_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return {
            "video_id": path.stem,
            "embedding": torch.load(path, weights_only=True),
        }

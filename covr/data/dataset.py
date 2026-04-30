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
    target_fps: int = 4,
) -> torch.Tensor:
    decoder = VideoDecoder(str(path))
    native_fps = decoder.metadata.average_fps or 24.0
    step = max(1, round(native_fps / target_fps))
    indices = list(range(0, len(decoder), step))
    frames = decoder.get_frames_at(indices=indices).data  # [T, C, H, W]
    return (frames.float() / 255.0).permute(1, 0, 2, 3)  # [C, T, H, W]


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
        target_fps: int = 4,
    ):
        self.video_root = Path(video_root)
        self.load_frames_flag = load_frames
        self.target_fps = target_fps

        with open(json_path, encoding="utf-8") as f:
            self.samples = json.load(f)[_SPLIT_INDEX[split]][split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "id": str(s["id"]),
            "description_source": s["description_source"],
            "description_target": s["description_target"],
            "modification_text": s["modification_text"],
            "source_video_id": str(s["video_source"]).split("/")[-1],
            "target_video_id": str(s["video_target"]).split("/")[-1],
        }
        if self.load_frames_flag:
            video_filename = Path(str(s["video_source"])).name
            item["source_frames"] = load_frames(
                find_video(self.video_root, video_filename),
                self.target_fps,
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

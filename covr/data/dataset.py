"""Datasets used for VJEPA-2.1 embeddings"""

import json
import csv
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
    for ext in (".mp4", ".webm"):
        p = video_root / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No video found for id '{stem}' in {video_root}")


class RichTextRetrievalDataset(Dataset):
    """Source video + modification text → target video id triples (CSV)."""

    def __init__(
        self,
        video_root: str | Path,
        csv_path: str | Path = "cvprw-covr_train.csv",
        load_frames: bool = True,
        target_fps: int = 4,
    ):
        self.video_root = Path(video_root)
        self.load_frames_flag = load_frames
        self.target_fps = target_fps

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.samples = [
                {
                    "id": row["id"],
                    "video_source": row["pth1"],
                    "video_target": row["pth2"],
                    "modification_text": row["edit"],
                }
                for row in reader
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "id": str(s["id"]),
            "modification_text": str(s["modification_text"]),
            "source_video_id": str(s["video_source"]),
            "target_video_id": str(s["video_target"]),
        }
        if self.load_frames_flag:
            item["source_frames"] = load_frames(
                find_video(self.video_root, str(s["video_source"])),
                self.target_fps,
            )
        return item


class RetrievalDatasetTest(Dataset):
    """Source video + modification text."""

    def __init__(
        self,
        video_root: str | Path,
        json_path: str | Path,
        split: Literal["ss2", "webvid", "all"] = "all",
        load_frames: bool = True,
        target_fps: int = 4,
    ):
        self.video_root = Path(video_root)
        self.load_frames_flag = load_frames
        self.target_fps = target_fps

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        splits = ("ss2", "webvid") if split == "all" else (split,)
        self.samples = [
            (s, sample) for s in splits for sample in data[_SPLIT_INDEX[s]][s]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        split, s = self.samples[idx]
        item = {
            "id": str(s["id"]),
            "source_video_id": str(s["video_source"]),
            "modification_text": s["modification_text"],
        }
        if self.load_frames_flag:
            item["source_frames"] = load_frames(
                find_video(self.video_root / split, str(s["video_source"])),
                self.target_fps,
            )
        return item

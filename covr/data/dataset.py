"""Datasets used for VJEPA-2.1 embeddings"""

import json
import csv
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

_SPLIT_INDEX = {"webvid": 0, "ss2": 1}


def _load_trunc(path: Path, max_patches: int | None) -> torch.Tensor:
    t = torch.load(path, weights_only=True)
    if max_patches is not None and t.shape[0] > max_patches:
        t = t[:max_patches]
    return t


def _clean_ids(ids: set[str], directory: Path) -> set[str]:
    """Return subset of ids whose .pt files contain no NaN values."""
    ok = set()
    for vid_id in ids:
        t = torch.load(directory / f"{vid_id}.pt", weights_only=True)
        if not torch.isnan(t).any():
            ok.add(vid_id)
    return ok


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


# Embedding-backed datasets
class TrainDataset(Dataset):
    """
    RichTextRetrievalDataset backed by pre-computed vjepa + flan embeddings.
    """

    def __init__(
        self,
        csv_path: str | Path,
        embeddings_dir: str | Path,
        queries_dir: str | Path,
        dev: bool = False,
        max_patches: int | None = None,
    ):
        full = RichTextRetrievalDataset(
            video_root="", csv_path=csv_path, load_frames=False
        )
        self.emb_dir = Path(embeddings_dir) / "train"
        self.query_dir = Path(queries_dir) / "train"
        self.max_patches = max_patches

        if dev:
            all_gallery = {
                str(p.relative_to(self.emb_dir).with_suffix(""))
                for p in self.emb_dir.rglob("*.pt")
            }
            available_gallery = _clean_ids(all_gallery, self.emb_dir)
            if len(available_gallery) < len(all_gallery):
                print(
                    f"[dev] filtered {len(all_gallery) - len(available_gallery)}"
                    " NaN gallery files"
                )
            available_queries = {
                str(p.relative_to(self.query_dir).with_suffix(""))
                for p in self.query_dir.rglob("*.pt")
            }
            self.indices = [
                i
                for i in range(len(full))
                if full[i]["source_video_id"] in available_gallery
                and full[i]["target_video_id"] in available_gallery
                and full[i]["source_video_id"] in available_queries
            ]
            print(f"[dev] {len(self.indices)}/{len(full)} pairs available")
        else:
            self.indices = list(range(len(full)))

        self.inner = full

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.inner[self.indices[idx]]
        return {
            "source_patches": _load_trunc(
                self.emb_dir / f"{item['source_video_id']}.pt", self.max_patches
            ),
            "target_patches": _load_trunc(
                self.emb_dir / f"{item['target_video_id']}.pt", self.max_patches
            ),
            "query_emb": torch.load(
                self.query_dir / f"{item['source_video_id']}.pt", weights_only=True
            ),
        }


class SingleVideoDataset(Dataset):
    """Individual videos from a flat metadata CSV (e.g., OpenVid-1M)."""

    def __init__(
        self,
        video_root: str | Path,
        csv_path: str | Path,
        id_col: str = "videoid",
        caption_col: str = "caption",
        load_frames: bool = True,
        target_fps: int = 4,
    ):
        self.video_root = Path(video_root)
        self.load_frames_flag = load_frames
        self.target_fps = target_fps
        self.id_col = id_col
        self.caption_col = caption_col

        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.samples = [
                {
                    "video_source": row[id_col],
                    "caption": row[caption_col],
                }
                for row in reader
            ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        item = {
            "source_video_id": str(s["video_source"]),
            "caption": str(s["caption"]),
        }
        if self.load_frames_flag:
            item["source_frames"] = load_frames(
                find_video(self.video_root, str(s["video_source"])),
                self.target_fps,
            )
        return item


class ValDataset(Dataset):
    """
    RetrievalDatasetTest backed by pre-computed vjepa + flan embeddings.
    """

    def __init__(
        self,
        json_path: str | Path,
        embeddings_dir: str | Path,
        queries_dir: str | Path,
        split_type: Literal["val", "test"] = "val",
        split: Literal["ss2", "webvid", "all"] = "all",
        max_patches: int | None = None,
    ):
        self.inner = RetrievalDatasetTest(
            video_root="", json_path=json_path, split=split, load_frames=False
        )
        self.emb_dir = Path(embeddings_dir) / split_type
        self.query_dir = Path(queries_dir) / split_type
        self.max_patches = max_patches

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        split_name, raw = self.inner.samples[idx]
        src_id = item["source_video_id"]
        tgt_id = str(raw["video_target_covr-r"])
        return {
            "source_patches": _load_trunc(
                self.emb_dir / split_name / f"{src_id}.pt", self.max_patches
            ),
            "target_patches": _load_trunc(
                self.emb_dir / split_name / f"{tgt_id}.pt", self.max_patches
            ),
            "query_emb": torch.load(
                self.query_dir / split_name / f"{src_id}.pt", weights_only=True
            ),
        }

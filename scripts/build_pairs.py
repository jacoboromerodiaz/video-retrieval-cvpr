import argparse
import csv
import difflib
import logging
from pathlib import Path

import faiss
import numpy as np
import torch
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _load_embeddings(
    embeddings_dir: Path,
) -> tuple[list[str], np.ndarray]:
    """Load all [768] .pt embeddings; return (video_ids, matrix [M, 768])."""
    paths = sorted(embeddings_dir.rglob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt files found in {embeddings_dir}")

    video_ids, vecs = [], []
    for p in tqdm(paths, desc="Loading embeddings"):
        t = torch.load(p, weights_only=True).float()
        video_ids.append(p.stem)
        vecs.append(t.numpy())

    matrix = np.stack(vecs, axis=0).astype(np.float32)  # [M, 768]
    faiss.normalize_L2(matrix)
    return video_ids, matrix


def _load_captions(
    csv_path: Path,
    id_col: str,
    caption_col: str,
) -> dict[str, str]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row[id_col]: row[caption_col] for row in reader}


def _edit_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def build_pairs(
    embeddings_dir: str | Path,
    captions_csv: str | Path,
    output_csv: str | Path,
    k: int = 50,
    sim_min: float = 0.7,
    sim_max: float = 0.9,
    min_edit_ratio: float = 0.2,
    id_col: str = "videoid",
    caption_col: str = "caption",
    mode: str = "prod",
) -> None:
    embeddings_dir = Path(embeddings_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading embeddings from %s", embeddings_dir)
    video_ids, matrix = _load_embeddings(embeddings_dir)
    M = len(video_ids)
    log.info("Loaded %d embeddings (dim=%d)", M, matrix.shape[1])

    log.info("Loading captions from %s", captions_csv)
    captions = _load_captions(Path(captions_csv), id_col, caption_col)

    log.info("Building FAISS index")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    k_actual = min(k + 1, M)  # +1 because the query itself is always returned
    log.info("Querying %d nearest neighbours per video", k_actual - 1)

    seen: set[tuple[str, str]] = set()
    rows: list[dict] = []
    pair_id = 0

    batch_size = 512
    for start in tqdm(range(0, M, batch_size), desc="k-NN search"):
        end = min(start + batch_size, M)
        scores, indices = index.search(matrix[start:end], k_actual)

        for local_i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            a_idx = start + local_i
            a_id = video_ids[a_idx]
            cap_a = captions.get(a_id, "")

            for score, b_idx in zip(score_row, idx_row):
                if b_idx == a_idx:
                    continue
                if not (sim_min <= float(score) <= sim_max):
                    continue

                b_id = video_ids[b_idx]
                key = (min(a_id, b_id), max(a_id, b_id))
                if key in seen:
                    continue
                seen.add(key)

                cap_b = captions.get(b_id, "")
                # filter out pairs whose captions are too similar or one is missing
                if not cap_a or not cap_b:
                    continue
                if 1.0 - _edit_ratio(cap_a, cap_b) < min_edit_ratio:
                    continue

                rows.append(
                    {
                        "id": pair_id,
                        "pth1": a_id,
                        "pth2": b_id,
                        "caption_a": cap_a,
                        "caption_b": cap_b,
                    }
                )
                pair_id += 1

    log.info("Found %d valid pairs after filtering", len(rows))

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pth1", "pth2", "caption_a", "caption_b"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Pairs written → %s", output_csv)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build k-NN pairs from V-JEPA embeddings")
    p.add_argument("--config", default="configs/build_pairs/openvid_dev.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    build_pairs(**cfg)

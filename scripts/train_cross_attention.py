"""Train CrossAttentionFusion + GalleryEncoder with InfoNCE loss."""

import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from covr.data.dataset import RetrievalDataset
from covr.models.cross_attention import (  # pylint: disable=no-name-in-module
    CrossAttentionFusion,
    FusionConfig,
    GalleryEncoder,
    InfoNCELoss,
)


def _clean_ids(ids: set[str], directory: Path) -> set[str]:
    """Return subset of ids whose .pt files contain no NaN values."""
    ok = set()
    for vid_id in ids:
        t = torch.load(directory / f"{vid_id}.pt", weights_only=True)
        if not torch.isnan(t).any():
            ok.add(vid_id)
    return ok


class TrainDataset(Dataset):
    """RetrievalDataset backed by pre-computed vjepa + flan embeddings."""

    def __init__(
        self,
        json_path: str | Path,
        embeddings_dir: str | Path,
        queries_dir: str | Path,
        split: str = "all",
        dev: bool = False,
        max_patches: int | None = None,
    ):
        full = RetrievalDataset(
            json_path, video_root="", split=split, load_frames=False
        )
        self.emb_dir = Path(embeddings_dir)
        self.query_dir = Path(queries_dir)
        self.max_patches = max_patches

        if dev:
            print(split)

            all_gallery = {p.stem for p in self.emb_dir.glob("*.pt")}
            available_gallery = _clean_ids(all_gallery, self.emb_dir)
            if len(available_gallery) < len(all_gallery):
                print(
                    f"[dev] filtered {len(all_gallery) - len(available_gallery)}",
                    "NaN gallery files",
                )
            available_queries = {p.stem for p in self.query_dir.glob("*.pt")}
            self.indices = [
                i
                for i in range(len(full))
                if full[i]["source_video_id"] in available_gallery
                and full[i]["target_video_id"] in available_gallery
                and full[i]["source_video_id"] in available_queries
            ]
            print(f"[dev] {len(self.indices)}/{len(full)} pairs available")
        else:  # change this
            self.indices = list(range(len(full)))

        self.inner = full

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.inner[self.indices[idx]]

        def _load_trunc(path: Path) -> torch.Tensor:
            t = torch.load(path, weights_only=True)
            if self.max_patches is not None and t.shape[0] > self.max_patches:
                t = t[: self.max_patches]
            return t

        return {
            "source_patches": _load_trunc(
                self.emb_dir / f"{item['source_video_id']}.pt"
            ),
            "target_patches": _load_trunc(
                self.emb_dir / f"{item['target_video_id']}.pt"
            ),
            "query_emb": torch.load(
                self.query_dir / f"{item['source_video_id']}.pt", weights_only=True
            ),
        }


def _pad_and_mask(tensors):
    B = len(tensors)
    N_max = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[1]
    padded = tensors[0].new_zeros(B, N_max, D)
    key_padding_mask = torch.ones(B, N_max, dtype=torch.bool)
    for i, t in enumerate(tensors):
        n = t.shape[0]
        padded[i, :n] = t
        key_padding_mask[i, :n] = False
    return padded, key_padding_mask


def _collate(samples: list[dict]) -> dict:
    src_padded, src_mask = _pad_and_mask([s["source_patches"] for s in samples])
    tgt_padded, tgt_mask = _pad_and_mask([s["target_patches"] for s in samples])
    query_padded, query_mask = _pad_and_mask([s["query_emb"] for s in samples])
    return {
        "source_patches": src_padded,  # [B, N_max, vid_dim]
        "source_mask": src_mask,  # [B, N_max]
        "target_patches": tgt_padded,  # [B, N_max', vid_dim]
        "target_mask": tgt_mask,  # [B, N_max']
        "query_emb": query_padded,  # [B, T_max, text_dim]
        "query_mask": query_mask,  # [B, T_max]
    }


def train(cfg: dict):
    device = torch.device(cfg["device"])

    fusion_cfg = FusionConfig(text_dim=cfg["text_dim"])
    fusion = CrossAttentionFusion(fusion_cfg).to(device)
    gallery_enc = GalleryEncoder(fusion_cfg).to(device)
    loss_fn = InfoNCELoss().to(device)

    optimizer = AdamW(
        [
            {"params": fusion.parameters()},
            {"params": gallery_enc.parameters()},
            {"params": loss_fn.parameters()},
        ],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["wd"]),
    )

    start_epoch = 0
    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=True)
        fusion.load_state_dict(ckpt["fusion"])
        gallery_enc.load_state_dict(ckpt["gallery_enc"])
        loss_fn.load_state_dict(ckpt["loss_fn"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"resumed from {cfg['resume']} (epoch {start_epoch})")

    dataset = TrainDataset(
        cfg["json_path"],
        cfg["embeddings_dir"],
        cfg["queries_dir"],
        split=cfg.get("split", "all"),
        dev=cfg["mode"] == "dev",
        max_patches=cfg.get("max_patches"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=_collate,
    )

    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg["epochs"]):
        fusion.train()
        gallery_enc.train()
        loss_fn.train()

        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            src = batch["source_patches"].to(device)  # [B, N,  vid_dim]
            tgt = batch["target_patches"].to(device)  # [B, N', vid_dim]
            text = batch["query_emb"].to(device)  # [B, T, text_dim]
            src_mask = batch["source_mask"].to(device)  # [B, N]
            tgt_mask = batch["target_mask"].to(device)  # [B, N']
            text_mask = batch["query_mask"].to(device)  # [B, T]

            query_embs = fusion(src, text, src_mask, text_mask)  # [B, embed_dim]
            target_embs = gallery_enc(tgt, tgt_mask)  # [B, embed_dim]

            loss = loss_fn(query_embs, target_embs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % cfg["log_every"] == 0:
                print(
                    f"epoch {epoch+1}/{cfg['epochs']}\t|\t"
                    f"step {step}/{len(loader)}\t|\t"
                    f"loss {loss.item():.4f}\t|\t"
                    f"τ {loss_fn.temperature.item():.4f}\t|\t"
                )

        avg = epoch_loss / len(loader)
        print(f"── epoch {epoch+1} avg loss: {avg:.4f}")

        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "fusion": fusion.state_dict(),
                    "gallery_enc": gallery_enc.state_dict(),
                    "loss_fn": loss_fn.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_dir / f"ckpt_epoch{epoch+1:04d}.pt",
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/fusion/cross_attention_dev.yaml")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()

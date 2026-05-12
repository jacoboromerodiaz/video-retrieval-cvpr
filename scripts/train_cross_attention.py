"""Train CrossAttentionFusion + GalleryEncoder with InfoNCE loss."""

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from covr.data.collate import collate_fn
from covr.data.dataset import TrainDataset, ValDataset
from covr.evaluation.metrics import recall_at_k  # pylint: disable=no-name-in-module
from covr.models.cross_attention import (  # pylint: disable=no-name-in-module
    CrossAttentionFusion,
    FusionConfig,
    GalleryEncoder,
    InfoNCELoss,
    HardNegativeNCELoss,
)


def get_loss_fn(loss_cfg):
    loss_fn_name = loss_cfg.get("name", "infonce")
    init_temperature = loss_cfg.get("init_temperature", 0.07)

    if loss_fn_name == "info-nce":
        return InfoNCELoss(init_temperature=init_temperature)
    if loss_fn_name == "hn-nce":
        return HardNegativeNCELoss(init_temperature=init_temperature)


def create_loaders(cfg):
    train_dataset = TrainDataset(
        cfg["csv_path"],
        cfg["embeddings_dir"],
        cfg["queries_dir"],
        dev=cfg["mode"] == "dev",
        max_patches=cfg.get("max_patches"),
    )

    val_dataset = ValDataset(
        cfg["val_json_path"],
        cfg["embeddings_dir"],
        cfg["queries_dir"],
        max_patches=cfg.get("max_patches"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def train(cfg: dict):
    train_cfg = cfg["train"]
    device = torch.device(train_cfg["device"])

    fusion_cfg = FusionConfig(**cfg["model"])
    fusion = CrossAttentionFusion(fusion_cfg).to(device)
    gallery_enc = GalleryEncoder(fusion_cfg).to(device)
    loss_fn = get_loss_fn(cfg["loss"]).to(device)

    max_lr = float(train_cfg["max_lr"])
    min_lr = float(train_cfg["min_lr"])

    optimizer = AdamW(
        [
            {"params": fusion.parameters()},
            {"params": gallery_enc.parameters()},
            {"params": loss_fn.parameters()},
        ],
        lr=max_lr,
        weight_decay=float(train_cfg["wd"]),
    )

    train_loader, val_loader = create_loaders(train_cfg)

    steps_per_epoch = len(train_loader)
    total_steps = train_cfg["epochs"] * steps_per_epoch
    warmup_steps = int(train_cfg.get("warmup_ratio", 0.0) * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + cosine * (max_lr - min_lr)) / max_lr

    scheduler = LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if train_cfg.get("resume", False):
        ckpt = torch.load(train_cfg["resume"], map_location=device, weights_only=True)
        fusion.load_state_dict(ckpt["fusion"])
        gallery_enc.load_state_dict(ckpt["gallery_enc"])
        loss_fn.load_state_dict(ckpt["loss_fn"])
        start_epoch = ckpt["epoch"]
        print(f"resumed from {train_cfg['resume']} (epoch {start_epoch})")

    ckpt_dir = Path(train_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    val_every = train_cfg.get("val_every", 1)

    for epoch in range(start_epoch, train_cfg["epochs"]):
        fusion.train()
        gallery_enc.train()
        loss_fn.train()

        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
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
            scheduler.step()

            epoch_loss += loss.item()
            if step % train_cfg["log_every"] == 0:
                print(
                    f"epoch {epoch+1}/{train_cfg['epochs']}\t|\t"
                    f"step {step}/{len(train_loader)}\t|\t"
                    f"loss {loss.item():.4f}\t|\t"
                    f"τ {loss_fn.temperature.item():.4f}\t|\t"
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )

        avg = epoch_loss / len(train_loader)
        print(f"── epoch {epoch+1} avg loss: {avg:.4f}")

        if (epoch + 1) % val_every == 0:
            fusion.eval()
            gallery_enc.eval()
            loss_fn.eval()
            val_loss = 0.0
            all_q, all_t = [], []
            with torch.no_grad():
                for batch in val_loader:
                    src = batch["source_patches"].to(device)
                    tgt = batch["target_patches"].to(device)
                    text = batch["query_emb"].to(device)
                    src_mask = batch["source_mask"].to(device)
                    tgt_mask = batch["target_mask"].to(device)
                    text_mask = batch["query_mask"].to(device)
                    query_embs = fusion(src, text, src_mask, text_mask)
                    target_embs = gallery_enc(tgt, tgt_mask)
                    val_loss += loss_fn(query_embs, target_embs).item()
                    all_q.append(query_embs.cpu())
                    all_t.append(target_embs.cpu())
            Q = torch.cat(all_q)  # [N_val, embed_dim]
            T = torch.cat(all_t)  # [N_val, embed_dim]
            scores = Q @ T.T  # [N_val, N_val]
            gt = torch.arange(len(Q))
            r1, r5, r10, r50 = recall_at_k(scores, gt, [1, 5, 10, 50])
            print(
                f"── epoch {epoch+1} val  loss: {val_loss / len(val_loader):.4f} │ "
                f"R@1={r1:.3f}  R@5={r5:.3f}  R@10={r10:.3f}  R@50={r50:.3f}"
            )

        if (epoch + 1) % train_cfg["save_every"] == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "fusion": fusion.state_dict(),
                    "gallery_enc": gallery_enc.state_dict(),
                    "loss_fn": loss_fn.state_dict(),
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

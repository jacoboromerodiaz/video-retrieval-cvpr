import argparse
import csv
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise video editor assistant. "
    "Given two video descriptions, write a single modification instruction "
    "that starts with an action verb and describes the visual changes needed "
    "to convert video A into video B. Be concise and specific."
)

_USER_TEMPLATE = (
    "Caption A: {caption_a}\n"
    "Caption B: {caption_b}\n\n"
    "Modification instruction:"
)


def _build_prompts(
    batch: list[dict],
    tokenizer,
    caption_a_col: str,
    caption_b_col: str,
) -> list[str]:
    prompts = []
    for row in batch:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    caption_a=row[caption_a_col],
                    caption_b=row[caption_b_col],
                ),
            },
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


@torch.inference_mode()
def generate_edits(
    pairs_csv: str | Path,
    output_csv: str | Path,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens: int = 256,
    batch_size: int = 8,
    caption_a_col: str = "caption_a",
    caption_b_col: str = "caption_b",
    mode: str = "prod",
) -> None:
    pairs_csv = Path(pairs_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(pairs_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d pairs from %s", len(rows), pairs_csv)

    device = "cuda" if mode == "prod" and torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    log.info("Loading model %s on %s", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    results: list[dict] = []

    for start in tqdm(range(0, len(rows), batch_size), desc="Generating edits"):
        batch = rows[start : start + batch_size]
        prompts = _build_prompts(batch, tokenizer, caption_a_col, caption_b_col)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = output_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        for row, text in zip(batch, decoded):
            edit = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
            results.append(
                {
                    "id": row["id"],
                    "pth1": row["pth1"],
                    "pth2": row["pth2"],
                    "txt1": row[caption_a_col],
                    "edit": edit,
                }
            )

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pth1", "pth2", "txt1", "edit"])
        writer.writeheader()
        writer.writerows(results)

    log.info("COVR-format CSV written → %s (%d rows)", output_csv, len(results))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate COVR modification texts with a local LLM")
    p.add_argument("--config", default="configs/generate_edits/openvid_dev.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    generate_edits(**cfg)

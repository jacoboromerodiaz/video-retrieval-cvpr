"""Download the CoVR-R dataset from Hugging Face."""

from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

# train set 1
snapshot_download(
    repo_id="kyrielw/Rich-Txt-Edit-CoVR",
    repo_type="dataset",
    local_dir="covr/data/rich-text-covr",
    ignore_patterns=["stage1.ckpt", "stage2.ckpt", "query_feat"],
)

# val and test set
snapshot_download(
    repo_id="orange-fox/CoVR-R",
    repo_type="dataset",
    local_dir="covr/data/val_test-covr-r",
)

subsets = [
    ("WebVid/8M/train", "webvid"),
    ("something_something_v2/20bn-something-something-v2", "ss2"),
]

local_dir = Path("covr/data/val_test-covr-r")
for src_rel, dst_name in subsets:
    src = local_dir / src_rel
    dst = local_dir / dst_name
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        shutil.move(str(item), dst / item.name)

shutil.rmtree(local_dir / "WebVid", ignore_errors=True)
shutil.rmtree(local_dir / "something_something_v2", ignore_errors=True)

shutil.move(
    "covr/data/val-set_with-targets.json", local_dir / "val-set_with-targets.json"
)

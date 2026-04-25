"""Download the CoVR-R dataset from Hugging Face."""

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="omkarthawakar/CoVR-R",
    repo_type="dataset",
    local_dir="src/data/covr-r",
)

#!/usr/bin/env python3
# downloader.py

from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

REPO_ID = "E10H1M/VibeVoice-Large"
DEST = Path("./VibeVoice-Large")

FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "configuration.json",
    "model-00001-of-00010.safetensors",
    "model-00002-of-00010.safetensors",
    "model-00003-of-00010.safetensors",
    "model-00004-of-00010.safetensors",
    "model-00005-of-00010.safetensors",
    "model-00006-of-00010.safetensors",
    "model-00007-of-00010.safetensors",
    "model-00008-of-00010.safetensors",
    "model-00009-of-00010.safetensors",
    "model-00010-of-00010.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
]

FOLDERS = [
    "figures",
]

def main():
    DEST.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")

    # expand folders -> files
    folder_files = []
    for d in FOLDERS:
        prefix = d.rstrip("/") + "/"
        folder_files.extend(sorted(p for p in repo_files if p.startswith(prefix)))

    wanted = FILES + folder_files
    total = len(wanted)

    for i, repo_path in enumerate(wanted, 1):
        out_path = DEST / repo_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{total}] {repo_path}")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="model",
            filename=repo_path,
            local_dir=str(DEST),
        )

if __name__ == "__main__":
    main()

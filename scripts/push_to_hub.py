from __future__ import annotations

import os
import subprocess
from pathlib import Path

def main() -> None:
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise SystemExit("Set HF_REPO_ID, for example cyrusmoazami/tabular-state-transformer")
    subprocess.run(["hf", "auth", "whoami"], check=True)
    subprocess.run(["hf", "upload", repo_id, str(Path.cwd()), "--exclude", ".git/*", "--exclude", "__pycache__/*", "--exclude", "*.pyc"], check=True)

if __name__ == "__main__":
    main()

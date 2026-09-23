"""Sequential, resumable bulk downloader for HF model/dataset repos.

Downloads run one at a time (parallelism was measured to give ~10% aggregate
throughput at best on this link - see docs/MODEL_AUDIT.md bandwidth note -
so there's no benefit to concurrent repo downloads, only added complexity).
Each call to snapshot_download is itself resumable via the HF cache, so
interrupting this script and rerunning it later picks up where it left off
without re-fetching completed files.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download

LOG_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "logs" / "bulk_download.jsonl"


def log(event: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = time.time()
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps(event))


def download_one(
    repo_id: str,
    revision: str = "main",
    repo_type: str = "model",
    allow_patterns: list[str] | None = None,
) -> None:
    log(
        {
            "event": "start",
            "repo_id": repo_id,
            "revision": revision,
            "repo_type": repo_type,
            "allow_patterns": allow_patterns,
        }
    )
    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            max_workers=4,
            # Many repos ship identical weights in several formats (PyTorch,
            # safetensors, TF, Flax, Rust); on a ~2.3 MB/s link fetching all
            # of them wastes hours, so callers can restrict to what's used.
            allow_patterns=allow_patterns,
        )
    except Exception as e:  # noqa: BLE001 - log and continue to next repo
        log({"event": "error", "repo_id": repo_id, "revision": revision, "error": str(e)})
        return
    log(
        {
            "event": "done",
            "repo_id": repo_id,
            "revision": revision,
            "path": path,
            "elapsed_seconds": time.time() - t0,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+", help="repo_id[:revision[:repo_type]] entries")
    ap.add_argument(
        "--allow-patterns",
        nargs="+",
        default=None,
        help="only fetch files matching these globs (e.g. '*.json' '*.safetensors' '*.txt')",
    )
    args = ap.parse_args()

    for entry in args.repos:
        parts = entry.split(":")
        repo_id = parts[0]
        revision = parts[1] if len(parts) > 1 and parts[1] else "main"
        repo_type = parts[2] if len(parts) > 2 and parts[2] else "model"
        download_one(repo_id, revision, repo_type, args.allow_patterns)


if __name__ == "__main__":
    main()

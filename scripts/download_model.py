"""Copy Qwen2.5-0.5B-Instruct into models/ (uses local HF cache first)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

LOCAL_SNAPSHOTS = [
    Path("/home/jpzhao_team/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"),
]


def copy_snapshot(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, symlinks=False)
        else:
            shutil.copy2(item.resolve() if item.is_symlink() else item, target)


def find_local_snapshot() -> Path | None:
    for root in LOCAL_SNAPSHOTS:
        if not root.is_dir():
            continue
        snaps = sorted(p for p in root.iterdir() if p.is_dir())
        for snap in snaps:
            if (snap / "config.json").exists() and (
                (snap / "model.safetensors").exists() or any(snap.glob("*.safetensors"))
            ):
                return snap
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    if (out / "config.json").exists() and any(out.glob("*.safetensors")):
        print(f"OK (already present): {out}")
        return

    local = find_local_snapshot()
    if local is not None:
        copy_snapshot(local, out)
        print(f"OK (copied snapshot {local}): {out}")
        return

    try:
        snapshot_download(args.repo, local_dir=str(out), local_files_only=True)
        print(f"OK (offline cache): {out}")
        return
    except Exception as exc:
        print(f"offline miss ({exc}); trying hub download")
    snapshot_download(args.repo, local_dir=str(out), local_files_only=False)
    print(f"OK (downloaded): {out}")


if __name__ == "__main__":
    main()

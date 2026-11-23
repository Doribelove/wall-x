"""Collect all model actions and the first and last inputs for a completed trial."""
import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path


def collect(work_dir, run_id):
    out = work_dir / "runs" / run_id
    queue = work_dir / "ipc"
    result = json.loads((out / "result.json").read_text())
    count = result["model_calls"]
    if result["status"] != "horizon_reached" or not count:
        raise ValueError("This recording did not finish its control horizon.")
    suffixes = (".action.npy", ".request", ".reply")
    for number in sorted({0, count - 1}):
        for suffix in (".npz", *suffixes):
            name = f"{number:03d}{suffix}"
            shutil.copyfile(queue / f"{run_id}-{name}", out / f"chunk_{name}")
    archive_path = out / "model_actions.tgz"
    temporary = archive_path.with_suffix(".part")
    with tarfile.open(temporary, "w:gz") as archive:
        for number in range(count):
            for suffix in suffixes:
                name = f"{number:03d}{suffix}"
                archive.add(queue / f"{run_id}-{name}", arcname=name)
    temporary.replace(archive_path)
    print(archive_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--work-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".runtime")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("run_id must contain letters, digits, underscores or hyphens")
    collect(args.work_dir.resolve(), args.run_id)

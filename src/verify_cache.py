"""
One-time repair tool for data/cache/.

If train.py ever gets interrupted mid-write (crash, disk full, power loss,
Ctrl+C at the wrong moment), it's possible for a single cached tensor file
to be left truncated/corrupt on disk while still existing at its expected
path. Since ChordSegmentDataset only checks os.path.exists(cache_path) to
decide whether a segment is "already cached," a corrupt-but-present file
gets silently treated as done -- and then blows up much later, mid-training,
when torch.load() actually tries to read it.

This script scans every cached file, tries to load it, and moves anything
that fails (plus any leftover *.pt.tmp files from an interrupted atomic
write) out of the way into _to_delete/corrupt_cache/ -- so the NEXT
train.py run correctly sees those specific segments as missing and
regenerates just those, instead of you needing to redo the whole cache.

Run from the repo root (same place you run train.py from):

    python src/verify_cache.py

This only ever MOVES files, never deletes -- consistent with the rest of
this project's cleanup approach. Nothing here needs librosa or your other
project modules, just torch, so it's safe/fast to run any time.
"""
import os
import shutil

import torch

CACHE_DIR = os.path.join("data", "cache")
QUARANTINE_DIR = os.path.join("_to_delete", "corrupt_cache")


def main():
    if not os.path.isdir(CACHE_DIR):
        print(f"No cache directory found at {CACHE_DIR!r} -- nothing to check.")
        return

    all_files = os.listdir(CACHE_DIR)
    real_cache_files = [f for f in all_files if f.endswith(".pt")]
    stray_tmp_files = [f for f in all_files if f.endswith(".pt.tmp")]

    print(f"Checking {len(real_cache_files)} cached tensor(s) in {CACHE_DIR}/...")
    if stray_tmp_files:
        print(f"Also found {len(stray_tmp_files)} leftover .tmp file(s) from an interrupted write.")

    corrupt = []
    for i, fname in enumerate(real_cache_files, 1):
        path = os.path.join(CACHE_DIR, fname)
        try:
            torch.load(path, weights_only=True)
        except Exception as e:
            corrupt.append((path, repr(e)))

        if i % 20000 == 0:
            print(f"  checked {i}/{len(real_cache_files)}...")

    if not corrupt and not stray_tmp_files:
        print("\nAll cached files loaded successfully -- nothing to fix. Safe to resume train.py.")
        return

    os.makedirs(QUARANTINE_DIR, exist_ok=True)

    for path, err in corrupt:
        dest = os.path.join(QUARANTINE_DIR, os.path.basename(path))
        shutil.move(path, dest)
        print(f"  CORRUPT -> moved: {os.path.basename(path)}   ({err})")

    for fname in stray_tmp_files:
        path = os.path.join(CACHE_DIR, fname)
        dest = os.path.join(QUARANTINE_DIR, fname)
        shutil.move(path, dest)
        print(f"  stray .tmp -> moved: {fname}")

    print(
        f"\nDone. {len(corrupt)} corrupt file(s) and {len(stray_tmp_files)} stray .tmp file(s) "
        f"moved to {QUARANTINE_DIR}/. Re-run train.py -- it will regenerate exactly those "
        f"segments and leave every other cached segment untouched."
    )


if __name__ == "__main__":
    main()

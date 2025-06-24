#!/usr/bin/env python3
import os
import re
import shutil
import argparse

def parse_t_from_filename(filename, prefix, suffix):
    """
    Extract the integer t from filenames of the form prefix{t}suffix.
    """
    pattern = re.escape(prefix) + r'(\d+)' + re.escape(suffix) + r'$'
    m = re.match(pattern, filename)
    return int(m.group(1)) if m else None

def extract_frame_ts(frames_dir):
    """
    Scan frames_dir for files named like '8.png', '42.png', etc.
    Return a set of their integer t values.
    """
    ts = set()
    for fname in os.listdir(frames_dir):
        m = re.match(r'^(\d+)\.png$', fname)
        if m:
            ts.add(int(m.group(1)))
    return ts

def main(frames_dir, gaussian_dir,
         gaussian_prefix='canonical_',
         gaussian_suffix='.pkl'):
    # 1) Determine start_t and end_t from frames_dir
    frame_ts = extract_frame_ts(frames_dir)
    if not frame_ts:
        print(f"Error: no files matching '{{t}}.png' in {frames_dir}")
        return
    start_t, end_t = min(frame_ts), max(frame_ts)
    print(f"Processing frames t = {start_t} … {end_t}")

    # 2) Gather existing gaussian timestamps
    existing_ts = set()
    for fname in os.listdir(gaussian_dir):
        t = parse_t_from_filename(fname, gaussian_prefix, gaussian_suffix)
        if t is not None:
            existing_ts.add(t)

    if not existing_ts:
        print(f"Error: no files matching {gaussian_prefix}{{t}}{gaussian_suffix} in {gaussian_dir}")
        return

    # 3) For each t in the full range, copy nearest if missing
    for t in range(start_t, end_t + 1):
        target_name = f"{gaussian_prefix}{t}{gaussian_suffix}"
        target_path = os.path.join(gaussian_dir, target_name)

        if t in existing_ts and os.path.exists(target_path):
            continue  # already present

        # find nearest existing frame
        nearest = min(existing_ts, key=lambda x: abs(x - t))
        src_name = f"{gaussian_prefix}{nearest}{gaussian_suffix}"
        src_path = os.path.join(gaussian_dir, src_name)

        if not os.path.exists(src_path):
            print(f"[Error] Reference file not found: {src_path}")
            continue

        if os.path.exists(target_path):
            os.remove(target_path)
            print(f"[Removed] {target_path}")

        shutil.copy2(src_path, target_path)
        print(f"[Copied] {src_name} → {target_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-fill missing canonical_{t}.pkl by nearest-copy, based on frames/{t}.png range"
    )
    parser.add_argument(
        "--frames_dir", required=True,
        help="Directory containing frame images named {t}.png"
    )
    parser.add_argument(
        "--gaussian_dir", required=True,
        help="Directory containing canonical_{t}.pkl files"
    )
    args = parser.parse_args()
    main(
        frames_dir=args.frames_dir,
        gaussian_dir=args.gaussian_dir
    )

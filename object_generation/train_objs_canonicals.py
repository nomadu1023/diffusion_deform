import os
import subprocess
import argparse
import cv2
import numpy as np

def process_object(object_name, args):
    mask_root = os.path.join(args.input_mask_dir, object_name)
    if not os.path.isdir(mask_root):
        print(f"[Warning] Mask directory not found for object '{object_name}', skipping.")
        return

    # collect all mask files under this object (recursive)
    mask_paths = []
    for root, _, files in os.walk(mask_root):
        for fname in sorted(files):
            if fname.lower().endswith('.png'):
                mask_paths.append(os.path.join(root, fname))

    if not mask_paths:
        print(f"[Info] No masks found for object '{object_name}'.")
        return

    # top-level output dirs per object
    out_sub = os.path.join(args.outdir, object_name)
    vis_sub = os.path.join(args.visdir, object_name)
    os.makedirs(out_sub, exist_ok=True)
    os.makedirs(vis_sub, exist_ok=True)

    # process every N-th mask according to stepsize
    for idx in range(0, len(mask_paths), args.stepsize):
        mask_path = mask_paths[idx]

        # load the mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[Warning] Failed to load mask: {mask_path}, skipping.")
            continue

        # skip completely black masks, but print their names
        if not np.any(mask):
            print(f"[Skipped] Empty mask (all black): {mask_path}")
            continue

        # derive frame ID from zero-padded mask filename
        mask_fname = os.path.basename(mask_path)           # e.g. '00110.png'
        mask_id    = os.path.splitext(mask_fname)[0]       # '00110'
        frame_id   = str(int(mask_id))                     # '110'
        image_fname = f"{frame_id}.png"
        image_path  = os.path.join(args.input_dir, image_fname)
        if not os.path.isfile(image_path):
            print(f"[Missing] No matching image for mask {mask_path}, expected {image_path}")
            continue

        # build save name
        save_name = f"{args.save_prefix}_{object_name}_{mask_id}_{idx}"

        cmd = [
            "python", "main_Sampling.py",
            "--config", args.config,
            f"input={image_path}",
            f"input_mask={mask_path}",
            f"outdir={out_sub}",
            f"visdir={vis_sub}",
            f"save_path={save_name}",
            f"iters={args.iters}"
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch sampling for each object using frame images and masks")
    parser.add_argument("--config",         type=str, default="configs/image.yaml",
                        help="Path to sampling config file")
    parser.add_argument("--input_dir",      type=str, required=True,
                        help="Directory of frame images named like '110.png' (no leading zeros)")
    parser.add_argument("--input_mask_dir", type=str, required=True,
                        help="Top-level directory containing object subfolders with zero-padded mask PNGs")
    parser.add_argument("--outdir",         type=str, default="./gaussians",
                        help="Output directory for gaussian samples")
    parser.add_argument("--visdir",         type=str, default="./vis",
                        help="Output directory for visualizations")
    parser.add_argument("--save_prefix",    type=str, default="result",
                        help="Prefix for output filenames")
    parser.add_argument("--stepsize",       type=int, default=1,
                        help="Process every N-th mask/image pair")
    parser.add_argument("--iters",          type=int, default=1000,
                        help="Number of iterations for sampling")
    args = parser.parse_args()

    for object_name in sorted(os.listdir(args.input_mask_dir)):
        process_object(object_name, args)

if __name__ == "__main__":
    main()

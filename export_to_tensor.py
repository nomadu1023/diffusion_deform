#!/usr/bin/env python3
import os
import pickle
import argparse
import re

import numpy as np
import torch

def _to_tensor(x):
    """
    NumPy ndarray, Python 리스트/스칼라, torch.Tensor 모두
    torch.Tensor 로 변환합니다. 이미 Tensor 면 그대로 리턴.
    """
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asanyarray(x))


def _load_data(path: str):
    """
    pickle.load 로 시도, 실패하면 torch.load 로 재시도합니다.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        # torch.save 로 저장된 경우
        return torch.load(path)


def load_gaussians_xyzrgb(pkl_dir: str) -> torch.Tensor:
    """
    Args:
        pkl_dir: directory containing files named global_gaussian_{t}.pkl
    Returns:
        Tensor of shape (N, T, 6): [x,y,z,R,G,B] per Gaussian per frame.
    """
    pattern = re.compile(r"global_gaussian_(\d+)\.pkl$")
    frames = []
    for fname in sorted(
        os.listdir(pkl_dir),
        key=lambda n: int(pattern.match(n).group(1)) if pattern.match(n) else -1
    ):
        if not pattern.match(fname):
            continue
        full_path = os.path.join(pkl_dir, fname)
        data = _load_data(full_path)

        mean = _to_tensor(data["mean_3d"])  # (N,3)
        rgb  = _to_tensor(data["rgb"])      # (N,3)
        frames.append(torch.cat([mean, rgb], dim=1).unsqueeze(1))  # (N,1,6)

    if not frames:
        raise RuntimeError(f"No global_gaussian_*.pkl found in {pkl_dir}")
    return torch.cat(frames, dim=1)  # (N,T,6)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pkl_dir", required=True,
        help="world/ folder with global_gaussian_{t}.pkl files"
    )
    p.add_argument(
        "--out_prefix", required=True,
        help="where to write combined: file path or directory"
    )
    args = p.parse_args()

    combined = load_gaussians_xyzrgb(args.pkl_dir)

    # out_prefix가 디렉토리인지 확인
    prefix = args.out_prefix
    if os.path.isdir(prefix) or prefix.endswith(os.sep):
        out_dir = prefix.rstrip(os.sep)
        base = os.path.basename(out_dir)
        out_dir = out_dir
    else:
        out_dir = os.path.dirname(prefix) or '.'
        base = os.path.basename(prefix)

    # 디렉토리 없으면 생성
    os.makedirs(out_dir, exist_ok=True)

    pt_path = os.path.join(out_dir, base + ".pt")
    pkl_path = os.path.join(out_dir, base + ".pkl")

    # 1) save as .pt
    torch.save(combined, pt_path)
    # 2) save as NumPy-pkl
    with open(pkl_path, "wb") as f:
        pickle.dump(combined.cpu().numpy(), f)

    print(f"[export_to_tensor] saved {pt_path} and {pkl_path}")

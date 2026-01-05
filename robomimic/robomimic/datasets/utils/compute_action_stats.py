#!/usr/bin/env python
import argparse
import json
from collections import OrderedDict

import h5py
import numpy as np


def _decode_keys(keys):
    decoded = []
    for key in keys:
        if isinstance(key, (bytes, np.bytes_)):
            decoded.append(key.decode("utf-8"))
        else:
            decoded.append(str(key))
    return decoded


def _demo_index(demo_key):
    try:
        return int(demo_key.split("_")[1])
    except Exception as exc:
        raise ValueError(f"Unexpected demo key format: {demo_key}") from exc


def _select_demos(h5_file, filter_key=None, num_demo=None):
    if filter_key:
        mask_key = f"mask/{filter_key}"
        if mask_key not in h5_file:
            raise KeyError(f"Missing {mask_key} in HDF5 file")
        demos = _decode_keys(h5_file[mask_key][:])
    else:
        demos = list(h5_file["data"].keys())

    demos = sorted(demos, key=_demo_index)
    if num_demo is not None:
        demos = [demo for demo in demos if _demo_index(demo) < num_demo]

    if not demos:
        raise ValueError("No demos selected; check filter_key and num_demo.")

    return demos


def _compute_traj_stats(traj_dict):
    traj_stats = {k: {} for k in traj_dict}
    for key, values in traj_dict.items():
        traj_stats[key]["n"] = values.shape[0]
        traj_stats[key]["mean"] = values.mean(axis=0, keepdims=True)
        traj_stats[key]["sqdiff"] = ((values - traj_stats[key]["mean"]) ** 2).sum(axis=0, keepdims=True)
        traj_stats[key]["min"] = values.min(axis=0, keepdims=True)
        traj_stats[key]["max"] = values.max(axis=0, keepdims=True)
    return traj_stats


def _aggregate_traj_stats(stats_a, stats_b):
    merged = {}
    for key in stats_a:
        n_a, avg_a, m2_a, min_a, max_a = (
            stats_a[key]["n"],
            stats_a[key]["mean"],
            stats_a[key]["sqdiff"],
            stats_a[key]["min"],
            stats_a[key]["max"],
        )
        n_b, avg_b, m2_b, min_b, max_b = (
            stats_b[key]["n"],
            stats_b[key]["mean"],
            stats_b[key]["sqdiff"],
            stats_b[key]["min"],
            stats_b[key]["max"],
        )
        n = n_a + n_b
        mean = (n_a * avg_a + n_b * avg_b) / n
        delta = avg_b - avg_a
        m2 = m2_a + m2_b + (delta ** 2) * (n_a * n_b) / n
        merged[key] = {
            "n": n,
            "mean": mean,
            "sqdiff": m2,
            "min": np.minimum(min_a, min_b),
            "max": np.maximum(max_a, max_b),
        }
    return merged


def _compute_action_stats(h5_file, demos, action_keys):
    merged_stats = None
    for demo in demos:
        traj = {}
        for key in action_keys:
            traj[key] = h5_file[f"data/{demo}/{key}"][()].astype(np.float32)
        traj_stats = _compute_traj_stats(traj)
        merged_stats = traj_stats if merged_stats is None else _aggregate_traj_stats(merged_stats, traj_stats)
    return merged_stats


def _action_stats_to_normalization_stats(action_stats, normalization):
    normalization_stats = OrderedDict()
    for action_key, stats in action_stats.items():
        if normalization == "none":
            normalization_stats[action_key] = {
                "scale": np.ones_like(stats["mean"], dtype=np.float32),
                "offset": np.zeros_like(stats["mean"], dtype=np.float32),
            }
        elif normalization == "min_max":
            range_eps = 1e-6
            input_min = stats["min"].astype(np.float32)
            input_max = stats["max"].astype(np.float32)
            output_min = -0.999999
            output_max = 0.999999

            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min

            scale = input_range / (output_max - output_min)
            offset = input_min - scale * output_min
            offset[ignore_dim] = input_min[ignore_dim] - (output_max + output_min) / 2

            normalization_stats[action_key] = {"scale": scale, "offset": offset}
        elif normalization == "gaussian":
            input_mean = stats["mean"].astype(np.float32)
            input_std = np.sqrt(stats["sqdiff"] / stats["n"]).astype(np.float32)
            std_eps = 1e-6
            ignore_dim = input_std < std_eps
            input_std[ignore_dim] = 1.0

            normalization_stats[action_key] = {"scale": input_mean, "offset": input_std}
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

    return normalization_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute action normalization stats and dump action_stats.json from an HDF5 dataset."
    )
    parser.add_argument("--hdf5_path", required=True, help="Path to the input HDF5 file")
    parser.add_argument("--output_path", required=True, help="Path to the output action_stats.json")
    parser.add_argument(
        "--action_key",
        action="append",
        default=["action_absolute"],
        help="Action dataset key under data/<demo>/ (repeatable)",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "min_max", "gaussian"],
        default="min_max",
        help="Normalization mode to compute scale/offset",
    )
    parser.add_argument(
        "--filter_key",
        default=None,
        help="Optional mask key (e.g., train or valid) to select demos",
    )
    parser.add_argument(
        "--num_demo",
        type=int,
        default=None,
        help="Optional cap on demo indices (keeps demo_i where i < num_demo)",
    )
    args = parser.parse_args()

    with h5py.File(args.hdf5_path, "r") as h5_file:
        demos = _select_demos(h5_file, filter_key=args.filter_key, num_demo=args.num_demo)
        action_stats = _compute_action_stats(h5_file, demos, args.action_key)

    normalization_stats = _action_stats_to_normalization_stats(action_stats, args.normalization)
    output = OrderedDict()
    for key, stats in normalization_stats.items():
        output[key] = {"scale": stats["scale"].tolist(), "offset": stats["offset"].tolist()}

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {args.output_path} using {len(demos)} demos.")


if __name__ == "__main__":
    main()

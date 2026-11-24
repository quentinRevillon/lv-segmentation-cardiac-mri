"""Evaluate the LV segmentation on a folder of ACDC cases.

Usage:
    python evaluate.py data/testing --phase ED
    python evaluate.py data/training --phase ES --output results_training_ES.csv
    python evaluate.py data/testing --phase ED --limit 3     # quick check on 3 cases

Prints, per normalised slice position (0 = base, 9 = apex) and overall:
the percentage of seeds inside the reference LV cavity, the slice-wise (2D) Dice
and the volume-wise (3D) Dice. ``--output`` saves one row per slice as CSV.
"""

import argparse
from pathlib import Path

import pandas as pd

from lv_segmentation.data import LV_LABEL, load_phase
from lv_segmentation.metrics import dice, seed_in_mask
from lv_segmentation.pipeline import initial_seed, layer_index, segment_volume


def evaluate_case(case_dir, phase):
    """Segment one case and return a DataFrame with one row per slice (2D Dice, seed hit, 3D Dice)."""
    ed, _ = load_phase(case_dir, "ED")
    es, _ = load_phase(case_dir, "ES")
    volume, label = load_phase(case_dir, phase)
    target = label == LV_LABEL
    mask, seeds = segment_volume(volume, initial_seed(ed, es))
    n_slices = volume.shape[2]
    return pd.DataFrame({
        "case": Path(case_dir).name,
        "slice": range(n_slices),
        "layer": [layer_index(z, n_slices) for z in range(n_slices)],
        "dice_2d": [dice(mask[:, :, z], target[:, :, z]) for z in range(n_slices)],
        "seed_in_lv": [seed_in_mask(seeds[z], target[:, :, z]) for z in range(n_slices)],
        "dice_3d": dice(mask, target),
    })


def summarize_layers(results):
    """Return per-layer slice counts, seed hit rate (%) and 2D Dice mean and std (%)."""
    grouped = results.groupby("layer")
    return pd.DataFrame({
        "slices": grouped.size(),
        "seed_in_lv_%": grouped["seed_in_lv"].mean() * 100,
        "dice_2d_%": grouped["dice_2d"].mean() * 100,
        "dice_2d_std_%": grouped["dice_2d"].std(ddof=0) * 100,
    }).round(1)


def summarize(results):
    """Return overall seed hit rate, 2D Dice mean and std, and mean 3D Dice over cases (%)."""
    per_case = results.groupby("case")["dice_3d"].first()
    return pd.Series({
        "cases": len(per_case),
        "seed_in_lv_%": results["seed_in_lv"].mean() * 100,
        "dice_2d_%": results["dice_2d"].mean() * 100,
        "dice_2d_std_%": results["dice_2d"].std(ddof=0) * 100,
        "dice_3d_%": per_case.mean() * 100,
    }).round(1)


def main():
    """Parse arguments, evaluate every case of the folder and print the summaries."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="folder of ACDC case directories, e.g. data/testing")
    parser.add_argument("--phase", choices=["ED", "ES"], default="ED", help="cardiac phase to segment")
    parser.add_argument("--limit", type=int, help="evaluate only the first N cases")
    parser.add_argument("--output", help="optional CSV with one row per slice")
    args = parser.parse_args()

    cases = sorted(Path(args.data_dir).glob("patient*"))[:args.limit]
    results = pd.concat([evaluate_case(case, args.phase) for case in cases], ignore_index=True)
    print(summarize_layers(results).to_string())
    print(summarize(results).to_string())
    if args.output:
        results.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

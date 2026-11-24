"""Segment the left-ventricle cavity of one ACDC case and save it as NIfTI.

Usage:
    python segment.py data/testing/patient101 --phase ED --output patient101_ED_lv.nii.gz
    python segment.py data/testing/patient101 --phase ES --output patient101_ES_lv.nii.gz --figure patient101_ES.png

The output mask has the geometry of the input frame (1 = LV cavity, 0 = background).
``--figure`` saves the mid slice with the predicted (red) and reference (blue) contours.
"""

import argparse

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from lv_segmentation.data import LV_LABEL, load_phase, normalize, phase_path
from lv_segmentation.pipeline import segment_case


def save_mask(mask, reference_path, output_path):
    """Save a boolean mask as uint8 NIfTI with the affine and header of ``reference_path``."""
    reference = nib.load(reference_path)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header), output_path)


def save_mid_slice_figure(image, mask, target, output_path):
    """Save the mid slice of ``image`` with predicted (red) and reference (blue) contours."""
    z = image.shape[2] // 2
    plt.figure(figsize=(5, 5))
    plt.imshow(normalize(image[:, :, z]), cmap="gray")
    plt.contour(mask[:, :, z], colors="r", levels=[0.5])
    plt.contour(target[:, :, z], colors="b", levels=[0.5])
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)


def main():
    """Parse arguments, segment the case and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case_dir", help="ACDC case directory, e.g. data/testing/patient101")
    parser.add_argument("--phase", choices=["ED", "ES"], default="ED", help="cardiac phase to segment")
    parser.add_argument("--output", required=True, help="output NIfTI mask (.nii.gz)")
    parser.add_argument("--figure", help="optional PNG of the mid slice with contours")
    args = parser.parse_args()

    mask, _ = segment_case(args.case_dir, args.phase)
    save_mask(mask, phase_path(args.case_dir, args.phase), args.output)
    if args.figure:
        image, label = load_phase(args.case_dir, args.phase)
        save_mid_slice_figure(image, mask, label == LV_LABEL, args.figure)


if __name__ == "__main__":
    main()

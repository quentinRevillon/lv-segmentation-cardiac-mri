"""Load ACDC cases and normalise intensities.

A case directory follows the ACDC layout (``patient042/Info.cfg``,
``patient042/patient042_frame01.nii.gz``, ``..._frame01_gt.nii.gz``...).
Volumes are returned as ``(H, W, Z)`` arrays.

Usage:
    from lv_segmentation.data import load_phase, normalize
    image, label = load_phase("data/training/patient042", "ED")
    image01 = normalize(image)
"""

from pathlib import Path

import nibabel as nib
import numpy as np

LV_LABEL = 3  # ACDC labels: 1 right ventricle, 2 myocardium, 3 left ventricle cavity


def read_info(case_dir):
    """Parse ``Info.cfg`` of a case into a ``{key: value}`` dict of strings."""
    lines = (Path(case_dir) / "Info.cfg").read_text().splitlines()
    return dict(line.split(": ", 1) for line in lines if line.strip())


def phase_path(case_dir, phase, gt=False):
    """Return the NIfTI path of phase ``"ED"`` or ``"ES"`` (``gt=True`` for its label map)."""
    case_dir = Path(case_dir)
    frame = int(read_info(case_dir)[phase])
    suffix = "_gt" if gt else ""
    return case_dir / f"{case_dir.name}_frame{frame:02d}{suffix}.nii.gz"


def load_phase(case_dir, phase):
    """Load the image and label map of phase ``"ED"`` or ``"ES"`` as float ``(H, W, Z)`` arrays."""
    image = nib.load(phase_path(case_dir, phase)).get_fdata()
    label = nib.load(phase_path(case_dir, phase, gt=True)).get_fdata()
    return image, label


def normalize(image, q_min=0.01, q_max=0.99):
    """Rescale ``image`` to [0, 1] between its ``q_min`` and ``q_max`` quantiles, then clip.

    A constant image (quantile range below 1e-5) is mapped to zeros.
    """
    low, high = np.quantile(image, [q_min, q_max])
    if abs(high - low) < 1e-5:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)


def to_uint8(image):
    """Normalise ``image`` between its 0.1 and 99.9 percentiles and cast it to uint8."""
    return np.uint8(normalize(image, 0.001, 0.999) * 255)

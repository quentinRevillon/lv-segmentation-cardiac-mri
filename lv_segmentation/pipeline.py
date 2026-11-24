"""Segment the left-ventricle cavity of a whole ED or ES volume.

1. Detect one Hough seed per slice on the ED/ES motion map.
2. On the ED mid slice, grow a region from every seed (``grow_to_size``) and keep the
   most circular one; its centre of mass is the initial seed.
3. Segment the mid slice of the requested phase with ``grow_until_jump``.
4. Propagate slice by slice towards the base and the apex, seeding each slice with the
   centre of mass of the previous mask.

Usage:
    from lv_segmentation.pipeline import segment_case
    mask, seeds = segment_case("data/testing/patient101", "ED")  # mask: bool (H, W, Z)
"""

import numpy as np

from lv_segmentation.data import load_phase
from lv_segmentation.growing import centroid, circularity, grow_to_size, grow_until_jump
from lv_segmentation.seed import detect_lv_centers

# Maximum LV area (pixels) per normalised slice position, from base (0) to apex (9).
MAX_SIZES = (2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 1000, 1000)
BACK_OFF = 10  # tolerance steps kept before the area jump
MIN_CANDIDATE_SIZE = 200  # regions smaller than this (pixels) are not candidates


def layer_index(z, n_slices):
    """Map slice ``z`` of an ``n_slices`` volume to a position 0-9 (base to apex)."""
    return round(z / (n_slices - 1) * 9)


def most_circular_region(image, seeds):
    """Grow a region from each seed with ``grow_to_size`` and return the most circular one."""
    masks = [grow_to_size(image, seed) for seed in seeds]
    masks = [mask for mask in masks if mask.sum() > MIN_CANDIDATE_SIZE]
    return max(masks, key=circularity)


def initial_seed(ed, es):
    """Return the (x, y) seed of the mid slice, located on the ED mid slice where the cavity is largest."""
    seeds = detect_lv_centers(ed, es)
    return centroid(most_circular_region(ed[:, :, ed.shape[2] // 2], seeds))


def segment_slice(volume, z, seed):
    """Segment slice ``z`` of ``volume`` from ``seed`` with the slice-dependent area cap."""
    max_size = MAX_SIZES[layer_index(z, volume.shape[2])]
    return grow_until_jump(volume[:, :, z], seed, max_size, BACK_OFF)


def propagate(volume, mask, slices):
    """Segment ``slices`` in order, seeding each one with the centroid of the previous mask."""
    masks, seeds = {}, {}
    for z in slices:
        seeds[z] = centroid(mask)
        mask = segment_slice(volume, z, seeds[z])
        masks[z] = mask
    return masks, seeds


def segment_volume(volume, seed):
    """Segment every slice of ``volume`` from a seed on its mid slice.

    Returns a bool mask of shape ``(H, W, Z)`` and the ``(Z, 2)`` array of (x, y) seeds.
    """
    n_slices = volume.shape[2]
    mid = n_slices // 2
    mid_mask = segment_slice(volume, mid, seed)
    masks, seeds = {mid: mid_mask}, {mid: np.asarray(seed)}
    for slices in (range(mid + 1, n_slices), range(mid - 1, -1, -1)):
        new_masks, new_seeds = propagate(volume, mid_mask, slices)
        masks.update(new_masks)
        seeds.update(new_seeds)
    mask = np.stack([masks[z] for z in range(n_slices)], axis=-1)
    return mask, np.array([seeds[z] for z in range(n_slices)])


def segment_case(case_dir, phase):
    """Segment the LV cavity of phase ``"ED"`` or ``"ES"`` of an ACDC case directory."""
    ed, _ = load_phase(case_dir, "ED")
    es, _ = load_phase(case_dir, "ES")
    volume = {"ED": ed, "ES": es}[phase]
    return segment_volume(volume, initial_seed(ed, es))

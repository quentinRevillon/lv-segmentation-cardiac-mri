"""Tolerance-based region growing from a seed point.

A region is the convex hull of the flood fill whose intensity stays within
``tolerance * I(seed)`` of the seed intensity. Two stopping rules are provided:

- ``grow_to_size``: raise the tolerance until the region exceeds a fixed area;
- ``grow_until_jump``: raise the tolerance, record the region area, and stop a few
  steps before the largest jump in area, i.e. before the region leaks through the
  endocardial border into the myocardium or the right ventricle.

Seeds are (x, y) pixel coordinates; images are 2D slices.

Usage:
    from lv_segmentation.growing import grow_until_jump, circularity, centroid
    mask = grow_until_jump(image_slice, seed=(104, 120), max_size=2000, back_off=10)
    print(mask.sum(), circularity(mask), centroid(mask))
"""

import numpy as np
from skimage.measure import perimeter
from skimage.morphology import convex_hull_image
from skimage.segmentation import flood

from lv_segmentation.data import to_uint8

STOP_SIZE = 3000  # area (pixels) at which grow_until_jump stops raising the tolerance
TOLERANCE_STEP = 0.01


def brightest_neighbor(image, seed, half_window=5):
    """Return the (x, y) position of the brightest pixel in the square window centred on ``seed``."""
    x, y = int(seed[0]), int(seed[1])
    window = image[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
    dy, dx = np.unravel_index(np.argmax(window), window.shape)
    return x - half_window + int(dx), y - half_window + int(dy)


def grow(image, seed, tolerance):
    """Return the convex hull of the flood fill from ``seed`` within ``tolerance * I(seed)``."""
    x, y = seed
    mask = flood(image, (y, x), tolerance=image[y, x] * tolerance)
    return convex_hull_image(mask)


def grow_to_size(image, seed, max_size=1600, tolerance=0.3):
    """Raise the tolerance until the region reaches ``max_size`` pixels, then step back by 0.04."""
    image = to_uint8(image)
    seed = brightest_neighbor(image, seed)
    mask = grow(image, seed, tolerance)
    while mask.sum() < max_size:
        tolerance += TOLERANCE_STEP
        mask = grow(image, seed, tolerance)
    return grow(image, seed, tolerance - 0.04)


def size_curve(image, seed, max_size, tolerance=0.1):
    """Raise the tolerance until the region area reaches ``STOP_SIZE``.

    Returns the list of tolerances, the list of areas (capped at ``max_size`` after the
    first step) and the region obtained at the last tolerance.
    """
    mask = grow(image, seed, tolerance)
    tolerances, sizes = [tolerance], [mask.sum()]
    while mask.sum() < STOP_SIZE:
        tolerance += TOLERANCE_STEP
        mask = grow(image, seed, tolerance)
        tolerances.append(tolerance)
        sizes.append(min(mask.sum(), max_size))
    return tolerances, sizes, mask


def find_jump(sizes):
    """Return the index of the largest area increase within the second half of ``sizes``."""
    half = len(sizes) // 2
    return half + int(np.argmax(np.diff(sizes[half:])))


def grow_until_jump(image, seed, max_size, back_off):
    """Grow the region and keep the one ``back_off`` tolerance steps before the largest area jump."""
    image = to_uint8(image)
    seed = brightest_neighbor(image, seed)
    tolerances, sizes, last_mask = size_curve(image, seed, max_size)
    if len(sizes) - len(sizes) // 2 < 2:
        return last_mask
    return grow(image, seed, tolerances[max(find_jump(sizes) - back_off, 0)])


def circularity(mask):
    """Return the isoperimetric ratio 4*pi*area / perimeter**2 (1 for a disk)."""
    return 4 * np.pi * mask.sum() / perimeter(mask) ** 2


def centroid(mask):
    """Return the (x, y) centre of mass of a binary mask as uint16."""
    rows, cols = np.nonzero(mask)
    return np.array((cols.mean(), rows.mean())).astype(np.uint16)

"""Locate the left-ventricle centre with a Hough circle transform on a motion map.

The LV cavity is the most circular structure of a short-axis slice and its walls move
the most between end-diastole (ED) and end-systole (ES). The seed is therefore the
centre of the first Hough circle found on the edges of the ED/ES difference image.

Usage:
    from lv_segmentation.data import load_phase
    from lv_segmentation.seed import detect_lv_center, detect_lv_centers
    ed, _ = load_phase("data/training/patient042", "ED")
    es, _ = load_phase("data/training/patient042", "ES")
    x, y = detect_lv_center(ed[:, :, 5], es[:, :, 5])
    centers = detect_lv_centers(ed, es)  # one (x, y) per slice, slice 0 excluded
"""

import cv2
import numpy as np

from lv_segmentation.data import normalize, to_uint8


def motion_map(ed_slice, es_slice):
    """Return the uint8 absolute difference between the normalised ED and ES slices."""
    ed = normalize(ed_slice, 0.001, 0.999) * 255
    es = normalize(es_slice, 0.001, 0.999) * 255
    return np.uint8(np.abs(ed - es))


def clean_motion_map(diff):
    """Remove speckle (3x3 opening), fill the moving walls (7x7 closing) and renormalise."""
    opened = cv2.morphologyEx(diff, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return to_uint8(closed)


def radius_range(width):
    """Return the (min, max) Hough radius in pixels: (15, 30) for images wider than 400 px, else (8, 15)."""
    return (15, 30) if width > 400 else (8, 15)


def detect_lv_center(ed_slice, es_slice):
    """Return the (x, y) centre of the first Hough circle on the cleaned motion map.

    Falls back to the image centre when no circle is detected.
    """
    diff = motion_map(ed_slice, es_slice)
    edges = cv2.Canny(clean_motion_map(diff), 500, 200)
    r_min, r_max = radius_range(diff.shape[1])
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=1, param2=1,
                               minRadius=r_min, maxRadius=r_max)
    if circles is None:
        return diff.shape[1] // 2, diff.shape[0] // 2
    return int(circles[0][0][0]), int(circles[0][0][1])


def detect_lv_centers(ed, es):
    """Return an ``(Z - 1, 2)`` array of (x, y) centres for slices 1 to Z-1 of the ED/ES volumes."""
    return np.array([detect_lv_center(ed[:, :, z], es[:, :, z]) for z in range(1, ed.shape[2])])

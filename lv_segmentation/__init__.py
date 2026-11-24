"""Left-ventricle cavity segmentation in cine MRI by Hough seed detection and region growing.

Modules:
    data      load ACDC cases, normalise intensities
    seed      Hough-circle seed detection on the ED/ES motion map
    growing   tolerance-based region growing and shape measures
    pipeline  whole-volume segmentation by slice-to-slice propagation
    metrics   Dice coefficient and seed localisation

Usage:
    from lv_segmentation.pipeline import segment_case
    mask, seeds = segment_case("data/testing/patient101", "ED")
"""

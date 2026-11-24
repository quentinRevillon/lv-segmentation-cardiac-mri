"""Evaluation metrics for LV cavity masks.

Usage:
    from lv_segmentation.metrics import dice, seed_in_mask
    score = dice(pred_mask, label == 3)      # 2D slice or 3D volume
    hit = seed_in_mask((104, 120), label[:, :, 5] == 3)
"""


def dice(pred, target):
    """Return the Dice coefficient 2|A n B| / (|A| + |B|) between two boolean arrays."""
    return 2 * (pred & target).sum() / (pred.sum() + target.sum())


def seed_in_mask(seed, target):
    """Return True if the (x, y) seed falls inside the boolean 2D ``target`` mask."""
    x, y = int(seed[0]), int(seed[1])
    return bool(x < target.shape[1] and y < target.shape[0] and target[y, x])

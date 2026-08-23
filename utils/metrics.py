"""Binary segmentation metrics used by training and evaluation."""

import numpy as np


def dice_coefficient(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return 1.0
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    return float(2 * intersection / volume_sum)


def intersection_over_union(mask_gt: np.ndarray,
                            mask_pred: np.ndarray) -> float:
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    return float(intersection / union) if union else 1.0


def pixel_accuracy(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    return float(np.mean(mask_gt == mask_pred))


def mean_absolute_error(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    return float(
        np.mean(np.abs(mask_gt.astype(float) - mask_pred.astype(float))))

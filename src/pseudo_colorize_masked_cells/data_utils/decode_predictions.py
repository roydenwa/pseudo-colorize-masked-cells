import cv2
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from scipy import ndimage as ndi


def basic_labeling(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.uint8)
    return ndi.label(img)[0].astype(np.uint16)


def pool_nms(heatmap: Tensor, pool_size: int = 3) -> Tensor:
    pad = (pool_size - 1) // 2
    hm_max = F.max_pool2d(heatmap, pool_size, stride=1, padding=pad)
    keep = (hm_max == heatmap).float()

    return heatmap * keep


def pool_blobs(heatmap: Tensor, pool_size: int = 3, thresh: float = 0.75) -> list:
    heatmap = pool_nms(heatmap, pool_size).cpu().numpy()[0]
    centroid_blobs = np.where(
        heatmap > thresh,
        1,
        0,
    )
    centroid_blobs = basic_labeling(centroid_blobs)
    blob_ids = np.unique(centroid_blobs)
    centroids = []

    # Skip 0 = background
    for blob_id in blob_ids[1:]:
        blob = (centroid_blobs == blob_id).astype(np.uint8).copy()
        moments = cv2.moments(blob)

        if moments["m00"] != 0.0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            centroids.append((centroid_x, centroid_y))

    return centroids


def centroids2bboxes(
    heatmap: Tensor, centroid_dim_blocks: Tensor, thresh: float = 0.75
) -> list:
    centroids = pool_blobs(heatmap[None, ...], thresh=thresh)
    bboxes = []

    for centroid in centroids:
        centroid_x, centroid_y = centroid
        h = centroid_dim_blocks[..., 0][centroid_y, centroid_x]
        w = centroid_dim_blocks[..., 1][centroid_y, centroid_x]
        bboxes.append(
            (
                centroid_x - w // 2,
                centroid_y - h // 2,
                w,
                h,
            )
        )
    return bboxes

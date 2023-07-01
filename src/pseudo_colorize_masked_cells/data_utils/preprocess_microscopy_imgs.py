import cv2
import numpy as np
import matplotlib.pyplot as plt


def pseudo_colorize_img(img, cmap_func, dtype=np.float32):
    return cmap_func(img)[..., 0:3].astype(dtype)  # RGBA -> RGB


def mask2bboxes(mask: np.ndarray) -> list:
    """Bbox format: x_min, y_min, w, h"""
    labels = np.unique(mask)
    bboxes = []

    # Skip label 0 (background):
    for label in labels[1:]:
        label_mask = mask == label
        bbox = cv2.boundingRect(label_mask.astype(np.uint8))
        bboxes.append(bbox)

    return bboxes


def bboxes2heatmap(bboxes: list, heatmap_shape: tuple, blur=True) -> np.ndarray:
    heatmap = np.zeros(heatmap_shape)
    img_h, img_w = heatmap_shape[0], heatmap_shape[1]

    for bbox in bboxes:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        tmp_heatmap = np.zeros(heatmap_shape)

        cv2.ellipse(
            img=tmp_heatmap,
            center=(x + w // 2, y + h // 2),
            axes=(int(w // 2.5), int(h // 2.5)),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
        )

        if blur and int(w // 1.5) and int(h // 1.5):
            blur_y1 = y - h if y - h > 0 else 0
            blur_y2 = y + 2 * h if y + 2 * h < img_h else img_h
            blur_x1 = x - w if x - w > 0 else 0
            blur_x2 = x + 2 * w if x + 2 * w < img_w else img_w

            tmp_heatmap[blur_y1:blur_y2, blur_x1:blur_x2] = cv2.blur(
                src=tmp_heatmap[blur_y1:blur_y2, blur_x1:blur_x2],
                ksize=(int(w // 1.5), int(h // 1.5)),
            )
        heatmap += tmp_heatmap

    return heatmap


def bboxes2centroid_dim_blocks(bboxes: list, img_shape: tuple, pad: int = 1) -> np.ndarray:
    img_h, img_w = img_shape[0], img_shape[1]
    dim_y_blocks = np.zeros(img_shape)
    dim_x_blocks = np.zeros(img_shape)

    for bbox in bboxes:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(
            img=dim_y_blocks,
            pt1=(x + w // 2 - pad, y + h // 2 - pad),
            pt2=(x + w // 2 + pad, y + h // 2 + pad),
            color=h,
            thickness=-1,
        )
        cv2.rectangle(
            img=dim_x_blocks,
            pt1=(x + w // 2 - pad, y + h // 2 - pad),
            pt2=(x + w // 2 + pad, y + h // 2 + pad),
            color=w,
            thickness=-1,
        )
    centroid_dim_blocks = np.dstack((dim_y_blocks, dim_x_blocks))

    return centroid_dim_blocks


def plot_bboxes(cell_img, bboxes, thickness=2):
    cell_img = np.dstack((cell_img, cell_img, cell_img))

    for bbox in bboxes:
        x_min, y_min, w, h = bbox
        x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
        cv2.rectangle(
            img=cell_img,
            pt1=(x_min, y_min),
            pt2=(x_min + w, y_min + h),
            color=(0, 1.0, 0),
            thickness=thickness,
        )

    return cell_img

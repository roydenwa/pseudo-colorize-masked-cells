import torch
import pandas as pd

from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .decode_predictions import centroids2bboxes


def run_coco_bbox_eval(model, dm, thresh=0.75, device="cuda"):
    mean_ap = MeanAveragePrecision(box_format="xywh")
    dm.setup(stage="predict")
    model.to(device)
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dm.predict_dataloader()):
            pred_heatmap, pred_dims, *_ = model(batch["img"].to(device))

            if model.rel_dims:
                pred_bboxes = centroids2bboxes(
                    heatmap=pred_heatmap[0][0],
                    centroid_dim_blocks=pred_dims.view(-1, 384, 384, 2)[0] * model.image_size,
                    thresh=thresh,
                )
            else:
                pred_bboxes = centroids2bboxes(
                    heatmap=pred_heatmap[0][0],
                    centroid_dim_blocks=pred_dims.view(-1, 384, 384, 2)[0],
                    thresh=thresh,
                )
            target_bboxes = centroids2bboxes(
                heatmap=batch["heatmap"][0],
                centroid_dim_blocks=batch["centroid_dim_blocks"][0],
                thresh=thresh,
            )

            preds.append({
                "boxes": torch.tensor(pred_bboxes, device=device),
                "scores": torch.ones(len(pred_bboxes), device=device),
                "labels": torch.ones(len(pred_bboxes), device=device),
            })
            targets.append({
                "boxes": torch.tensor(target_bboxes, device=device),
                "labels": torch.ones(len(target_bboxes), device=device),
            })

    mean_ap.update(preds, targets)

    return pd.DataFrame(dict(mean_ap.compute()), index=[0])
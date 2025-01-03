import numpy as np
import torch
import torchvision


def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  # xmin
    y1 = bboxes[:, 1]  # ymin
    x2 = bboxes[:, 2]  # xmax
    y2 = bboxes[:, 3]  # ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms_class_agnostic_tensor(
    scores: torch.Tensor,
    labels: torch.Tensor,
    bboxes: torch.Tensor,
    nms_thresh: float
) -> tuple[torch.Tensor]:
    keep = torchvision.ops.nms(bboxes, scores, nms_thresh)
    return scores[keep], labels[keep], bboxes[keep]


def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms_class_aware_tensor(
    scores: torch.Tensor,
    labels: torch.Tensor,
    bboxes: torch.Tensor,
    nms_thresh: float,
    num_classes: int
) -> tuple[torch.Tensor]:
    """Multiclass NMS aware

    Args:
        scores (torch.Tensor): a tensor [N,] containing prediction scores  
        labels (torch.Tensor): a tensor [N, C] , C is num classes
        bboxes (torch.Tensor): a tensor [N, 4] containing bounding boxes
        nms_thresh (float): nms threshold
        num_classes (int): number of classes

    Returns:
        Tuple[torch.Tensor]: output (scores, labels, bboxes) after nms
    """
    keep = torch.zeros(bboxes.size(0), dtype=torch.int32)
    for i in range(num_classes):
        inds = torch.where(labels == i)[0]
        # if inds.size(0) == 0:
        #     continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = torchvision.ops.nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = torch.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


def multiclass_nms_tensor(
    scores: torch.Tensor,
    labels: torch.Tensor,
    bboxes: torch.Tensor,
    nms_thresh: float,
    num_classes: int,
    class_agnostic: bool = False
) -> tuple[torch.Tensor]:
    if class_agnostic:
        return multiclass_nms_class_agnostic_tensor(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware_tensor(scores, labels, bboxes, nms_thresh, num_classes)

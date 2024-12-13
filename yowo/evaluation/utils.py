import numpy as np
import torch


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Compute the IoU (Intersection over Union) between two bounding boxes using PyTorch tensors.

    Args:
        box1 (torch.Tensor): Tensor of shape (4,), representing the first box.
        box2 (torch.Tensor): Tensor of shape (4,), representing the second box.
        x1y1x2y2 (bool): If True, boxes are in (x1, y1, x2, y2) format. If False, boxes are in (cx, cy, w, h) format.

    Returns:
        float: IoU value between the two boxes.
    """
    if x1y1x2y2:
        mx = torch.min(box1[0], box2[0])
        Mx = torch.max(box1[2], box2[2])
        my = torch.min(box1[1], box2[1])
        My = torch.max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = torch.min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = torch.max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = torch.min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = torch.max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh

    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea

    return carea / uarea


def np_bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(float(box1[0]-box1[2]/2.0), float(box2[0]-box2[2]/2.0))
        Mx = max(float(box1[0]+box1[2]/2.0), float(box2[0]+box2[2]/2.0))
        my = min(float(box1[1]-box1[3]/2.0), float(box2[1]-box2[3]/2.0))
        My = max(float(box1[1]+box1[3]/2.0), float(box2[1]+box2[3]/2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea


def area2d(b):
    return (b[:, 2]-b[:, 0]+1)*(b[:, 3]-b[:, 1]+1)


def overlap2d(b1, b2):
    """
    Compute the 2D overlap between two sets of bounding boxes using PyTorch tensors.

    Args:
        b1 (torch.Tensor): Tensor of shape (N, 4), representing the first set of boxes.
        b2 (torch.Tensor): Tensor of shape (N, 4), representing the second set of boxes.

    Returns:
        torch.Tensor: Tensor of shape (N,), representing the overlap areas.
    """
    xmin = torch.maximum(b1[:, 0], b2[:, 0])
    xmax = torch.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    width = torch.maximum(torch.tensor(0.0), xmax - xmin)

    ymin = torch.maximum(b1[:, 1], b2[:, 1])
    ymax = torch.minimum(b1[:, 3] + 1, b2[:, 3] + 1)
    height = torch.maximum(torch.tensor(0.0), ymax - ymin)

    return width * height


def np_overlap2d(b1, b2):
    xmin = np.maximum(b1[:, 0], b2[:, 0])
    xmax = np.minimum(b1[:, 2]+1, b2[:, 2]+1)
    width = np.maximum(0, xmax-xmin)
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    ymax = np.minimum(b1[:, 3]+1, b2[:, 3]+1)
    height = np.maximum(0, ymax-ymin)
    return width*height


def nms_3d(detections, overlap=0.5):
    # detections: [(tube1, score1), (tube2, score2)]
    if len(detections) == 0:
        return torch.tensor([], dtype=torch.int32)
    I = torch.argsort(torch.tensor([d[1] for d in detections]))
    indices = torch.zeros(I.size()[0], dtype=torch.int32)
    counter = 0
    while I.size()[0] > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = torch.tensor([iou3dt(detections[ii][0], detections[i][0])
                             for ii in I[:-1]])
        I = I[torch.where(ious <= overlap)[0]]
    return indices[:counter]


def np_nms_3d(detections, overlap=0.5):
    # detections: [(tube1, score1), (tube2, score2)]
    if len(detections) == 0:
        return np.array([], dtype=np.int32)
    I = np.argsort([d[1] for d in detections])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0
    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([np_iou3dt(detections[ii][0], detections[i][0])
                        for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]
    return indices[:counter]


def iou3d(b1, b2):
    """
    Compute the 3D IoU (Intersection over Union) between two sets of bounding boxes.

    Args:
        b1 (torch.Tensor): Tensor of shape (N, 5), where the first column is batch index, and the next 4 are (x1, y1, x2, y2).
        b2 (torch.Tensor): Tensor of shape (N, 5), same format as b1.

    Returns:
        torch.Tensor: The mean IoU over all batches.
    """
    assert b1.shape[0] == b2.shape[0]
    assert torch.all(b1[:, 0] == b2[:, 0])
    o = overlap2d(b1[:, 1:5], b2[:, 1:5])
    areas = area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - o
    return torch.mean(o / areas)


def iou3dt(b1: torch.Tensor, b2: torch.Tensor):
    """
    Compute the 3D IoU (Intersection over Union) between two sets of 3D bounding boxes, 
    considering both spatial and temporal dimensions.

    Args:
        b1 (torch.Tensor): Tensor of shape (N, 5), where the first column is time (t),
                           and the next 4 columns are (x1, y1, x2, y2).
        b2 (torch.Tensor): Tensor of shape (N, 5), same format as b1.

    Returns:
        torch.Tensor: The 3D IoU considering both spatial and temporal overlap.
    """
    # Extract the time range
    tmin = torch.max(b1[0, 0], b2[0, 0]).to(b1.device)
    tmax = torch.min(b1[-1, 0], b2[-1, 0]).to(b1.device)

    # If there's no overlap in the time dimension, return 0
    if tmax <= tmin:
        return torch.tensor(0.0).to(b1.device)

    # Calculate the temporal overlap and union
    temporal_inter = tmax - tmin + 1
    temporal_union = torch.max(
        b1[-1, 0], b2[-1, 0]) - torch.min(b1[0, 0], b2[0, 0]) + 1

    # Slice the tensors to the time range [tmin, tmax]
    b1_t = b1[(b1[:, 0] >= tmin) & (b1[:, 0] <= tmax)]
    b2_t = b2[(b2[:, 0] >= tmin) & (b2[:, 0] <= tmax)]

    # Compute spatial IoU using the previously implemented iou3d
    spatial_iou = iou3d(b1_t, b2_t)

    # Return the adjusted IoU, considering both spatial and temporal factors
    return spatial_iou * (temporal_inter / temporal_union)


def np_iou3d(b1, b2):
    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])
    o = np_overlap2d(b1[:, 1:5], b2[:, 1:5])
    return np.mean(o/(area2d(b1[:, 1:5])+area2d(b2[:, 1:5])-o))


def np_iou3dt(b1, b2):
    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])
    if tmax <= tmin:
        return 0.0
    temporal_inter = tmax-tmin+1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1
    return np_iou3d(b1[np.where(b1[:, 0] == tmin)[0][0]:np.where(b1[:, 0] == tmax)[0][0]+1, :], b2[np.where(b2[:, 0] == tmin)[0][0]:np.where(b2[:, 0] == tmax)[0][0]+1, :]) * temporal_inter / temporal_union


def voc_ap(pr, use_07_metric=False):
    """
    Compute VOC AP given precision and recall in PyTorch.
    If use_07_metric is true, uses the VOC 07 11 point method (default: False).

    Args:
        pr (torch.Tensor): Tensor of shape (N, 2), where the first column is precision and the second is recall.
        use_07_metric (bool): Whether to use the VOC 07 11-point method (default: False).

    Returns:
        float: The computed Average Precision (AP).
    """
    rec, prec = pr[:, 1], pr[:, 0]

    if use_07_metric:
        # 11-point metric
        ap = 0.
        for t in torch.arange(0., 1.1, 0.1):
            if torch.sum(rec >= t) == 0:
                p = 0
            else:
                p = torch.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # Correct AP calculation
        # First append sentinel values at the end
        mrec = torch.cat((torch.tensor([0]), rec, torch.tensor([1])))
        mpre = torch.cat((torch.tensor([0]), prec, torch.tensor([0])))

        # Compute the precision envelope
        for i in range(mpre.size(0) - 1, 0, -1):
            mpre[i - 1] = torch.max(mpre[i - 1], mpre[i])

        # To calculate area under PR curve, look for points where recall changes value
        i = torch.where(mrec[1:] != mrec[:-1])[0]

        # Sum (\Delta recall) * precision
        ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap  # Return as a Python float


def np_voc_ap(pr, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    rec, prec = pr[:, 1], pr[:, 0]
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

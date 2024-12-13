# -*- coding:utf-8 -*-
import numpy as np
import os

import torch
from .utils import bbox_iou, nms_3d, iou3d, iou3dt, iou3d, iou3dt, np_bbox_iou, np_iou3d, np_iou3dt, np_nms_3d, np_voc_ap, voc_ap


def compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    """
    Compute scores for bounding boxes between two sets of boxes.
    Args:
        bbox1 (torch.Tensor): Bounding boxes for frame 1 (N, 5), where each row is [x1, y1, x2, y2, score]
        bbox2 (torch.Tensor): Bounding boxes for frame 2 (M, 5)
        w_iou (float): Weight for IoU score
        w_scores (float): Weight for sum of class scores
        w_scores_mul (float): Weight for product of class scores

    Returns:
        torch.Tensor: Score matrix of shape (N, M)
    """
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]

    scores = torch.zeros((n_bbox1, n_bbox2), dtype=torch.float32)
    for i in range(n_bbox1):
        box1 = bbox1[i, :4]
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            # Assuming bbox_iou is implemented for PyTorch tensors
            # You need to implement bbox_iou for PyTorch
            bbox_iou_frames = bbox_iou(box1, box2, x1y1x2y2=True)
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * \
                sum_score_frames + w_scores_mul * mul_score_frames

    return scores


def np_compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx: <x1> <y1> <x2> <y2> <class score>
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]
    # for saving all possible scores between each two bbxes in successive frames
    scores = np.zeros([n_bbox1, n_bbox2], dtype=np.float32)
    for i in range(n_bbox1):
        box1 = bbox1[i, :4]
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            bbox_iou_frames = np_bbox_iou(box1, box2, x1y1x2y2=True)
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * \
                sum_score_frames + w_scores_mul * mul_score_frames

    return scores


def link_bbxes_between_frames(bbox_list, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    """
    Link bounding boxes between frames using PyTorch.
    Args:
        bbox_list (list of torch.Tensor): List of bounding boxes for each frame (N, 5), where each row is [x1, y1, x2, y2, score]
        w_iou (float): Weight for IoU score
        w_scores (float): Weight for sum of class scores
        w_scores_mul (float): Weight for product of class scores

    Returns:
        list: Linked bounding boxes
    """
    nfr = len(bbox_list)

    # Check for non-empty frames
    ind_notempty = [i for i in range(nfr) if bbox_list[i].size(0) > 0]

    if not ind_notempty:
        return []
    elif len(ind_notempty) != nfr:
        for i in range(nfr):
            if bbox_list[i].size(0) == 0:
                # Copy the nearest detections to fill in the missing frames
                ind_dis = torch.abs(torch.tensor(ind_notempty) - i)
                nn = torch.argmin(ind_dis)
                bbox_list[i] = bbox_list[ind_notempty[nn]]

    detect = bbox_list
    nframes = len(detect)
    res = []

    isempty_vertex = torch.zeros(nframes, dtype=torch.bool)
    edge_scores = [compute_score_one_class(
        detect[i], detect[i + 1], w_iou=w_iou, w_scores=w_scores, w_scores_mul=w_scores_mul) for i in range(nframes - 1)]
    copy_edge_scores = edge_scores

    while not torch.any(isempty_vertex):
        scores = [torch.zeros(d.shape[0], dtype=torch.float32) for d in detect]
        index = [torch.full((d.shape[0],), float(
            'nan'), dtype=torch.float32) for d in detect]

        # Viterbi algorithm
        for i in range(nframes - 2, -1, -1):
            edge_score = edge_scores[i] + scores[i + 1]
            scores[i] = torch.max(edge_score, dim=1)[0]
            index[i] = torch.argmax(edge_score, dim=1)

        idx = -torch.ones(nframes, dtype=torch.int32)
        idx[0] = torch.argmax(scores[0])
        for i in range(nframes - 1):
            idx[i + 1] = index[i][idx[i]]

        this = torch.empty((nframes, 6), dtype=torch.float32)
        this[:, 0] = 1 + torch.arange(nframes, dtype=torch.float32)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i < nframes - 1:
                iouscore = copy_edge_scores[i][j, idx[i + 1]] - \
                    bbox_list[i][j, 4] - bbox_list[i + 1][idx[i + 1], 4]

            if i < nframes - 1:
                edge_scores[i] = torch.cat(
                    (edge_scores[i][:j, :], edge_scores[i][j + 1:, :]), dim=0)
            if i > 0:
                edge_scores[i - 1] = torch.cat(
                    (edge_scores[i - 1][:, :j], edge_scores[i - 1][:, j + 1:]), dim=1)
            this[i, 1:5] = detect[i][j, :4]
            this[i, 5] = detect[i][j, 4]
            detect[i] = torch.cat(
                (detect[i][:j, :], detect[i][j + 1:, :]), dim=0)
            isempty_vertex[i] = (detect[i].size(0) == 0)
        res.append(this)
        if len(res) == 3:
            break

    return res


def np_link_bbxes_between_frames(bbox_list, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx_list: list of bounding boxes <x1> <y1> <x2> <y2> <class score>
    # check no empty detections
    ind_notempty = []
    nfr = len(bbox_list)
    for i in range(nfr):
        if np.array(bbox_list[i]).size:
            ind_notempty.append(i)
    # no detections at all
    if not ind_notempty:
        return []
    # miss some frames
    elif len(ind_notempty) != nfr:
        for i in range(nfr):
            if not np.array(bbox_list[i]).size:
                # copy the nearest detections to fill in the missing frames
                ind_dis = np.abs(np.array(ind_notempty) - i)
                nn = np.argmin(ind_dis)
                bbox_list[i] = bbox_list[ind_notempty[nn]]

    detect = bbox_list
    nframes = len(detect)
    res = []

    isempty_vertex = np.zeros([nframes,], dtype=np.bool)
    edge_scores = [np_compute_score_one_class(
        detect[i], detect[i+1], w_iou=w_iou, w_scores=w_scores, w_scores_mul=w_scores_mul) for i in range(nframes-1)]
    copy_edge_scores = edge_scores

    while not np.any(isempty_vertex):
        # initialize
        scores = [np.zeros([d.shape[0],], dtype=np.float32) for d in detect]
        index = [np.nan*np.ones([d.shape[0],], dtype=np.float32)
                 for d in detect]
        # viterbi
        # from the second last frame back
        for i in range(nframes-2, -1, -1):
            edge_score = edge_scores[i] + scores[i+1]
            # find the maximum score for each bbox in the i-th frame and the corresponding index
            scores[i] = np.max(edge_score, axis=1)
            index[i] = np.argmax(edge_score, axis=1)
        # decode
        idx = -np.ones([nframes], dtype=np.int32)
        idx[0] = np.argmax(scores[0])
        for i in range(0, nframes-1):
            idx[i+1] = index[i][idx[i]]
        # remove covered boxes and build output structures
        this = np.empty((nframes, 6), dtype=np.float32)
        this[:, 0] = 1 + np.arange(nframes)
        for i in range(nframes):
            j = idx[i]
            iouscore = 0
            if i < nframes-1:
                iouscore = copy_edge_scores[i][j, idx[i+1]] - \
                    bbox_list[i][j, 4] - bbox_list[i+1][idx[i+1], 4]

            if i < nframes-1:
                edge_scores[i] = np.delete(edge_scores[i], j, 0)
            if i > 0:
                edge_scores[i-1] = np.delete(edge_scores[i-1], j, 1)
            this[i, 1:5] = detect[i][j, :4]
            this[i, 5] = detect[i][j, 4]
            detect[i] = np.delete(detect[i], j, 0)
            # it is true when there is no detection in any frame
            isempty_vertex[i] = (detect[i].size == 0)
        res.append(this)
        if len(res) == 3:
            break

    return res


def link_video_one_class(vid_det, bNMS3d=False):
    '''
    linking for one class in a video (in full length)
    vid_det: a list of [frame_index, [bbox cls_score]]
    gtlen: the mean length of gt in training set
    return a list of tube [array[frame_index, x1,y1,x2,y2, cls_score]]
    '''
    # list of bbox information [[bbox in frame 1], [bbox in frame 2], ...]
    vdets = [vid_det[i][1] for i in range(len(vid_det))]
    vres = link_bbxes_between_frames(vdets)
    if len(vres) != 0:
        if bNMS3d:
            tube = [b[:, :5] for b in vres]
            # compute score for each tube
            tube_scores = [torch.mean(b[:, 5]) for b in vres]
            dets = [(tube[t], tube_scores[t]) for t in range(len(tube))]
            # nms for tubes
            keep = nms_3d(dets, 0.3)  # bug for nms3dt
            if keep.size()[0]:
                vres_keep = [vres[k] for k in keep]
                # max subarray with penalization -|Lc-L|/Lc
                vres = vres_keep

    return vres


def np_link_video_one_class(vid_det, bNMS3d=False):
    '''
    linking for one class in a video (in full length)
    vid_det: a list of [frame_index, [bbox cls_score]]
    gtlen: the mean length of gt in training set
    return a list of tube [array[frame_index, x1,y1,x2,y2, cls_score]]
    '''
    # list of bbox information [[bbox in frame 1], [bbox in frame 2], ...]
    vdets = [vid_det[i][1] for i in range(len(vid_det))]
    vres = np_link_bbxes_between_frames(vdets)
    if len(vres) != 0:
        if bNMS3d:
            tube = [b[:, :5] for b in vres]
            # compute score for each tube
            tube_scores = [np.mean(b[:, 5]) for b in vres]
            dets = [(tube[t], tube_scores[t]) for t in range(len(tube))]
            # nms for tubes
            keep = np_nms_3d(dets, 0.3)  # bug for nms3dt
            if np.array(keep).size:
                vres_keep = [vres[k] for k in keep]
                # max subarray with penalization -|Lc-L|/Lc
                vres = vres_keep

    return vres


def video_ap_one_class(gt, pred_videos, device, iou_thresh=0.2, bTemporal=False):
    '''
    gt: [ video_index, tensor[frame_index, x1,y1,x2,y2] ]
    pred_videos: [ video_index, [ [frame_index, [[x1,y1,x2,y2, score]] ] ] ]
    '''
    # link for prediction
    pred = []
    for pred_v in pred_videos:
        video_index = pred_v[0]
        # [tensor<frame_index, x1,y1,x2,y2, cls_score>]
        pred_link_v = link_video_one_class(pred_v[1], True)
        for tube in pred_link_v:
            pred.append((video_index, tube))

    # sort tubes according to scores (descending order)
    argsort_scores = torch.argsort(-torch.tensor(
        [torch.mean(b[:, 5]) for _, b in pred]))
    # precision, recall
    pr = torch.empty((len(pred)+1, 2), dtype=torch.float32).to(device)
    pr[0, 0] = 1.0
    pr[0, 1] = 0.0
    fn = len(gt)  # sum([len(a[1]) for a in gt])
    fp = 0
    tp = 0

    gt_v_index = [g[0] for g in gt]
    for i, k in enumerate(argsort_scores):
        # if i % 100 == 0:
        #     print ("%6.2f%% boxes processed, %d positives found, %d remain" %(100*float(i)/argsort_scores.size, tp, fn))
        video_index, boxes = pred[k]
        ispositive = False
        if video_index in gt_v_index:
            gt_this_index, gt_this = [], []
            for j, g in enumerate(gt):
                if g[0] == video_index:
                    gt_this.append(g[1])
                    gt_this_index.append(j)
            if len(gt_this) > 0:
                if bTemporal:
                    iou = torch.tensor([iou3dt(g, boxes[:, :5]).item()
                                       for g in gt_this]).to(device)
                    # print(iou)
                else:
                    if boxes.shape[0] > gt_this[0].shape[0]:
                        # in case some frame don't have gt
                        iou = torch.tensor(
                            [iou3d(g, boxes[int(g[0, 0]-1):int(g[-1, 0]), :5]) for g in gt_this]).to(device)
                    elif boxes.shape[0] < gt_this[0].shape[0]:
                        # in flow case
                        iou = torch.tensor(
                            [iou3d(g[int(boxes[0, 0]-1):int(boxes[-1, 0]), :], boxes[:, :5]) for g in gt_this]).to(device)
                    else:
                        iou = torch.tensor([iou3d(g, boxes[:, :5])
                                            for g in gt_this]).to(device)

                if iou.size()[0] > 0:  # on ucf101 if invalid annotation ....
                    argmax = torch.argmax(iou)
                    if iou[argmax] >= iou_thresh:
                        ispositive = True
                        del gt[gt_this_index[argmax]]
        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1
        pr[i+1, 0] = float(tp)/float(tp+fp)
        pr[i+1, 1] = float(tp)/float(tp+fn + 0.00001)
    ap = voc_ap(pr)

    return ap


def np_video_ap_one_class(gt, pred_videos, iou_thresh=0.2, bTemporal=False):
    '''
    gt: [ video_index, array[frame_index, x1,y1,x2,y2] ]
    pred_videos: [ video_index, [ [frame_index, [[x1,y1,x2,y2, score]] ] ] ]
    '''
    # link for prediction
    pred = []
    for pred_v in pred_videos:
        video_index = pred_v[0]
        # [array<frame_index, x1,y1,x2,y2, cls_score>]
        pred_link_v = np_link_video_one_class(pred_v[1], True)
        for tube in pred_link_v:
            pred.append((video_index, tube))

    # sort tubes according to scores (descending order)
    argsort_scores = np.argsort(-np.array([np.mean(b[:, 5]) for _, b in pred]))
    pr = np.empty((len(pred)+1, 2), dtype=np.float32)  # precision, recall
    pr[0, 0] = 1.0
    pr[0, 1] = 0.0
    fn = len(gt)  # sum([len(a[1]) for a in gt])
    fp = 0
    tp = 0

    gt_v_index = [g[0] for g in gt]
    for i, k in enumerate(argsort_scores):
        # if i % 100 == 0:
        #     print ("%6.2f%% boxes processed, %d positives found, %d remain" %(100*float(i)/argsort_scores.size, tp, fn))
        video_index, boxes = pred[k]
        ispositive = False
        if video_index in gt_v_index:
            gt_this_index, gt_this = [], []
            for j, g in enumerate(gt):
                if g[0] == video_index:
                    gt_this.append(g[1])
                    gt_this_index.append(j)
            if len(gt_this) > 0:
                if bTemporal:
                    iou = np.array([np_iou3dt(np.array(g), boxes[:, :5])
                                   for g in gt_this])
                else:
                    if boxes.shape[0] > gt_this[0].shape[0]:
                        # in case some frame don't have gt
                        iou = np.array(
                            [np_iou3d(g, boxes[int(g[0, 0]-1):int(g[-1, 0]), :5]) for g in gt_this])
                    elif boxes.shape[0] < gt_this[0].shape[0]:
                        # in flow case
                        iou = np.array(
                            [np_iou3d(g[int(boxes[0, 0]-1):int(boxes[-1, 0]), :], boxes[:, :5]) for g in gt_this])
                    else:
                        iou = np.array([np_iou3d(g, boxes[:, :5])
                                       for g in gt_this])

                if iou.size > 0:  # on ucf101 if invalid annotation ....
                    argmax = np.argmax(iou)
                    if iou[argmax] >= iou_thresh:
                        ispositive = True
                        del gt[gt_this_index[argmax]]
        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1
        pr[i+1, 0] = float(tp)/float(tp+fp)
        pr[i+1, 1] = float(tp)/float(tp+fn + 0.00001)
    ap = np_voc_ap(pr)

    return ap


def gt_to_videts(gt_v):
    # return  [label, video_index, [[frame_index, x1,y1,x2,y2], [], []] ]
    keys = list(gt_v.keys())
    keys.sort()
    res = []
    for i in range(len(keys)):
        # annotation of the video: tubes and gt_classes
        v_annot = gt_v[keys[i]]
        for j in range(len(v_annot['tubes'])):
            res.append([v_annot['gt_classes'], i+1, v_annot['tubes'][j]])
    return res


def evaluate_videoAP(gt_videos, all_boxes, num_classes, device, iou_thresh=0.2, bTemporal=False):
    '''
    gt_videos: {vname:{tubes: [[frame_index, x1,y1,x2,y2]], gt_classes: vlabel}} 
    all_boxes: {imgname:{cls_ind:array[x1,y1,x2,y2, cls_score]}}
    '''
    def imagebox_to_videts(img_boxes, num_classes):
        # image names
        keys = list(all_boxes.keys())
        keys.sort()
        res = []
        # without 'background'
        for cls_ind in range(num_classes):
            v_cnt = 1
            frame_index = 1
            v_dets = []
            cls_ind += 1
            # get the directory path of images
            preVideo = os.path.dirname(keys[0])
            for i in range(len(keys)):
                curVideo = os.path.dirname(keys[i])
                img_cls_dets = img_boxes[keys[i]][cls_ind]
                v_dets.append([frame_index, img_cls_dets])
                frame_index += 1
                if preVideo != curVideo:
                    preVideo = curVideo
                    frame_index = 1
                    # tmp_dets = v_dets[-1]
                    del v_dets[-1]
                    res.append([cls_ind, v_cnt, v_dets])
                    v_cnt += 1
                    v_dets = []
                    # v_dets.append(tmp_dets)
                    v_dets.append([frame_index, img_cls_dets])
                    frame_index += 1
            # the last video
            # print('num of videos:{}'.format(v_cnt))
            res.append([cls_ind, v_cnt, v_dets])
        return res

    gt_videos_format = gt_to_videts(gt_videos)
    pred_videos_format = imagebox_to_videts(all_boxes, num_classes)
    ap_all = []
    for cls_ind in range(num_classes):
        cls_ind += 1
        # [ video_index, [[frame_index, x1,y1,x2,y2]] ]
        gt = [g[1:] for g in gt_videos_format if g[0] == cls_ind]
        pred_cls = [p[1:] for p in pred_videos_format if p[0] == cls_ind]
        ap = video_ap_one_class(gt, pred_cls, iou_thresh,
                                bTemporal, device=device)
        ap_all.append(ap)

    return torch.stack(ap_all).mean()

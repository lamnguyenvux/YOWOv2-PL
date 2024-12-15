import numpy as np
import torch
from scipy.io import loadmat
from torchmetrics import Metric

from yowo.evaluation.calculate_video_map import evaluate_videoAP


class VideoMeanAveragePrecision(Metric):
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: list[float],
        gt_file: str,
        test_file: str
    ):
        super().__init__()
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.gt_file = gt_file
        self.test_file = test_file
        self.add_state("predictions", default=[], dist_reduce_fx="cat")

    def update(self, preds, targets=None):
        '''
            preds: list of dict
            {
                'img_name': list[str],
                'boxes': torch.Tensor(N, 4),
                'scores': torch.Tensor(N),
                'labels': torch.Tensor(N),
            }
        '''
        self.predictions.extend(preds)

    def compute(self):
        # load gt action tubes
        video_testlist = read_test_split(self.test_file)
        gts = read_tube_gt(
            video_testlist=video_testlist,
            gt_file=self.gt_file,
            device=self.device
        )

        # process predictions
        detected_boxes = {}
        for pred in self.predictions:
            img_name = pred['img_name']
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']

            img_annotation = {}
            for cls_idx in range(self.num_classes):
                inds = torch.where(labels == cls_idx)[0]
                c_bboxes = boxes[inds]
                c_scores = scores[inds]
                # [n_box, 5]
                tboxes = torch.concat(
                    [c_bboxes, c_scores[..., None]], axis=-1)
                img_annotation[cls_idx+1] = tboxes
            detected_boxes[img_name] = img_annotation

        result = {}

        for iou_th in self.iou_thresholds:
            video_map = evaluate_videoAP(
                gt_videos=gts,
                all_boxes=detected_boxes,
                num_classes=24,
                iou_thresh=iou_th,
                bTemporal=True,
                device=self.device
            )
            result[f'map_{int(iou_th*100)}'] = video_map
        return result


def read_tube_gt(video_testlist, gt_file: str, device):
    gt_videos = {}

    gt_data = loadmat(file_name=gt_file)['annot']
    # print(gt_data[0][0][2][0])
    n_videos = gt_data.shape[1]
    print('loading gt tubes ...')
    for i in range(n_videos):
        video_name = gt_data[0][i][1][0]
        if video_name in video_testlist:
            n_tubes = len(gt_data[0][i][2][0])
            v_annotation = {}
            all_gt_boxes = []
            for j in range(n_tubes):
                gt_one_tube = []
                tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                tube_class = gt_data[0][i][2][0][j][2][0][0]
                tube_data = gt_data[0][i][2][0][j][3]
                tube_length = tube_end_frame - tube_start_frame + 1

                for k in range(tube_length):
                    gt_boxes = []
                    gt_boxes.append(
                        int(tube_start_frame.astype(np.uint16)+k))
                    gt_boxes.append(float(tube_data[k][0]))
                    gt_boxes.append(float(tube_data[k][1]))
                    gt_boxes.append(
                        float(tube_data[k][0]) + float(tube_data[k][2]))
                    gt_boxes.append(
                        float(tube_data[k][1]) + float(tube_data[k][3]))
                    gt_one_tube.append(gt_boxes)
                # print(f"tube {j} len = {len(gt_one_tube)}")
                all_gt_boxes.append(torch.tensor(
                    gt_one_tube, dtype=torch.float32).to(device))

            # print(f"video {video_name} len = {len(all_gt_boxes)}")

            v_annotation['gt_classes'] = torch.tensor(
                tube_class, device=device)
            v_annotation['tubes'] = all_gt_boxes
            gt_videos[str(video_name)] = v_annotation

    return gt_videos


def read_test_split(test_file: str):
    video_testlist = []
    with open(test_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()
            video_testlist.append(line)

    return video_testlist

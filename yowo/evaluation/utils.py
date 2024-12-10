from torchmetrics.functional.detection.iou import intersection_over_union


class AveragePrecision:
    def __init__(self, iou_thresholds, metric_iou_thresholds, metric_rec_thresholds, metric_max_detection_thresholds):
        self.iou_thresholds = iou_thresholds
        self.metric_iou_thresholds = metric_iou_thresholds
        self.metric_rec_thresholds = metric_rec_thresholds
        self.metric_max_detection_thresholds = metric_max_detection_thresholds

        self.gts = []
        self.preds = []
        self.ious = []

    def __call__(self, gts, preds):
        pass

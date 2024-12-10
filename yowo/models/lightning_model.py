from dataclasses import asdict
from typing import Any, Literal, Mapping

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from yowo.utils.box_ops import rescale_bboxes_tensor
from .yowov2.model import YOWO
from .yowov2.loss import build_criterion
from yowo.evaluation.calculate_frame_map import BBType, BoundingBox, BoundingBoxes, CoordinatesType, BBFormat, Evaluator, MethodAveragePrecision

from .schemas import (
    LossConfig,
    ModelConfig,
    LRSChedulerConfig
)


class YOWOv2Lightning(LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        optimizer: OptimizerCallable,
        scheduler_config: LRSChedulerConfig,
        warmup_config: LRSChedulerConfig | None,
        freeze_backbone_2d: bool = True,
        freeze_backbone_3d: bool = True,
        metric_iou_thresholds: list[float] | None = [0.25, 0.5, 0.75, 0.95],
        metric_rec_thresholds: list[float] | None = [0.1, 0.3, 0.5, 0.7, 0.9],
        metric_max_detection_thresholds: list[int] | None = [1, 10, 100]
    ) -> None:
        """
        Initializes the YOWOv2Lightning model with the provided configurations.

        Args:
            model_config (ModelConfig): Configuration for the model.
            loss_config (LossConfig): Configuration for the loss function.
            optimizer (OptimizerCallable): The optimizer used for training.
            scheduler_config (LRSChedulerConfig): Configuration for the learning rate scheduler.
            warmup_config (LRSChedulerConfig | None): Configuration for the warmup scheduler if provided.
            freeze_backbone_2d (bool): Whether to freeze the 2D backbone.
            freeze_backbone_3d (bool): Whether to freeze the 3D backbone.
            metric_iou_thresholds (list[float] | None): IoU thresholds for metrics.
            metric_rec_thresholds (list[float] | None): Recall thresholds for Mean Average Recall.
            metric_max_detection_thresholds (list[int] | None): Thresholds for maximum detections.

        Returns:
            None
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.scheduler_config = scheduler_config
        self.warmup_config = warmup_config
        self.num_classes = model_config.num_classes
        self.model = YOWO(model_config)

        if freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in self.model.backbone_2d.parameters():
                m.requires_grad = False
        if freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in self.model.backbone_3d.parameters():
                m.requires_grad = False

        self.criterion = build_criterion(
            img_size=model_config.img_size,
            num_classes=model_config.num_classes,
            multi_hot=model_config.multi_hot,
            loss_cls_weight=loss_config.loss_cls_weight,
            loss_reg_weight=loss_config.loss_reg_weight,
            loss_conf_weight=loss_config.loss_conf_weight,
            focal_loss=loss_config.focal_loss,
            center_sampling_radius=loss_config.center_sampling_radius,
            topk_candicate=loss_config.topk_candicate
        )

        self.all_boxes_val = BoundingBoxes()
        self.all_boxes_test = BoundingBoxes()
        self.evaluator = Evaluator(dataset="ucf24")

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, video_clip: torch.Tensor, infer_mode=True):
        return self.model.inference(video_clip) if infer_mode else self.model(video_clip)

    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        frame_ids, video_clips, targets = batch
        batch_size = video_clips.size(0)
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        outputs = self.forward(video_clips, infer_mode=False)
        loss_dict = self.criterion(outputs, targets)
        total_loss = loss_dict['losses']

        out_log = {
            "lr": lr,
            "total_loss": total_loss,
            "loss_conf": loss_dict["loss_conf"],
            "loss_cls": loss_dict["loss_cls"],
            "loss_box": loss_dict["loss_box"]
        }

        _sync_dist_log = self.trainer.world_size > 1 or self.trainer.num_devices > 1

        self.log_dict(
            dictionary=out_log,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=_sync_dist_log,
            batch_size=batch_size
        )
        return total_loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        self.eval_step(batch, mode="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        self.eval_step(batch, mode="test")

    def eval_step(self, batch, mode: Literal["val", "test"]):
        batch_img_name, batch_video_clip, batch_target = batch
        batch_scores, batch_labels, batch_bboxes = self.forward(
            batch_video_clip, infer_mode=True)

        # ground truth
        for i in range(len(batch_img_name)):
            img_size = batch_target[i]['orig_size']
            img_name = batch_img_name[i]
            boxes = rescale_bboxes_tensor(
                bboxes=batch_target[i]['boxes'],
                dest_width=img_size[0],
                dest_height=img_size[1]
            )
            labels = batch_target[i]['labels']
            for box, label in zip(boxes, labels):
                bb = BoundingBox(
                    img_name,
                    x=box[0].item(),
                    y=box[1].item(),
                    w=box[2].item(),
                    h=box[3].item(),
                    classId=label,
                    imgSize=img_size,
                    bbType=BBType.GroundTruth,
                    typeCoordinates=CoordinatesType.Absolute,
                    format=BBFormat.XYX2Y2
                )
                if mode == "val":
                    self.all_boxes_val.addBoundingBox(bb)
                elif mode == "test":
                    self.all_boxes_test.addBoundingBox(bb)

        # predict
        for i in range(len(batch_img_name)):
            img_size = batch_target[i]['orig_size']
            img_name = batch_img_name[i]
            boxes = rescale_bboxes_tensor(
                bboxes=batch_bboxes[i],
                dest_width=img_size[0],
                dest_height=img_size[1]
            )
            scores = batch_scores[i]
            labels = batch_labels[i]
            for box, score, label in zip(boxes, scores, labels):
                bb = BoundingBox(
                    img_name,
                    x=box[0].item(),
                    y=box[1].item(),
                    w=box[2].item(),
                    h=box[3].item(),
                    classId=label + 1,
                    imgSize=img_size,
                    bbType=BBType.Detected,
                    typeCoordinates=CoordinatesType.Absolute,
                    classConfidence=score.item(),
                    format=BBFormat.XYX2Y2
                )
                if mode == "val":
                    self.all_boxes_val.addBoundingBox(bb)
                elif mode == "test":
                    self.all_boxes_test.addBoundingBox(bb)

    def eval_epoch(self, mode: Literal["val", "test"]):
        if mode == "val":
            detections = self.evaluator.PlotPrecisionRecallCurve(
                boundingBoxes=self.all_boxes_val,
                IOUThreshold=0.5,
                method=MethodAveragePrecision.EveryPointInterpolation,
                showGraphic=False
            )
            result = self.val_metric.compute()
        else:
            detections = self.evaluator.PlotPrecisionRecallCurve(
                boundingBoxes=self.all_boxes_test,
                IOUThreshold=0.5,
                method=MethodAveragePrecision.EveryPointInterpolation,
                showGraphic=False
            )

        # AP_res = []
        acc_AP = 0
        validClasses = 0
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']

            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                prec = ['%.2f' % p for p in precision]
                rec = ['%.2f' % r for r in recall]
                ap_str = "{0:.2f}%".format(ap * 100)
                # AP_res.append('AP: %s (%s)' % (ap_str, cl))

        mAP = acc_AP / validClasses
        # AP_res.append('mAP: %s' % mAP_str)

        _sync_dist_log = self.trainer.world_size > 1 and self.trainer.num_devices > 1
        if _sync_dist_log:
            metrics = {
                k: v.to(self._device)
                for k, v in metrics.items() if isinstance(v, torch.Tensor)
            }

        self.log(
            name=f"frame_map",
            value=torch.tensor(mAP),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            sync_dist=_sync_dist_log
        )

    def on_validation_epoch_end(self) -> None:
        self.eval_epoch("val")
        self.val_metric.reset()

    def on_test_epoch_end(self) -> None:
        self.eval_epoch("test")
        self.test_metric.reset()

    def build_scheduler(self, config: LRSChedulerConfig, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        config_dict = asdict(config)
        config_dict['scheduler'] = config.scheduler(optimizer)
        return config_dict

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters())

        scheduler_dict = self.build_scheduler(
            config=self.scheduler_config,
            optimizer=optimizer
        )

        schedulers = [scheduler_dict]

        if self.warmup_config is not None:
            schedulers.append(self.build_scheduler(
                config=self.warmup_config,
                optimizer=optimizer
            ))

        return [optimizer], schedulers

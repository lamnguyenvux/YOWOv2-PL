from dataclasses import asdict
from typing import Any, Literal, Mapping

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import OptimizerCallable
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from yowo.utils.box_ops import rescale_bboxes_tensor
from .yowov2.model import YOWO
from .yowov2.loss import build_criterion

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

        self.include_metric_res = [
            f"mar_{n_det}" for n_det in metric_max_detection_thresholds]
        self.include_metric_res.extend(["map", "map_50", "map_75"])

        self.val_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=metric_iou_thresholds,
            rec_thresholds=metric_rec_thresholds,
            max_detection_thresholds=metric_max_detection_thresholds,
            average="macro"
        )
        self.test_metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=metric_iou_thresholds,
            rec_thresholds=metric_rec_thresholds,
            max_detection_thresholds=metric_max_detection_thresholds,
            average="macro"
        )

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.img_size = model_config.img_size
        self.multihot = model_config.multi_hot

    def forward(self, video_clip: torch.Tensor):
        x = self.model(video_clip)
        return x

    def post_processing(self, outputs):
        return self.model.post_processing(outputs)

    def inference(self, video_clips: torch.Tensor) -> list[torch.Tensor]:
        return self.post_processing(
            self.forward(video_clips))

    def training_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        frame_ids, video_clips, targets = batch
        batch_size = video_clips.size(0)
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        outputs = self.forward(video_clips)
        loss_dict = self.criterion(outputs, targets)
        total_loss = loss_dict['losses']

        out_log = {
            "lr": lr,
            "total_loss": total_loss,
            "loss_conf": loss_dict["loss_conf"],
            "loss_cls": loss_dict["loss_cls"],
            "loss_box": loss_dict["loss_box"]
        }

        self.log_dict(
            dictionary=out_log,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=True,
            batch_size=batch_size
        )
        return total_loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        self.eval_step(batch, mode="val")

    def test_step(self, batch, batch_idx) -> torch.Tensor | Mapping[str, Any] | None:
        self.eval_step(batch, mode="test")

    def eval_step(self, batch, mode: Literal["val", "test"]):
        batch_img_name, batch_video_clip, batch_target = batch
        outputs = self.inference(batch_video_clip)

        # process batch gt
        gts = list(map(
            lambda x: {
                "boxes": rescale_bboxes_tensor(
                    bboxes=x["boxes"],
                    dest_width=x["orig_size"][0],
                    dest_height=x["orig_size"][1]
                ),
                "labels": x["labels"].long(),
            },
            batch_target
        ))

        # process batch predict
        preds = []
        for idx, output in enumerate(outputs):
            pred = {
                "boxes": rescale_bboxes_tensor(
                    bboxes=output[:, :4],
                    dest_width=batch_target[idx]["orig_size"][0],
                    dest_height=batch_target[idx]["orig_size"][1]
                ),
                "scores": output[:, 4],
                # int64
                "labels": output[:, 5:].long() if self.multihot else output[:, 5].long(),
            }
            preds.append(pred)

        if mode == "val":
            self.val_metric.update(preds, gts)
        else:
            self.test_metric.update(preds, gts)

    def eval_epoch(self, mode: Literal["val", "test"]):
        if mode == "val":
            result = self.val_metric.compute()
        else:
            result = self.test_metric.compute()

        metrics = {
            k: v for k, v in result.items() if k in self.include_metric_res
        }

        metrics = {
            k: v.to(self._device)
            for k, v in metrics.items() if isinstance(v, torch.Tensor)
        }

        self.log_dict(
            dictionary=metrics,
            prog_bar=False,
            logger=True,
            on_epoch=True,
            sync_dist=self.trainer.num_devices > 1
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


class YOWOv2PP(YOWOv2Lightning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multihot = kwargs.get("model_config", False).multi_hot

    def forward(self, video_clip, conf_threshold):
        x = super().post_processing(super().forward(video_clip))
        outputs = []
        for output in x:
            if not self.multihot:
                mask = output[..., 4] > conf_threshold
                filtered_indices = mask.nonzero(as_tuple=True)
                result = output[filtered_indices]

            else:
                # Extract confidence scores and class probabilities
                confidence_scores = output[..., 4]  # Shape (B, N)
                class_probabilities = output[..., 5:]  # Shape (B, N, 80)

                # Calculate class scores
                class_scores = torch.sqrt(
                    confidence_scores.unsqueeze(-1) * class_probabilities)  # Shape (N, 80)
                # Create a mask for scores above the threshold
                mask = class_scores > conf_threshold  # Shape (N, 80)
                keep_instance = mask.any(dim=-1)  # Shape (N,)
                bboxes = output[keep_instance, :4]
                cls_scores = class_scores[keep_instance, :]
                result = torch.cat([bboxes, cls_scores], dim=-1)
            outputs.append(result)

        return outputs

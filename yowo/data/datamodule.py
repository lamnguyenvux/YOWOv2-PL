from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS
)
from torch import device
from torch._C import device
from torch.utils.data import DataLoader, random_split
import os
from typing import Callable, Iterable, Union, Literal, Optional, Any, Dict
from dataclasses import dataclass

from .dataset.ava import AVA_Dataset
from .dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from .dataset.transforms import Augmentation, BaseTransform
from .utils import collate_fn
from yowo.utils.validate import validate_literal_types


@dataclass(frozen=True)
class AugmentationParams:
    jitter: float = 0.2
    hue: float = 0.1
    saturation: float = 1.5
    exposure: float = 1.5


DEFAULT_SPLIT_FILE = dict(
    train="trainlist.txt",
    test="testlist.txt"
)

DATASET = Literal['ucf24', 'jhmdb21']

BATCH_SIZE_SETTING = {
    "train": 64,
    "val": 32,
    "test": 64
}


class UCF24_JHMDB21_DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: DATASET,
        data_dir: str,
        aug_params: AugmentationParams,
        collate_fn: Optional[Callable[[Iterable], Any]] = None,
        split_file: dict = DEFAULT_SPLIT_FILE,
        num_workers: Union[Literal["auto"], int] = "auto",
        img_size: int = 224,
        len_clip: int = 16,
        sampling_rate: int = 1,
        batch_size: Dict[str, int] = BATCH_SIZE_SETTING,
    ):
        """Lightning Data Module for UCF24 and JHMDB21 dataset

        Args:
            dataset (DATASET): type of dataset. "ucf24" or "jhmdb21"
            data_dir (str): path to dataset directory
            aug_params (AugmentationParams): argumentation parameters
            split_file (Optional[SplitFile], optional): a dataclass that contain 2 field "train" and "test", each is a split file for train/test. Defaults to None.
            num_workers (Union[Literal["auto"], int], optional): number of workers for Dataloader. Defaults to 0.
            img_size (int, optional): size of input data. Defaults to 224.
            len_clip (int, optional): number of sequence of frames for input data. Defaults to 16.
            sampling_rate (int, optional): sampling rate to create sequence of frames. Defaults to 1.
            batch_size (int, optional): batch size. Defaults to 8.
            collate_fn (Optional[Any], optional): collate function for Dataloader. Defaults to None.
        """
        super().__init__()
        self.dataset = dataset
        self.data_dir = data_dir
        self.aug_params = aug_params
        self.split_file = split_file
        self.num_workers = os.cpu_count() if num_workers == "auto" else num_workers
        # self.transform = transform
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def prepare_data(self) -> None:
        validate_literal_types(self.dataset, DATASET)
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"""Data diretory {self.data_dir} doesn't exist."""
            )

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']) -> None:
        if self.collate_fn is None:
            self.collate_fn = collate_fn

        self.train_tfms = Augmentation(
            img_size=self.img_size,
            jitter=self.aug_params.jitter,
            hue=self.aug_params.hue,
            saturation=self.aug_params.saturation,
            exposure=self.aug_params.exposure
        )
        self.train_set = UCF_JHMDB_Dataset(
            data_root=self.data_dir,
            dataset=self.dataset,
            split_list=self.split_file.get("train", None),
            img_size=self.img_size,
            transform=self.train_tfms,
            is_train=True,
            len_clip=self.len_clip,
            sampling_rate=self.sampling_rate
        )

        self.val_test_tfms = BaseTransform(
            img_size=self.img_size
        )

        self.allset = UCF_JHMDB_Dataset(
            data_root=self.data_dir,
            dataset=self.dataset,
            split_list=self.split_file.get("test", None),
            img_size=self.img_size,
            transform=self.val_test_tfms,
            is_train=False,
            len_clip=self.len_clip,
            sampling_rate=self.sampling_rate
        )

        self.val_set, _ = random_split(
            dataset=self.allset,
            lengths=[0.3, 0.7]
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size.get("train", 64),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size.get("val", 32),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.allset,
            batch_size=self.batch_size.get("test", 64),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    def transfer_batch_to_device(self, batch: TRAIN_DATALOADERS, device: device, dataloader_idx: int) -> TRAIN_DATALOADERS:
        return super().transfer_batch_to_device(batch, device, dataloader_idx)


class AVADataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        aug_params: AugmentationParams,
        frames_dir: str = 'frames_dir',
        frame_list: str = 'frame_list',
        annotation_dir: str = 'annotation_dir',
        labelmap_file: str = 'labelmap_file',
        batch_size: int = 8,
        collate_fn: Optional[Any] = None,
        gt_box_list: Optional[str] = None,
        exclusion_file: Optional[str] = None,
        img_size: int = 224,
        len_clip: int = 16,
        sampling_rate: int = 1
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.aug_params = aug_params
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.gt_box_list = gt_box_list
        self.exclusion_file = exclusion_file
        self.frames_dir = frames_dir
        self.frame_list = frame_list
        self.annotation_dir = annotation_dir
        self.labelmap_file = labelmap_file
        self.img_size = img_size
        # self.transform = transform
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.tfms = Augmentation(
                img_size=self.img_size,
                jitter=self.aug_params.jitter,
                hue=self.aug_params.hue,
                saturation=self.aug_params.saturation,
                exposure=self.aug_params.exposure
            )

            self.train_set = AVA_Dataset(
                data_root=self.data_root,
                gt_box_list=self.gt_box_list,
                exclusion_file=self.exclusion_file,
                frames_dir=self.frames_dir,
                frame_list=self.frame_list,
                annotation_dir=self.annotation_dir,
                labelmap_file=self.labelmap_file,
                is_train=True,
                img_size=self.img_size,
                transform=self.tfms,
                len_clip=self.len_clip,
                sampling_rate=self.sampling_rate
            )

        if stage == "test":
            self.tfms = BaseTransform(
                img_size=self.img_size
            )

            self.test_set = AVA_Dataset(
                data_root=self.data_root,
                gt_box_list=self.gt_box_list,
                exclusion_file=self.exclusion_file,
                frames_dir=self.frames_dir,
                frame_list=self.frame_list,
                annotation_dir=self.annotation_dir,
                labelmap_file=self.labelmap_file,
                is_train=False,
                img_size=self.img_size,
                transform=self.tfms,
                len_clip=self.len_clip,
                sampling_rate=self.sampling_rate
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=False
        )

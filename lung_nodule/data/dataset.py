
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import pandas as pd

from lung_nodule.config import config
from lung_nodule.data.transforms import (
    extract_patch,
    clip_and_scale,
    worker_init_fn,
)


class CTCaseDataset(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset: pd.DataFrame,
        translations: bool = None,
        rotations: tuple = None,
        size_px: int = 64,
        size_mm: int = 50,
        mode: str = "2D",
        config=None,
        use_monai_transforms: str="val"
    ):
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.rotations = rotations
        self.translations = translations
        self.size_px = size_px
        self.size_mm = size_mm
        self.mode = mode
        self.config = config
        self.use_monai_transforms = use_monai_transforms

    def __getitem__(self, idx):
        pd = self.dataset.iloc[idx]
        label = pd.label
        annotation_id = pd.AnnotationID

        image_path = self.data_dir / "image" / f"{annotation_id}.npy"
        metadata_path = self.data_dir / "metadata" / f"{annotation_id}.npy"

        img = np.load(image_path, mmap_mode="r")
        metadata = np.load(metadata_path, allow_pickle=True).item()

        origin = metadata["origin"]
        spacing = metadata["spacing"]
        transform = metadata["transform"]

        translations = None
        translation_radius = 2.5
        if self.translations == True:
            translation_radius = getattr(self.config, 'TRANSLATION_RADIUS', 2.5)
            translations = translation_radius if translation_radius > 0 else None

        if self.mode == "2D":
            output_shape = (1, self.size_px, self.size_px)
        else:
            output_shape = (self.size_px, self.size_px, self.size_px)

        # Extract patch with augmentations
        patch = extract_patch(
            CTData=img,
            coord=tuple(np.array([64, 128, 128]) // 2),
            srcVoxelOrigin=origin,
            srcWorldMatrix=transform,
            srcVoxelSpacing=spacing,
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            rotations=self.rotations,
            translations=translations,
            translation_radius=translation_radius,
            coord_space_world=False,
            mode=self.mode,
            config=self.config,
        )

        patch = patch.astype(np.float32)
        # patch = self.transforms(patch)

        patch = clip_and_scale(patch)

        target = torch.ones((1,)) * label

        sample = {
            # "image": patch,
            "image": torch.from_numpy(patch),
            "label": target.long(),
            "ID": annotation_id,
        }

        return sample

    def __len__(self):
        return len(self.dataset)


def get_data_loader(
    data_dir,
    dataset,
    mode="2D",
    sampler=None,
    workers=0,
    batch_size=64,
    size_px=64,
    size_mm=70,
    rotations=None,
    translations=None,
    config=None,
    use_monai_transforms="val",
):
    data_set = CTCaseDataset(
        data_dir=data_dir,
        translations=translations,
        dataset=dataset,
        rotations=rotations,
        size_mm=size_mm,
        size_px=size_px,
        mode=mode,
        config=config,
        use_monai_transforms=use_monai_transforms,
    )

    shuffle = False
    if sampler == None:
        shuffle = True

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )

    return data_loader

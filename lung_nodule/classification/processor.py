
import numpy as np
from lung_nodule import data as dataloader
import torch
import torch.nn as nn
from torchvision import models
from lung_nodule.models.model_3d import I3D
from lung_nodule.models.model_2d import (ResNet18, ResNet50, ResNet101, ResNet152,
                            ConvNeXtBase, ConvNeXtSmall, ConvNeXtLarge)
import os
import math
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s][%(asctime)s] %(message)s",
    datefmt="%I:%M:%S",
)

# define processor
class MalignancyProcessor:
    def __init__(self, mode="2D", suppress_logs=False, model_name="LUNA25-baseline-2D"):

        self.size_px = 64
        self.size_mm = 50

        self.model_name = model_name
        self.mode = mode
        self.suppress_logs = suppress_logs

        if not self.suppress_logs:
            logging.info("Initializing the deep learning system")

        if self.mode == "2D":
            # self.model_2d = ResNet152(weights=None).cuda()
            self.model_2d = ConvNeXtBase(num_classes=1, pretrained=False).cuda()
        elif self.mode == "3D":
            self.model_3d = I3D(num_classes=1, pre_trained=False, input_channels=3).cuda()

        self.model_root = str(Path(__file__).parent.parent.parent / "weights")

    def define_inputs(self, image, header, coords):
        self.image = image
        self.header = header
        self.coords = coords

    def extract_patch(self, coord, output_shape, mode):

        patch = dataloader.extract_patch(
            CTData=self.image,
            coord=coord,
            srcVoxelOrigin=self.header["origin"],
            srcWorldMatrix=self.header["transform"],
            srcVoxelSpacing=self.header["spacing"],
            output_shape=output_shape,
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            coord_space_world=True,
            mode=mode,
        )

        # ensure same datatype...
        patch = patch.astype(np.float32)

        # clip and scale...
        patch = dataloader.clip_and_scale(patch)
        return patch

    def _process_model(self, mode):

        if not self.suppress_logs:
            logging.info("Processing in " + mode)

        if mode == "2D":
            output_shape = [1, self.size_px, self.size_px]
            model = self.model_2d
        else:
            output_shape = [self.size_px, self.size_px, self.size_px]
            model = self.model_3d

        nodules = []

        for _coord in self.coords:

            patch = self.extract_patch(_coord, output_shape, mode=mode)
            nodules.append(patch)

        nodules = np.array(nodules)
        nodules = torch.from_numpy(nodules).cuda()

        ckpt = torch.load(
            os.path.join(
                self.model_root,
                self.model_name,
                "best_metric_model.pth",
            )
        )
        model.load_state_dict(ckpt)
        model.eval()
        logits = model(nodules)
        logits = logits.data.cpu().numpy()

        logits = np.array(logits)
        return logits

    def predict(self):

        logits = self._process_model(self.mode)

        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return probability, logits

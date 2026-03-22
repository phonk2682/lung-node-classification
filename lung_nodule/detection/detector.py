import torch
from pathlib import Path

from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    DeleteItemsd,
)

from monai.apps.detection.transforms.dictionary import (
        ClipBoxToImaged,
        AffineBoxToWorldCoordinated,
        ConvertBoxModed,
    )

# from configs import *

device = "cpu" if not torch.cuda.is_available() else "cuda"

# Detector configs
dt_model_path = str(Path(__file__).parent.parent.parent / "weights" / "dt_model.ts")
spatial_dims = 3
num_classes = 1
size_divisible = [16, 16, 8]
infer_patch_size = [512, 512, 192]
feature_map_scales = [1, 2, 4]
base_anchor_shapes = [
    [6, 8, 4],
    [8, 6, 5],
    [10, 10, 6],
]
box_key = "box"
label_key = "label"
score_thresh = 0.02
topk_candidates_per_level = 1000
nms_thresh = 0.22
detections_per_img = 300
overlap = 0.25
sw_batch_size = 1
mode = "constant"
pixdim = [0.703125, 0.703125, 1.25]
a_min=-1024.0
a_max=300.0
b_min=0.0
b_max=1.0
clip=True
score_keep = 0.3


def build_preprocess(image_key="image"):
    keys = [image_key]
    transforms = [
        LoadImaged(keys=keys, reader="itkreader", affine_lps_to_ras=True),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd(keys=keys, pixdim=pixdim, mode="bilinear", padding_mode="border"),
        ScaleIntensityRanged(
            keys=keys,
            a_min=a_min, a_max=a_max,
            b_min=b_min, b_max=b_max,
            clip=clip
        ),
        EnsureTyped(keys=keys),
    ]
    return Compose(transforms)


def build_postprocess(image_key: str = "image", affine_lps_to_ras: bool = True):
    return Compose(
        [
            ClipBoxToImaged(
                box_keys="box",
                label_keys="label",
                box_ref_image_keys=image_key,
                remove_empty=True,
            ),
            AffineBoxToWorldCoordinated(
                box_keys="box",
                box_ref_image_keys=image_key,
                affine_lps_to_ras=affine_lps_to_ras,
            ),
            ConvertBoxModed(
                box_keys="box",
                src_mode="xyzxyz",
                dst_mode="cccwhd",
            ),
            DeleteItemsd(keys=[image_key]),
        ]
    )


def build_detector(model_path, device):
    network = torch.jit.load(model_path, map_location=device)

    anchor_generator = AnchorGeneratorWithAnchorShape(
        feature_map_scales=[1, 2, 4],
        base_anchor_shapes=[
            [6, 8, 4],
            [8, 6, 5],
            [10, 10, 6],
        ],
    )
    detector = RetinaNetDetector(
        network=network,
        anchor_generator=anchor_generator,
        spatial_dims=spatial_dims,
        num_classes=num_classes,
        size_divisible=size_divisible,
    )
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=score_thresh,
        topk_candidates_per_level=topk_candidates_per_level,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )
    detector.set_sliding_window_inferer(
        roi_size=infer_patch_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        mode=mode,
        device=device,
    )
    detector.eval()

    return detector

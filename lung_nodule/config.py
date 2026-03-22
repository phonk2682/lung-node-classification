
from pathlib import Path


class Configuration(object):
    def __init__(self) -> None:

        # Working directory (relative to this file)
        self.WORKDIR = Path(__file__).parent
        self.RESOURCES = self.WORKDIR / "resources"
        # Starting weights for the I3D model (only needed for 3D training)
        self.MODEL_RGB_I3D = self.RESOURCES / "model_rgb.pth"

        # Data parameters — overridden by train.py CLI arguments
        self.CSV_DIR = Path("./data/")
        self.CSV_DIR_TRAIN = Path("./data/data_fold_1_train.csv")
        self.CSV_DIR_VALID = Path("./data/data_fold_1_test.csv")

        # Results directory — overridden by train.py CLI arguments
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        self.EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

        self.EXPERIMENT_NAME = "ResNet152-confirm-2"
        self.MODE = "2D" # 2D or 3D
        self.MODEL_NAME = "ResNet152"
        # Training parameters
        self.SEED = 2025
        self.NUM_WORKERS = 8
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 64
        # self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        # self.TRANSLATION = True

        self.ROTATION = ((-30, 30), (-30, 30), (-30, 30))
        self.TRANSLATION = True
        self.TRANSLATION_RADIUS = 3.5

        self.INTENSITY_SHIFT = True
        self.INTENSITY_SHIFT_RANGE = (-0.1, 0.1)

        self.INTENSITY_SCALE = True
        self.INTENSITY_SCALE_RANGE = (0.9, 1.1)

        self.GAUSSIAN_NOISE = True
        self.GAUSSIAN_NOISE_STD = 0.02

        self.GAMMA_CORRECTION = False
        self.GAMMA_RANGE = (0.88, 1.12)

        self.CONTRAST_ADJUSTMENT = False
        self.CONTRAST_RANGE = (0.92, 1.08)

        # CLIP_HU_MIN = -1000.0
        # CLIP_HU_MAX = 400.0
        # CLIP_MARGIN = 0.0

        self.EPOCHS = 200
        self.PATIENCE = 50
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 5e-4


config = Configuration()

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import os
import platform
import torch

setup_logger()

# Number of Tangram shapes
TANGRAM_NUM = 7

if __name__ == "__main__":
    # Register the datasets
    register_coco_instances("tangram_train", {}, "dataset/train.json", "dataset")
    register_coco_instances("tangram_test", {}, "dataset/test.json", "dataset")

    # Configure the model
    cfg = get_cfg()

    # Load the base configuration for Mask R-CNN
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Dataset settings
    cfg.DATASETS.TRAIN = ("tangram_train",)
    cfg.DATASETS.TEST = ("tangram_test",)

    # DataLoader settings
    cfg.DATALOADER.NUM_WORKERS = 2

    # Pre-trained weights from COCO
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    # Solver settings
    cfg.SOLVER.IMS_PER_BATCH = 2  # Adjust based on GPU memory
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = 1000  # Number of iterations

    # ROI head settings
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = TANGRAM_NUM  # Ensure this matches dataset categories

    # Platform-specific settings
    if platform.system() == "Windows":
        cfg.MODEL.DEVICE = "cuda"  # Use CUDA for GPU acceleration on Windows
    elif platform.system() == "Darwin":
        cfg.MODEL.DEVICE = "mps"  # Use Metal Performance Shaders on macOS
    else:
        cfg.MODEL.DEVICE = "cpu"  # Default to CPU for other platforms

    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Build the model and reset the ROI heads
    model = build_model(cfg)
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(1024, TANGRAM_NUM + 1)  # Classes + background
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(1024, TANGRAM_NUM * 4)  # 4 bbox coordinates per class
    model.roi_heads.mask_head.predictor = torch.nn.Conv2d(256, TANGRAM_NUM, kernel_size=(1, 1))

    # Load pre-trained weights excluding the ROI heads
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # Initialize the trainer
    trainer = DefaultTrainer(cfg)
    trainer.model = model  # Override the default model with our modified one

    # Start training
    trainer.resume_or_load(resume=False)
    trainer.train()

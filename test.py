# test.py

import pytorch_lightning as pl
import argparse
import yaml
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.urban_nav_module import UrbanNavModule
import torch
import glob

# Optional: If you use Wandb for logging during testing
try:
    from pytorch_lightning.loggers import WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

torch.set_float32_matmul_precision('medium')


class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description='Test UrbanNav model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint. If not provided, the latest checkpoint will be used.')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save predictions')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DictNamespace(**cfg_dict)
    return cfg


def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint in the given directory based on modification time.
    
    Args:
        checkpoint_dir (str): Path to the directory containing checkpoints.
    
    Returns:
        str: Path to the latest checkpoint file.
    
    Raises:
        FileNotFoundError: If no checkpoint files are found in the directory.
    """
    checkpoint_pattern = os.path.join(checkpoint_dir, '*.ckpt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
    
    # Sort checkpoints by modification time (latest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    return latest_checkpoint


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Create a directory for test results
    test_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # Initialize the DataModule for testing
    datamodule = CityWalkDataModule(cfg)

    # Initialize the model
    model = UrbanNavModule(cfg)

    # Determine the checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    else:
        # Automatically find the latest checkpoint
        checkpoint_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name, 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        print(f"No checkpoint specified. Using the latest checkpoint: {checkpoint_path}")

    # Load the model from the checkpoint
    model = UrbanNavModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    model.result_dir = test_dir
    print(f"Loaded model from checkpoint: {checkpoint_path}")

    # Define callbacks
    callbacks = [
        pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate),
    ]

    # Initialize Trainer
    trainer = pl.Trainer(
        default_root_dir=test_dir,
        devices=cfg.training.gpus,
        precision='16-mixed' if cfg.training.amp else 32,
        accelerator='ddp' if cfg.training.gpus > 1 else 'gpu',
        callbacks=callbacks,
        log_every_n_steps=1,
        # You can add more Trainer arguments if needed
    )

    # Run testing
    trainer.test(model, datamodule=datamodule, verbose=True)


if __name__ == '__main__':
    main()
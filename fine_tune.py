# fine_tune.py

import pytorch_lightning as pl
import argparse
import yaml
import os
from pl_modules.citywalk_datamodule import CityWalkDataModule
from pl_modules.urbannav_datamodule import UrbanNavDataModule
from pl_modules.urban_nav_module import UrbanNavModule
from pytorch_lightning.strategies import DDPStrategy
import torch
import glob

torch.set_float32_matmul_precision('medium')
pl.seed_everything(42, workers=True)


class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-Tune UrbanNav model')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint for fine-tuning')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = DictNamespace(**cfg_dict)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Create result directory
    result_dir = os.path.join(cfg.project.result_dir, cfg.project.run_name)
    os.makedirs(result_dir, exist_ok=True)
    cfg.project.result_dir = result_dir  # Update result_dir in cfg

    # Save config file in result directory
    with open(os.path.join(result_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg.__dict__, f)

    # Initialize the DataModule
    if cfg.data.type == 'citywalk':
        datamodule = CityWalkDataModule(cfg)
    elif cfg.data.type == 'urbannav':
        datamodule = UrbanNavDataModule(cfg)
    else:
        raise ValueError(f"Invalid dataset: {cfg.data.type}")

    # Initialize the model from checkpoint
    model = UrbanNavModule.load_from_checkpoint(args.checkpoint, cfg=cfg)
    print(f"Loaded model from checkpoint: {args.checkpoint}")
    print(pl.utilities.model_summary.ModelSummary(model, max_depth=2))

    # Initialize logger
    logger = None  # Default to no logger

    # Check if logging with Wandb is enabled in config
    use_wandb = cfg.logging.enable_wandb

    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger  # Import here to handle ImportError
            wandb_logger = WandbLogger(
                project=cfg.project.name,
                name=cfg.project.run_name,
                save_dir=result_dir
            )
            logger = wandb_logger
            print("WandbLogger initialized.")
        except ImportError:
            print("Wandb is not installed. Skipping Wandb logging.")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(result_dir, 'checkpoints'),
        save_last=True,
        save_top_k=1,
        monitor='val/direction_loss',
    )

    num_gpu = torch.cuda.device_count()

    # Set up Trainer
    trainer = pl.Trainer(
        default_root_dir=result_dir,
        max_epochs=cfg.training.max_epochs,
        logger=logger,  # Pass the logger (WandbLogger or None)
        devices=num_gpu,
        precision='16-mixed' if cfg.training.amp else 32,
        accelerator='gpu' if num_gpu > 0 else 'cpu',
        callbacks=[
            checkpoint_callback,
            pl.callbacks.TQDMProgressBar(refresh_rate=cfg.logging.pbar_rate),
        ],
        log_every_n_steps=1,
        strategy=DDPStrategy(find_unused_parameters=True) if num_gpu > 1 else None,
    )

    # Start fine-tuning
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
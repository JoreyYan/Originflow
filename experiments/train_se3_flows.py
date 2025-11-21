import datetime
import os
import GPUtil
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
print(torch.version.cuda)
print(torch.version.__version__)

from data.pdb_dataloader import sPdbDataModule
from models.flow_module import FlowModule
from experiments import utils as eu
import wandb

from data.regenerate_data_callback import RegenerateDataCallback
from data.collect_pkl import colletdata
def collectpkl():
    # Your function to regenerate the .pkl data file
    colletdata()
    pass




from datetime import timedelta
import wandb




log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')
# torch.autograd._set_static_graph()
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import TensorDataset, DataLoader
import torch


# 尝试使用这个简单的数据加载器进行训练

class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._datamodule: LightningDataModule = sPdbDataModule(self._data_cfg)
        self._model: LightningModule = FlowModule(self._cfg)
        
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.loader.num_workers = 24
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            
            # Save config
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg,allow_val_change=True)


        devices = GPUtil.getAvailable(order='memory', limit = 8)[:self._exp_cfg.num_devices]
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            precision=self._cfg.model.precision,
            gradient_clip_val=1.0,               # 设置梯度裁剪值
            gradient_clip_algorithm='norm' ,     # 使用裁剪范数的方法，另一个选项是'value'，用于裁剪梯度的绝对值
            num_sanity_val_steps=0,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
            # auto_lr_find=True
        )


        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path="../configs", config_name="binder.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()

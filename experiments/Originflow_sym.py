"""DDP inference script."""
import os
import time
import numpy as np
import hydra
import torch
import GPUtil
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from models.flow_module import FlowModule

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

class Sampler:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        #ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        # OmegaConf.set_struct(ckpt_cfg, False)
        # cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name =ckpt_path.replace('.ckpt', '').split('/')[-1]
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._ckpt_name,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f'Saving results to {self._output_dir}')
        config_path = os.path.join(self._output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f)
        log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        # init_FlowModule=FlowModule(cfg)
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,strict=True
        )
        self._flow_module.eval()
        self._flow_module._cfg = self._cfg
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir
        self._flow_module.interpolant._cfg.temp=self._infer_cfg.lam


    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        eval_dataset = eu.LengthDataset(self._samples_cfg)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        trainer = Trainer(
            accelerator="gpu",
            devices=devices,
            precision=32,
        )

        trainer.predict(self._flow_module, dataloaders=dataloader)




@hydra.main(version_base=None, config_path="../configs", config_name="inference_base_ss")
def run_ss(cfg: DictConfig) -> None:

    lam=[1 ]

    file_path = '..//data/alphabeta.txt'

    # 读取文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

        # 解析每一行
        for line in lines:
            # 分割PDB名称和SS列表
            pdb_name, ss_list_str = line.strip().split(' ', 1)

            for l0 in lam:
                cfg.inference.lam=l0
                cfg.pdb_name=pdb_name
                cfg.ss_list_str = ss_list_str

                # Read model checkpoint.
                log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
                start_time = time.time()


                sampler = Sampler(cfg)
                sampler._flow_module.design_task = 'base_ss'
                sampler.run_sampling()
                elapsed_time = time.time() - start_time
                log.info(f'Finished in {elapsed_time:.2f}s')


@hydra.main(version_base=None, config_path="../configs", config_name="inference_sym")
def run(cfg: DictConfig) -> None:

    syms=['C3']

    for sym in syms:
        cfg.inference.sym=sym


        # Read model checkpoint.
        log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
        start_time = time.time()


        sampler = Sampler(cfg)
        sampler._flow_module.design_task='sym'
        sampler.run_sampling()
        elapsed_time = time.time() - start_time
        log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    import torch

    run()

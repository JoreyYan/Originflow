import os
import time
import random
import numpy as np
import hydra
import torch
import GPUtil
import pandas as pd
import tqdm
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from models.flow_module import FlowModule
from data import utils as du
from data.m2 import MotifSampler,MotifSamplerMultiChain
from torch.utils.data import  DataLoader

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
        try:
            ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

            del ckpt_cfg.model
            # Set-up config.
            OmegaConf.set_struct(cfg, False)
            OmegaConf.set_struct(ckpt_cfg, False)
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
        except:
            OmegaConf.set_struct(cfg, False)

        cfg.experiment.checkpointer.dirpath = './'

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
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
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path, strict=True
        )
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir
        self._flow_module.interpolant._cfg.temp = self._infer_cfg.lam
        self._flow_module.interpolant._sample_cfg =self._flow_module._infer_cfg.interpolant.sampling


    def get_total_length(self, total_length_str):
        if '-' in total_length_str:
            start, end = map(int, total_length_str.split('-'))
            return random.randint(start, end)
        else:
            return int(total_length_str)
    def run_sampling(self, pdb_input_list, pkl_folder_path):
        devices = GPUtil.getAvailable(
            order='memory', limit=8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=devices,
            precision='16-mixed',
        )

        # trainer.predict(self._flow_module, dataloaders=dataloader)

        for name, input_str, total_lengths in pdb_input_list:
            ref_pdb = os.path.join(pkl_folder_path, f"{name.lower()}.pkl")
            ref_data = du.read_pkl(ref_pdb)
            domain = name.lower()

            # total_length = self.get_total_length(total_lengths)
            #total_length_start,total_length_stop = int(total_length.split('-')[0]),int(total_length.split('-')[1])

            sampler = MotifSamplerMultiChain(input_str)
            results = sampler.get_results()
            final_output = sampler.get_final_output()
            #print(sampler)

            self._flow_module._output_dir = os.path.join(self._output_dir, f'{domain}__motif_test_Scaffolding')
            os.makedirs(self._flow_module._output_dir, exist_ok=True)
            os.makedirs(self._flow_module._output_dir + '/native/', exist_ok=True)
            os.makedirs(self._flow_module._output_dir + '/motif_masks/', exist_ok=True)


            from help.test_process_mc_input import process_input
            from chroma.data.protein import Protein

            device = f'cuda:{torch.cuda.current_device()}'
            self._flow_module.interpolant.set_device(device)

            positions = np.take(ref_data['atom_positions'], ref_data['modeled_idx'], axis=0)[..., [0, 1, 2, 4], :]
            chain_index = np.take(ref_data['chain_index'], ref_data['modeled_idx'], axis=0)
            aatype = np.take(ref_data['aatype'], ref_data['modeled_idx'], axis=0)
            p = Protein.from_XCS(torch.tensor(positions).unsqueeze(0), torch.tensor(chain_index).unsqueeze(0),
                                 torch.tensor(aatype).unsqueeze(0))
            #p.to_PDB(os.path.join(self._flow_module._output_dir+'/native/', f'{domain}_motif_native.pdb'))

            _, _, _, indices_mask,_ = process_input(final_output, ref_data)
            np.savetxt(os.path.join(self._flow_module._output_dir+ '/motif_masks/', f'{domain}_motif_native_mask.npy'), indices_mask)


            for i in tqdm.tqdm(range(10)):
                # total_length = self.get_total_length(total_lengths)
                sampler = MotifSamplerMultiChain(input_str)
                results = sampler.get_results()
                final_output = sampler.get_final_output()

                init_motif, fixed_mask, aa_motifed, indices_mask,chain_ids = process_input(final_output, ref_data)

                # chain_idx = torch.ones_like(fixed_mask).unsqueeze(0)

                chain_idx = torch.tensor(chain_ids).unsqueeze(0)

                fixed_mask = fixed_mask.unsqueeze(0)
                bbatoms = torch.tensor(init_motif)[..., [0, 1, 2, 4], :].unsqueeze(0).to(device).float()
                sample_length = bbatoms.shape[1]

                X, C, S = self._flow_module.interpolant.hybrid_motif_sample(
                    1,
                    [sample_length],
                    self._flow_module.model,

                    chain_idx=chain_idx.to(device),
                    native_X=bbatoms.to(device),
                    mode='motif',
                    fixed_mask=fixed_mask.to(device),
                )

                # # 提取C=26位置的数据
                # C_position = 26
                # X = X[:, C_position, :, :]
                # C = C[:, C_position]
                # S = S[:, C_position]

                native_aatype = torch.tensor(aa_motifed).unsqueeze(0).to(S.device)
                S = native_aatype * fixed_mask.to(S.device)
                p = Protein.from_XCS(X, C, S)

                # fixed_mask = fixed_mask.unsqueeze(0)[:, C_position].squeeze(0)

                p.to_PDB(os.path.join(self._flow_module._output_dir, f'{domain}_motif_{i}.pdb'))
                np.savetxt(os.path.join(self._flow_module._output_dir+ '/motif_masks/', f'{domain}_motif_{i}_mask.npy'),
                           fixed_mask.cpu().numpy())
                np.savetxt(os.path.join(self._flow_module._output_dir+ '/motif_masks/', f'{domain}_motif_{i}_Chain_mask.npy'),
                           C.squeeze(0).cpu().numpy())


@hydra.main(version_base=None, config_path="../configs", config_name="inference_motif")
def run(cfg: DictConfig) -> None:
    lam = [1]
    for l0 in lam:
        cfg.inference.lam = l0

        # Read model checkpoint.
        log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
        start_time = time.time()

        sampler = Sampler(cfg)

        # Read ref_pdb and input_str from CSV
        csv_path = "../examples/motif_PDB_Input_Data.csv"
        pkl_folder_path = "../examples/rf_pdb_pkl/"
        df = pd.read_csv(csv_path)
        pdb_input_list = df[['Name', 'Input', 'TotalLength']].values.tolist()

        # eval_dataset = eu.CSVDataset(df)
        # dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

        sampler.run_sampling(pdb_input_list, pkl_folder_path)
        elapsed_time = time.time() - start_time
        log.info(f'Finished in {elapsed_time:.2f}s')


if __name__ == '__main__':
    run()
